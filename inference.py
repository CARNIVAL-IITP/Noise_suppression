from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
import argparse, data, json, models, numpy as np, os, time, torch
import glob, librosa
# from data.feature_utils import get_istft
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
from itertools import permutations
from pypesq import pesq
import csv



def get_free_gpu():

    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


class trainer:
    def __init__(self, args):
        self.model_name = args.model_name
        self.loss_name = args.loss_option
        self.dataset = args.dataset
        
        if args.cuda_option == "True":
            print("GPU mode on...")
            available_device = get_free_gpu()
            print("We found an available GPU: %d!"%available_device)
            self.device = torch.device('cuda:%d'%available_device)
        else:
            self.device = torch.device('cpu')
        self.device = torch.device('cuda:2')

        # build model
        self.model = self.init_model(args.feature_options)
        print("Loaded the model...")
        # build loss fn
        self.loss_fn = self.build_lossfn(args.loss_option)
        print("Built the loss function...")
        # # build optimizer
        # self.optimizer = self.build_optimizer(self.model.parameters(), args.optimizer_options)
        # print("Built the optimizer...")
        self.file_list = []
        full_path = args.feature_options.data_path + 'tt/mix/*.wav'
        self.file_list = glob.glob(full_path)
        self.feature_options = args.feature_options
        # build DataLoaders
        self.test_loader = data.sitec_inference_dataloader(args.model_name, args.feature_options, 'tt', args.cuda_option, self.device)
        
        # training options
        self.num_epoch = args.num_epoch

        self.output_path = args.output_path+'{}/'.format(self.model_name)
        if not os.path.exists(f'{self.output_path}/wav/'):
            os.makedirs(f'{self.output_path}/wav/')

        self.min_loss = float("inf")
        self.early_stop_count = 0

    def init_model(self,feature_options):
        model = models.DCCRN_mimo(win_len=feature_options.window_size,
                    win_inc=feature_options.hop_size, 
                    use_clstm=True)
        checkpoint = torch.load('./output/mimodccrn_MS_300ms_init_sitec_Sitec_snr_WPE/model.epoch147', map_location='cpu')
        # model.load_state_dict(checkpoint['model'])
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint['model'].items()}
        
        model.load_state_dict(checkpoint, strict=True)    
        model.to(self.device)        
        model.eval()
        # model.load_state_dict(checkpoint['model'])
        # model.to(self.device)
        return model


    def build_lossfn(self, fn_name):
        return get_lossfns()[fn_name]


    def build_optimizer(self, params, optimizer_options):
        if optimizer_options.name == "adam":
            return torch.optim.Adam(params, lr=optimizer_options.lr)
        if optimizer_options.name == "sgd":
            return torch.optim.SGD(params, lr=optimizer_options.lr, momentum=0.9)
        if optimizer_options.name == "rmsprop":
            return torch.optim.RMSprop(params, lr=optimizer_options.lr)

    def run(self):
        # for epoch in range(self.num_epoch):
        # self.train(epoch)
        self.inference()

        print("Model inferencing is finished.")

    def inference(self):
        # losses = AverageMeter()
        times = AverageMeter()
        # losses.reset()
        times.reset()
        len_d = len(self.test_loader)
        end = time.time()
        pesqs = []
        snris = []
        pesqs_n = []
        with open(self.output_path + '/out.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['filename', 'sisnri', 'noisy_pesq', 'pesq'])
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                input, input_frames, label, noise_MS, infdat = data
                input_frames = input_frames.to(self.device).float()
                label = label.to(self.device).float()
                input = input.to(self.device).float()
                noise_MS = noise_MS.to(self.device).float()
                    

                batch, channel, length = input.shape
                out_wav, specs_o, specs_t = self.model(input_frames, label, noise_MS)
                
                loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, specs_o, specs_t)
                
                # out_wavs = out_wav.squeeze(dim=0)
                # frames_cat = [out_wav[x, :, :1664] for x in range(95)]
                # frames_cat.append(out_wavs[-1, :, 1664:])
                # wav_out = torch.cat(frames_cat, -1)
                
                # frames_catl = [label[0, x, :, :1664] for x in range(95)]
                # frames_catl.append(label[0, -1, :, 1664:])
                # label_out = torch.cat(frames_catl, -1)
    
                
                label_out = label[0].cpu().detach().numpy()
                wav_out = out_wav[0].cpu().detach().numpy()
                input = input[0].cpu().detach().numpy()
                # if label.shape[-1] > audio_out.shape[-1]:
                #     label = label[...,:audio_out.shape[-1]]
                length = label_out.shape[0]

                result = sum([pesq(s, t) for s, t in zip(label_out, wav_out)])/length
                result_n = sum([pesq(s, t) for s, t in zip(label_out, input)])/length

                pesqs.append(result)
                pesqs_n.append(result_n)
                snris.append(snri)

                with open(f'{self.output_path}/out.csv', 'a', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow([os.path.split(self.file_list[i])[-1], snri.cpu().detach().numpy(),result_n, result])

                fn = f'{self.output_path}/wav/' + os.path.split(self.file_list[i])[-1]

                wav_out = wav_out.squeeze().T
                sf.write(fn.replace('.wav', '_{}.wav'.format(round(result,2))), wav_out, self.feature_options.sampling_rate, subtype='PCM_16')
                times.update(time.time() - end)
                end = time.time()
                print('%d/%d, pesq: {}, time estimated: %.2f seconds' .format(result) % (i + 1, len_d, times.avg * len_d), end='\r')
        print("\n")
        print('total average of PESQ: ', sum(pesqs) / len(pesqs))
        print('SNRi: ', sum(snris)/len(snris))

    def evaluate(self):
        pass


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path",default='./configs/inference.json',
                        help='The path to the config file. e.g. python train.py --config configs/dc_config.json')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    t = trainer(args)
    t.run()


if __name__ == "__main__":
    # writer = SummaryWriter("./tensorboard/log_echo")
    main()
    # writer.close()
