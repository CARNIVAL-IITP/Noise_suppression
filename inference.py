from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
import argparse, data, json, nn, numpy as np, os, time, torch
import glob, librosa
# from data.feature_utils import get_istft
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
from itertools import permutations
from pypesq import pesq


def get_free_gpu():

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
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
        # self.device = torch.device('cpu')

        # build model
        self.model = self.init_model(args.model_name, args.model_options)
        print("Loaded the model...")
        # build loss fn
        self.loss_fn = self.build_lossfn(args.loss_option)
        print("Built the loss function...")
        # build optimizer
        self.optimizer = self.build_optimizer(self.model.parameters(), args.optimizer_options)
        print("Built the optimizer...")
        self.file_list = []
        full_path = args.feature_options.data_path + 'tt' + '/mixed/*.wav'
        self.file_list = glob.glob(full_path)
        self.feature_options = args.feature_options
        # build DataLoaders
        if args.dataset == "wsj0-2mix":
            self.test_loader = data.wsj0_2mix_inference_dataloader(args.model_name, args.feature_options, 'tt', args.cuda_option, self.device)
        elif args.dataset == "daps":
            self.train_loader = data.daps_enhance_dataloader(args.train_num_batch, args.feature_options, 'train', args.cuda_option, self.device)
            self.valid_loader = data.daps_enhance_dataloader(args.vaildate_num_batch, args.feature_options, 'validation', args.cuda_option, self.device)
        elif args.dataset == "edinburgh_tts":
            self.train_loader = data.edinburgh_tts_dataloader(args.model_name, args.feature_options, 'train', args.cuda_option, self.device)
            self.valid_loader = data.edinburgh_tts_dataloader(args.model_name, args.feature_options, 'validation', args.cuda_option, self.device)


        # training options
        self.num_epoch = args.num_epoch

        self.output_path = args.output_path+'{}50/'.format(self.model_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.min_loss = float("inf")
        self.early_stop_count = 0

    def init_model(self, model_name, model_options):
        model = torch.load('/home/dail/Workspace/DCCRN/output/DCCRN_mimo_DNS2020_4_snr_WPE_50/model.epoch70')
        model.to(self.device)
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

        print("Model training is finished.")

    def inference(self):
        # losses = AverageMeter()
        times = AverageMeter()
        # losses.reset()
        times.reset()
        len_d = len(self.test_loader)
        end = time.time()
        pesqs = []
        snris = []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                begin = time.time()
                input, label, infdat = data
                input = [ele.to(self.device) for ele in input]
                label = [ele.to(self.device) for ele in label]

                est_phase, out_wav, specs_t = self.model(input, label)
                loss, loss_snr, WPE, snri = self.loss_fn(input, out_wav, label, est_phase, specs_t)

                out_wavs = out_wav.squeeze(dim=1)
                audio_out = out_wavs.cpu().detach().numpy()
                label = label[0][0].cpu().detach().numpy()
                if label.shape[-1] > audio_out.shape[-1]:
                    label = label[:,:audio_out.shape[-1]]
                length = label.shape[0]

                result = sum([pesq(s, t) for s, t in zip(label, audio_out[0])])/length
                pesqs.append(result)
                snris.append(snri)

                fn = self.output_path + os.path.split(self.file_list[i])[-1]
                audio_out = audio_out.squeeze().T
                # sf.write(fn.replace('.wav', '_{}.wav'.format(result)), audio_out, self.feature_options.sampling_rate, subtype='PCM_16')
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
