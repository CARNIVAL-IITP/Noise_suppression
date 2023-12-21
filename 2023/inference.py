from attrdict import AttrDict
from loss_functions.loss_util import get_lossfns
import argparse, data, json, models, numpy as np, os, time, torch
import glob, librosa
import matplotlib.pyplot as plt
import soundfile as sf
from itertools import permutations
from pypesq import pesq
import csv
import scipy.stats as st



def get_free_gpu():

    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def child_freeze(model):
        for name, child in model.named_children():
            for param in child.parameters():  
                param.requires_grad = False            
            child_freeze(child)


class trainer:
    def __init__(self, args):
        
        print('Model:', os.path.realpath(__file__).split('/')[-3][20:])
        self.model_name = os.path.realpath(__file__).split('/')[-3][20:]
        args.model_name = self.model_name
        
        self.loss_name = args.loss_option
        self.dataset = args.dataset

        if args.cuda_option == "True":
            print("GPU mode on...")
            available_device = get_free_gpu()
            print("We found an available GPU: %d!"%available_device)
            self.device = torch.device('cuda:%d'%available_device)
        else:
            self.device = torch.device('cpu')


        self.model = self.init_model(args)
       
        self.loss_fn = self.build_lossfn(args.loss_option)
        
        self.feature_options = args.feature_options

        self.test_loader = data.dns_inference_dataloader(args.model_name, args.feature_options, 'tt', args.cuda_option, self.device)
        
        self.output_path = args.inference_path
        self.args = args
        os.makedirs(f'{self.output_path}/wav/', exist_ok=True)


    def init_model(self, args):
        if args.kd:
            teacher_model = models.DCCRN_mimo(win_len=args.feature_options.window_size,
                        win_inc=args.feature_options.hop_size, 
                        use_clstm=True)
            checkpoint_t = torch.load(args.teacher_model, map_location='cpu')
            teacher_model.load_state_dict(checkpoint_t['model'], strict=False)
            child_freeze(teacher_model)
            model = models.DCCRN_mimo(win_len=args.feature_options.window_size,
                        win_inc=args.feature_options.hop_size, 
                        use_clstm=True, teacher_model=teacher_model)
        else:
            model = models.DCCRN_mimo(win_len=args.feature_options.window_size,
                        win_inc=args.feature_options.hop_size, 
                        use_clstm=True)

        checkpoint = torch.load(args.inference_model, map_location='cpu')
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint['model'].items()}
        
        model.load_state_dict(checkpoint, strict=True)    
        model.to(self.device)        
        model.eval()
        return model

    def cal_confidence(self, data):
        m = sum(data) / len(data)
        inter = st.t.interval(alpha=0.95, df = len(data)-1, loc=sum(data) / len(data), scale=st.sem(data))
        return m, inter[1]-m

    def build_lossfn(self, fn_name):
        return get_lossfns()[fn_name]


    def run(self):
        
        self.inference()


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
            wr.writerow(['filename', 'snri', 'noisy_pesq', 'pesq'])
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                input, label, noise, noise_est, infdat, ch = data
                label = label.to(self.device).float()
                input = input.to(self.device).float()
                noise = noise.to(self.device).float()
                noise_est = noise_est.to(self.device).float()
                ch = ch.to(self.device)
                
                out_wav_t, est_phase_t, out_wav, est_phase, specs_t= self.model(input, label, noise, noise_est, teacher=False)
               
                loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, est_phase, specs_t) 
                label_out = label[0].cpu().detach().numpy()
                wav_out = out_wav[0].cpu().detach().numpy()
                input = input[0].cpu().detach().numpy()
                                
                length = label_out.shape[0]

                result = sum([pesq(s, t) for s, t in zip(label_out[:ch, :], wav_out[:ch, :])])/length
                result_n = sum([pesq(s, t) for s, t in zip(label_out[:ch, :], input[:ch, :])])/length

                pesqs.append(result)
                pesqs_n.append(result_n)
                snris.append(snri.cpu().detach().numpy())
                
                with open(f'{self.output_path}/out.csv', 'a', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow([infdat[0], snri.cpu().detach().numpy(),result_n, result])

                fn = f'{self.output_path}/wav/' + os.path.split(infdat[0])[-1]

                wav_out = wav_out.squeeze().T

                sf.write(fn.replace('.wav', '_{}.wav'.format(round(result,2))), wav_out, self.feature_options.sampling_rate, subtype='PCM_16')
                times.update(time.time() - end)
                end = time.time()
                print('%d/%d, pesq: {}, time estimated: %.2f seconds' .format(result) % (i + 1, len_d, times.avg * len_d), end='\r')
                # if i==5:
                #     break

        f = open(f"{self.output_path}/result.txt", 'w')
        # line = f.readlines()
        # print("\n")
        # print(line[2])
        # exit()
        print("\n")
        # f.write(self.output_path)
        # print(self.output_path)
        # f.write("\n")
        # n = f'\n PESQ_noisy: {self.cal_confidence(pesqs_n)}\n'
        # f.write(n)
        # print(n)
        p = f'\n PESQ: {self.cal_confidence(pesqs)}\n'

        f.write(p)
        print(p)
        print(f'delay time: {(self.args.feature_options.window_size+self.args.feature_options.hop_size)/16} ms.')
        # s = f'\n SNRi: {self.cal_confidence(snris)}\n'
        # f.write(s)
        # print(s)
        f.close()
        


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path",default='./configs/conf.json',
                        help='The path to the config file. e.g. python train.py --config configs/dc_config.json')

    config = parser.parse_args()
    with open(config.path, 'r') as f:
        args = json.load(f)
        args = AttrDict(args)
        args.model_name = os.path.realpath(__file__).split('/')[-3][20:]
    with open(config.path, 'w') as f:
        json.dump(args, f, indent="\t")
    t = trainer(args)
    t.run()


if __name__ == "__main__":
    main()
