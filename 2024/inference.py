from attrdict import AttrDict
from loss_functions.loss_util import get_lossfns
import argparse, data, json, models, numpy as np, os, time, torch
import glob, librosa
import matplotlib.pyplot as plt
import soundfile as sf
from pypesq import pesq
import scipy.stats as st
from tqdm import tqdm
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
        
        # print('Model:', os.path.realpath(__file__).split('/')[-3][20:])
        self.model_name = os.path.realpath(__file__).split('/')[-3][20:]
        args.model_name = self.model_name
        
        self.loss_name = args.loss_option
        self.dataset = args.dataset

        if args.cuda_option == "True":
            # print("GPU mode on...")
            available_device = get_free_gpu()
            # print("We found an available GPU: %d!"%available_device)
            self.device = torch.device('cuda:%d'%available_device)
        else:
            self.device = torch.device('cpu')


        self.model = self.init_model(args)
       
        
        self.feature_options = args.feature_options

        self.test_loader = data.dns_inference_dataloader(args.model_name, args.feature_options, 'tt', args.cuda_option, self.device)
        
        self.output_path = args.inference_path
        self.args = args
        os.makedirs(f'{self.output_path}/wav/', exist_ok=True)



    def init_model(self, args):
        model = models.DCCRN_mimo(win_len=args.feature_options.window_size,
                        win_inc=args.feature_options.hop_size, 
                        use_clstm=True)

        checkpoint = torch.load(args.inference_model, map_location='cpu')
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint['model'].items()}
        
        model.load_state_dict(checkpoint, strict=False)    
        model.to(self.device)        
        model.eval()
        return model

    def cal_confidence(self, data):
        m = sum(data) / len(data)
        inter = st.t.interval(alpha=0.95, df = len(data)-1, loc=sum(data) / len(data), scale=st.sem(data))
        return m, inter[1]-m


    def run(self):
        
        self.inference()


    def inference(self):
        # times = AverageMeter()
        # times.reset()
        # len_d = len(self.test_loader)
        end = time.time()
        total_t = 0
        pesqs = []

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                input, label, noise, noise_est, infdat = data
                label = label.to(self.device).float()
                input = input.to(self.device).float()
                noise = noise.to(self.device).float()
                noise_est = noise_est.to(self.device).float()
               
                
                out_wav, est_phase, tar_phase= self.model(input, label, noise, noise_est, teacher=False)

                # times.update(time.time() - end)
                total_t +=time.time() - end
                end = time.time()
                
                
                label_out = label[0].cpu().detach().numpy()
                wav_out = out_wav[0].cpu().detach().numpy()
                input = input[0].cpu().detach().numpy()
                

                result = sum([pesq(label_out[j], wav_out[j]) for j in range(8)])/8
                pesqs.append(result)
                fn = f'{self.output_path}wav/' + os.path.split(infdat[0])[-1]
                sf.write(fn, wav_out[0], self.feature_options.sampling_rate, subtype='PCM_16')
                
                # print(f'%d/%d, pesq: {result:.2f}, time estimated: %.2f seconds' .format(result) % (i + 1, len_d, times.avg * len_d), end='\r')
                
                
        rtf = total_t/(150*out_wav.shape[-1]/16000)
        print(f'\n\nRTF: {rtf:.3f}\n')
        print(f'PESQ: {self.cal_confidence(pesqs)[0]:.2f} +- {self.cal_confidence(pesqs)[1]:.2f}\n')
        # print(f'delay: {}\n')s
        print(f'delay time: {(self.args.feature_options.window_size+self.args.feature_options.hop_size)/16} ms.\n')

        # print(f'MOS: ')


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
    # print('\n')
    scores = glob.glob('../MOS/*.txt')

    print(f'MOS : total {len(scores)} participants.')
    score={}
    for s in scores:
        f = open(s, 'r', encoding='euc-kr')
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i<120:
                if line.split('/')[0] in score.keys():
                    score[line.split('/')[0]].append(int(line.split(' ')[-1].replace('\n', '')))
                else:
                    score[line.split('/')[0]]=[int(line.split(' ')[-1].replace('\n', ''))]
        f.close()

    for name in score.keys():
        s = sum(score[name])/len(score[name])
        print('- ', name, round(s,2))
    print('\n')
