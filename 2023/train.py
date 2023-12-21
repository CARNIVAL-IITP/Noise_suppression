from attrdict import AttrDict
from loss_functions.loss_util import get_lossfns
import argparse, data, json, models, numpy as np, os, time, torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from asteroid.losses import SingleSrcMultiScaleSpectral
import datetime
import soundfile as sf

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
        self.num_epoch = args.num_epoch
        self.args = args
        self.wandb = args.wandb

        if  self.wandb:
            wandb.init(project="iitp3", save_code=True)
            wandb.run.name = self.model_name
            wandb.config.update(args)

        if args.cuda_option == "True":
            print("GPU mode on...")
            available_device = get_free_gpu()
            print("We found an available GPU: %d!"%available_device)
            self.device = torch.device('cuda:%d'%available_device)
        else:
            self.device = torch.device('cpu')
            
       
        self.model = self.init_model(args)
        
    
        self.loss_fn = self.build_lossfn(args.loss_option)

        self.optimizer = self.build_optimizer(self.model.parameters(), args.optimizer_options)

        self.train_loader = data.dns_train_dataloader(args.model_name, args.feature_options, 'tr', args.cuda_option, self.device)
        self.valid_loader = data.dns_inference_dataloader(args.model_name, args.feature_options,'cv', args.cuda_option, self.device)
        
        self.output_path = args.output_path
        self.checkpoint_path = self.output_path+'checkpoint/'
        self.validation_check_path = self.output_path+'validation_check/'

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.validation_check_path, exist_ok=True)

        self.epoch = 0
        self.min_loss = float("inf")
        self.early_stop_count = 0
        self.earlystop = args.earlystop

        if args.resum:
            checkpoint = torch.load(args.resum_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=100, eta_min=0, last_epoch=-1)


    def init_model(self, args):

        if args.kd:
            teacher_model = models.DCCRN_mimo(win_len=args.feature_options.window_size,
                        win_inc=args.feature_options.hop_size, 
                        use_clstm=True)
            checkpoint_t = torch.load(args.teacher_model, map_location='cpu')
            teacher_model.load_state_dict(checkpoint_t['model'], strict=True)
            child_freeze(teacher_model)

            model = models.DCCRN_mimo(win_len=args.feature_options.window_size,
                        win_inc=args.feature_options.hop_size, 
                        use_clstm=True, teacher_model=teacher_model)
            checkpoint_s = torch.load(args.student_model, map_location='cpu')
            model.load_state_dict(checkpoint_s['model'], strict=False)
        else:
            model = models.DCCRN_mimo(win_len=args.feature_options.window_size,
                        win_inc=args.feature_options.hop_size, 
                        use_clstm=True)
        
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
        if optimizer_options.name == "adamW":
            return torch.optim.AdamW(params, lr=optimizer_options.lr)
    
    

    def run(self):
        for epoch in range(self.epoch, self.num_epoch, 1):
            print(self.model_name)
            loss_t = self.train(epoch)
            loss_v = self.validate(epoch)
            self.scheduler.step()
        return self.args


    def train(self, epoch):
        losses = AverageMeter()
        times = AverageMeter()
        losses_snr = AverageMeter()
        losses.reset()
        times.reset()
        losses_snr.reset()
        self.model.train()
        len_d = len(self.train_loader)
        end = time.time()
        for i, data in enumerate(self.train_loader):
            input, label, noise, noise_est, _, _ = data
            label = label.to(self.device).float()
            input = input.to(self.device).float()
            noise = noise.to(self.device).float()
            noise_est = noise_est.to(self.device).float()

            if self.args.kd:
                out_wav_t, est_phase_t, out_wav, est_phase, specs_t= self.model(input, label, noise, noise_est)
                loss_kd, loss_snr_kd, WPE_kd, snri_t = self.loss_fn(input,  out_wav, out_wav_t, est_phase, specs_t) 
                loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, est_phase, specs_t) 
                loss = loss + 0.5*loss_snr_kd

            elif self.args.teacher:
                out_wav, est_phase, specs_t = self.model(input, label, noise, noise_est, teacher=True)
                loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, est_phase, specs_t) 

            else:  #student
                out_wav, est_phase, specs_t = self.model(input, label, noise, noise_est)
                loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, est_phase, specs_t) 
            
            losses.update(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            losses_snr_avg = torch.mean(loss)
            losses_snr.update(losses_snr_avg.item())
            times.update(time.time() - end)

            end = time.time()
            print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f seconds' % (epoch, i + 1, len_d, losses.avg, times.avg * len_d), end='\r')
            
            if self.wandb:
                wandb.log({
                    "train_total_loss": losses.avg,
                    "train_WPE": WPE,
                    "train_snr": loss_snr,
                })

            for a in [out_wav, est_phase, specs_t, label, input, noise, loss, WPE, loss_snr]:
                del a
            
        print("\n")
        return loss


    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        times = AverageMeter()

        losses_snr = AverageMeter()

        losses.reset()
        times.reset()

        losses_snr.reset()

        len_d = len(self.valid_loader)
        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.valid_loader):
                input, label, noise, noise_est, info, ch = data
                label = label.to(self.device).float()
                input = input.to(self.device).float()
                noise = noise.to(self.device).float()
                noise_est = noise_est.to(self.device).float()

                if self.args.kd:
                    out_wav_t, est_phase_t, out_wav, est_phase, specs_t= self.model(input, label, noise, noise_est)
                    loss_kd, loss_snr_kd, WPE_kd, snri_t = self.loss_fn(input,  out_wav, out_wav_t, est_phase, specs_t) 
                    loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, est_phase, specs_t) 
                    loss = loss + 0.5*loss_snr_kd
                
                elif self.args.teacher:
                    out_wav, est_phase, specs_t = self.model(input, label, noise, noise_est, teacher=True)
                    loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, est_phase, specs_t) 
                    
                else:  
                    out_wav, est_phase, specs_t = self.model(input, label, noise, noise_est)
                    loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, est_phase, specs_t) 

                losses.update(loss.item())
                losses_snr_avg = torch.mean(loss)
                losses_snr.update(losses_snr_avg.item())

                times.update(time.time() - end)
                end = time.time()
                print('epoch %d, %d/%d, valid loss: %f, time: %.2f seconds' % (epoch, i + 1, len_d, losses.avg, times.avg * len_d), end='\r')
                if i==0:
                    sf.write(f'{self.validation_check_path}input.wav', input[0,:ch[0],:].cpu().detach().numpy().squeeze().T, 16000, subtype='PCM_16')
                    sf.write(f'{self.validation_check_path}output.wav', out_wav[0,:ch[0],:].cpu().detach().numpy().squeeze().T, 16000, subtype='PCM_16')
                    sf.write(f'{self.validation_check_path}target.wav', label[0,:ch[0],:].cpu().detach().numpy().squeeze().T, 16000, subtype='PCM_16')
                    
                if self.wandb:
                    wandb.log({
                        "valid_loss": losses.avg,
                        "valid_WPE": WPE,
                        "valid_snr": loss_snr,
                    })

                for a in [out_wav, est_phase, specs_t, label, input, noise, loss, WPE, loss_snr]:
                    del a
                                      
            print("\n")
            
        self.final_loss = losses.avg
        if self.final_loss < self.min_loss:
            self.early_stop_count = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            if epoch>-1:
                torch.save(checkpoint, self.checkpoint_path+"model.epoch%d"%epoch)
                torch.save(checkpoint, self.checkpoint_path+"best")
                print('model is saved.')
            self.min_loss = self.final_loss
            
        else:
            self.early_stop_count += 1
        return self.final_loss


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path", default='./configs/conf.json',
                        help='The path to the config file. e.g. python train.py --config configs/dc_config.json')

    config = parser.parse_args()
    with open(config.path, 'r') as f:
        args = json.load(f)
        args = AttrDict(args)
        args.model_name = os.path.realpath(__file__).split('/')[-3][20:]
    with open(config.path, 'w') as f:
        json.dump(args, f, indent="\t")
        
    t = trainer(args)
    args = t.run()
   


if __name__ == "__main__":
    main()
