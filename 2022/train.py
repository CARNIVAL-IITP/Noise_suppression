from attrdict import AttrDict
from losses.loss_util import get_lossfns
import argparse, data, json, models, numpy as np, os, time, torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch.nn as nn
import torch.optim as optim


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
        # build model
        self.model = models.DCCRN_mimo(win_len=args.feature_options.window_size,
                    win_inc=args.feature_options.hop_size, 
                    use_clstm=True)
        checkpoint = torch.load('./best', map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
       
        # build loss fn
        self.loss_fn = self.build_lossfn(args.loss_option)
        # build optimizer
        self.optimizer = self.build_optimizer(self.model.parameters(), args.optimizer_options)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=50, eta_min=0)
    
        # build DataLoaders
        self.train_loader = data.sitec_dataloader(args.model_name, args.feature_options, 'tr', args.cuda_option, self.device)
        self.valid_loader = data.sitec_dataloader(args.model_name, args.feature_options, 'cv', args.cuda_option, self.device)

        # training options
        self.num_epoch = args.num_epoch
        self.output_path = args.output_path+'%s_%s_%s'%(self.model_name, self.dataset, self.loss_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.min_loss = float("inf")
        self.early_stop_count = 0
        self.final_loss = 0


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
        for epoch in range(self.num_epoch):
            print(self.model_name)
            self.train(epoch)
            self.validate(epoch)
            self.scheduler.step()
        print("Model training is finished.")


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
            input, input_frames, label, noise_MS = data
            input_frames = input_frames.to(self.device).float()
            label = label.to(self.device).float()
            input = input.to(self.device).float()
            noise_MS = noise_MS.to(self.device).float()
            out_wav, specs_o, specs_t = self.model(input_frames, label, noise_MS)
            
            loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, specs_o, specs_t)
            losses.update(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            losses_snr_avg = torch.mean(loss)
            losses_snr.update(losses_snr_avg.item())
            times.update(time.time() - end)
            end = time.time()
            writer.add_scalar('train_loss/loss(snr)', losses.avg, epoch * len_d + i + 1)
            print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f seconds' % (epoch, i + 1, len_d, losses.avg, times.avg * len_d), end='\r')
            wandb.log({
                "train_loss/loss": losses.avg,
                "train_WPE": WPE,
                "train_snr": loss_snr,
            })
        print("\n")


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
                input, input_frames, label, noise_MS = data
                input_frames = input_frames.to(self.device).float()
                label = label.to(self.device).float()
                input = input.to(self.device).float()
                noise_MS = noise_MS.to(self.device).float()
                out_wav, specs_o, specs_t = self.model(input_frames, label, noise_MS)
                
                loss, loss_snr, WPE, snri = self.loss_fn(input,  out_wav, label, specs_o, specs_t)           
                losses.update(loss.item())
                losses_snr_avg = torch.mean(loss)
                losses_snr.update(losses_snr_avg.item())
                times.update(time.time() - end)
                end = time.time()
                writer.add_scalar('valid_loss/loss(snr)', losses.avg, epoch * len_d + i + 1)
                print('epoch %d, %d/%d, validation loss: %f, time estimated: %.2f seconds' % (epoch, i + 1, len_d, losses.avg, times.avg * len_d), end='\r')
                wandb.log({
                    "valid_loss": losses.avg,
                    "valid_WPE": WPE,
                    "valid_snr": loss_snr,
                })
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
                torch.save(checkpoint, self.output_path+"/model.epoch%d"%epoch)
                torch.save(checkpoint, self.output_path+"/best")
                print('model is saved.')
            self.min_loss = self.final_loss
        else:
            self.early_stop_count += 1



    def evaluate(self):
        pass


def main():
    wandb.init()
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path", default='./configs/train.json',
                        help='The path to the config file. e.g. python train.py --config configs/dc_config.json')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    t = trainer(args)
    t.run()


if __name__ == "__main__":
    writer = SummaryWriter("./tensorboard")
    main()
    writer.close()
