from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
import argparse, data, json, nn, numpy as np, os, time, torch
from torch.utils.tensorboard import SummaryWriter

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def signal_framing(signal, frame_length, frame_step):
    shape = list(signal.size())
    shape = shape[:-1] + [(shape[-1]-frame_length+frame_step)//frame_step, frame_length]
    strides = list(signal.stride())
    strides.insert(-1, frame_step*strides[-1])
    signal = torch.as_strided(signal, size=shape, stride=strides)
    return signal

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
        self.model = self.init_model(args.model_name, args.model_options)
        print("Loaded the model...")
        # build loss fn
        self.loss_fn = self.build_lossfn(args.loss_option)
        print("Built the loss function...")
        # build optimizer
        self.optimizer = self.build_optimizer(self.model.parameters(), args.optimizer_options)
        print("Built the optimizer...")
        # build DataLoader
        self.train_loader = data.wsj0_2mix_dataloader(args.model_name, args.feature_options, 'tr', args.cuda_option, self.device)
        self.valid_loader = data.wsj0_2mix_dataloader(args.model_name, args.feature_options, 'cv', args.cuda_option, self.device)
        

        # training options
        self.num_epoch = args.num_epoch
        self.min_loss = float("inf")
        self.early_stop_count = 0
        self.frame = 16000
        self.shift = 8000
        self.output_path = args.output_path + '/%s_%s' % (self.model_name, self.frame)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def init_model(self, model_name, model_options):
        model = nn.DCCRN_mimo(rnn_units=256, masking_mode='E', use_clstm=True, kernel_num=[32, 64, 128, 256, 256, 256])
        model = torch.load('/home/dail/Workspace/DCCRN_mimo/output/DCCRN_mimo_16000/model.epoch1')
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
        for epoch in range(self.num_epoch):
            self.train(epoch)
            self.validate(epoch)
            if self.early_stop_count == 10:
                print("Model stops improving, stop the training")
                break
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
            # break
            input, label = data
            input = [ele.to(self.device) for ele in input]
            label = [ele.to(self.device) for ele in label]
            # print(input[0].shape)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            tmp_input = signal_framing(input[0], self.frame, self.shift)
            tmp_input = tmp_input.reshape(4, -1, tmp_input.shape[-1])
            input = [tmp_input]
            del tmp_input
            zero = torch.zeros(1, input[0].shape[0]*input[0].shape[1], 128, device=input[0].device).float()
            rr = (zero, zero)
            ri = (zero, zero)
            ir = (zero, zero)
            ii = (zero, zero)
            out_wav, _ = self.model(input, state=[rr, ri, ir, ii])

            channel, batch, segment_size = out_wav.shape
            segment_stride = segment_size // 2
            # r = torch.zeros(channel, segment_stride).to(self.device)
            # input1 = torch.concat([out_wav[:, :, :segment_stride].contiguous().view(channel, -1), r],dim=1)
            # input2 = torch.concat([r, out_wav[:, :, segment_stride:].contiguous().view(channel, -1)],dim=1)
            # output = input1 + input2
            r = out_wav[:,-1,segment_stride:]
            output = torch.concat([out_wav[:, :, :segment_stride].contiguous().view(channel, -1), r], dim=1)

            loss_snr = self.loss_fn(output, label)

            loss = -loss_snr
            loss_avg = torch.mean(loss)
            losses.update(loss_avg.item())
            self.optimizer.zero_grad()
            loss_avg.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            # losses_sisnr_avg = torch.mean(loss_sisnr)
            losses_snr_avg = torch.mean(loss)

            losses_snr.update(losses_snr_avg.item())

            times.update(time.time() - end)
            end = time.time()
            writer.add_scalar('train_loss/loss(snr)', losses.avg, epoch * len_d + i + 1)

            print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f seconds' % (epoch, i + 1, len_d, losses.avg, times.avg * len_d), end='\r')
        print("\n")


    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        times = AverageMeter()
        # losses_sisnr = AverageMeter()
        losses_snr = AverageMeter()

        losses.reset()
        times.reset()
        # losses_sisnr.reset()
        losses_snr.reset()
        # losses_mse.reset()
        # losses_ec_lmse.reset()
        len_d = len(self.valid_loader)
        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.valid_loader):
                begin = time.time()
                input, label = data
                input = [ele.to(self.device) for ele in input]
                label = [ele.to(self.device) for ele in label]
                # print(input[0].shape)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                tmp_input = signal_framing(input[0], self.frame, self.shift)
                tmp_input = tmp_input.reshape(4, -1, tmp_input.shape[-1])

                input = [tmp_input]
                del tmp_input

                zero = torch.zeros(1, input[0].shape[0] * input[0].shape[1], 128, device=input[0].device).float()
                rr = (zero, zero)
                ri = (zero, zero)
                ir = (zero, zero)
                ii = (zero, zero)

                out_wav, _ = self.model(input, state=[rr, ri, ir, ii])

                channel, batch, segment_size = out_wav.shape
                segment_stride = segment_size // 2
                # r = torch.zeros(channel, segment_stride).to(self.device)
                # input1 = torch.concat([out_wav[:, :, :segment_stride].contiguous().view(channel, -1), r],dim=1)
                # input2 = torch.concat([r, out_wav[:, :, segment_stride:].contiguous().view(channel, -1)],dim=1)
                # output = input1 + input2
                r = out_wav[:, -1, segment_stride:]
                output = torch.concat([out_wav[:, :, :segment_stride].contiguous().view(channel, -1), r], dim=1)

                loss_snr = self.loss_fn(output, label)
                loss = -loss_snr
                loss_avg = torch.mean(loss)
                losses.update(loss_avg.item())
                losses_snr_avg = torch.mean(loss)
                losses_snr.update(losses_snr_avg.item())
                times.update(time.time() - end)
                end = time.time()
                writer.add_scalar('valid_loss/loss(snr)', losses.avg, epoch * len_d + i + 1)
                print('epoch %d, %d/%d, validation loss: %f, time estimated: %.2f seconds' % (epoch, i + 1, len_d, losses.avg, times.avg * len_d), end='\r')

            print("\n")
        if losses.avg < self.min_loss:
            self.early_stop_count = 0
            self.min_loss = losses.avg
            torch.save(self.model,self.output_path+"/model.epoch%d"%epoch)
            print("Saved new model")
        else:
            self.early_stop_count += 1


    def evaluate(self):
        pass


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path", default="/home/dail/Workspace/DCCRN_mimo/configs/train.json",
                        help='The path to the config file. e.g. python train_frame.py --config configs/speakerbeam_timitdb_config.json')


    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    t = trainer(args)
    t.run()


if __name__ == "__main__":
    writer = SummaryWriter("./tensorboard/speakerbeam")
    main()
    writer.close()
