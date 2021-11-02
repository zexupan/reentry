import time
from utils import *
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

class Solver(object):
    def __init__(self, train_data, validation_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.args = args
        self.amp = amp

        self.CEloss = nn.BCEWithLogitsLoss()

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)

        self.model, self.optimizer = self.amp.initialize(model, optimizer,
                                                        opt_level=args.opt_level,
                                                        patch_torch_functions=args.patch_torch_functions)

        if self.args.distributed:
            self.model = DDP(self.model)

        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('logs/%s/model_dict.pt' % self.args.continue_from, map_location='cpu')

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.amp.load_state_dict(checkpoint['amp'])

            self.start_epoch=checkpoint['epoch']
            self.prev_val_loss = checkpoint['prev_val_loss']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))
            
        else:
            # checkpoint = torch.load('logs/sync-2mix-large/model_dict_large.pt', map_location='cpu')
            # self.model.load_state_dict(checkpoint['model'])
            # print("loaded from avConv_2020-12-14(22:58:42)")
            
            self.prev_val_loss = float("inf")
            self.best_val_loss = float("inf")
            self.val_no_impv = 0
            self.start_epoch=1
            if self.print: print('Start new training')

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs+1):
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)
            # Train
            self.model.train()
            start = time.time()
            tr_loss, tr_acc = self._run_one_epoch(data_loader = self.train_data)
            reduced_tr_loss = self._reduce_tensor(tr_loss)
            reduced_tr_acc = self._reduce_tensor(tr_acc)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Train Loss {2:.3f}'.format(
                        epoch, time.time() - start, reduced_tr_loss))

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss, val_acc = self._run_one_epoch(data_loader = self.validation_data, state='val')
                reduced_val_loss = self._reduce_tensor(val_loss)
                reduced_val_acc = self._reduce_tensor(val_acc)

            if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Valid Loss {2:.3f}'.format(
                          epoch, time.time() - start, reduced_val_loss))


            # Check whether to adjust learning rate and early stop
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 20:
                    if self.print: print("No imporvement for 20 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0

            if self.val_no_impv == 6:
                self.halving = True

            # Halfing the learning rate
            self.halving = True
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] * 0.96
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = reduced_val_loss

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train_loss', reduced_tr_loss, epoch)
                    self.writer.add_scalar('Validation_loss', reduced_val_loss, epoch)
                    self.writer.add_scalar('Train_acc', reduced_tr_acc, epoch)
                    self.writer.add_scalar('Validation_acc', reduced_val_acc, epoch)

                # Save model
                if reduced_val_loss < self.best_val_loss:
                    self.best_val_loss = reduced_val_loss
                    checkpoint = {'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'amp': self.amp.state_dict(),
                                    'epoch': epoch+1,
                                    'prev_val_loss': self.prev_val_loss,
                                    'best_val_loss': self.best_val_loss,
                                    'val_no_impv': self.val_no_impv}
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict.pt")
                    print("Fund new best model, dict saved")

    def _run_one_epoch(self, data_loader, state='train'):
        total_loss = 0
        total_acc= torch.tensor([0.0]).cuda()
        self.optimizer.zero_grad()
        for i, (audio, visual, label) in enumerate(data_loader):
            audio = audio.cuda().squeeze(0).float()
            visual = visual.cuda().squeeze(0).float()
            label = label.cuda().squeeze(0)

            est_label = self.model(audio, visual)

            loss = self.CEloss(est_label, label.float())
  
            if state == 'train':
                with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                               self.args.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.data
            acc = self.cal_acc(est_label, label)
            total_acc += acc
            
        return total_loss / (i+1), total_acc / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt

    def cal_acc(self, output, target):
        output[output> 0] = 1
        output[output<=0] = 0

        correct = 0
        total = 0
        for i in range(target.shape[0]):
            total += 1
            if (output[i] == target[i]):
                correct += 1
        return correct/total
