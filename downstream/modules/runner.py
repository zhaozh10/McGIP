import logging
import os
from abc import abstractmethod
import torch
from numpy import inf
import time


class BaseTrainer(object):
    def __init__(self, accelerator,model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args
        self.accelerator=accelerator
        self.log_dir=args.record_dir
        # self.log_dir='./work_dirs/neo/'
        t=time.localtime()
        if args.preTrain==False:
            self.log_dir='spec/'
            self.checkpoint_dir='spec/'
        else:
            self.checkpoint_dir = args.record_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        filename=os.path.join(self.log_dir,f"{time.strftime('%Y%m%d',t)}_{time.strftime('%H%M%S',t)}.log")
        logging.basicConfig(filename=filename,
                            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.timePrefix=f"{time.strftime('%Y%m%d',t)}_{time.strftime('%H%M%S',t)}"
        self.logger = logging.getLogger(__name__)
        self.logger.info(args)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.early_stop=args.early_stop

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.metric_ftns=metric_ftns
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.start_epoch = 1


        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}



        # if args.resume is not None:
        #     self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            self._print_best()
            self.logger.info("=============== The end ===============")

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        # self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        # for key, value in self.best_recorder['test'].items():
        #     self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        # filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        # torch.save(state, filename)
        # self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, f'{self.timePrefix}_model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Runner(BaseTrainer):
    def __init__(self, accelerator, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Runner, self).__init__(accelerator, model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader


    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, train_sample in enumerate(self.train_dataloader):
            img=train_sample['img'].to(self.device)
            gt=train_sample['gt'].to(self.device)
        
            outputs = self.model(img)
            loss = self.criterion(outputs, gt)
            self.optimizer.zero_grad()
            # loss.backward()
            self.accelerator.backward(loss)
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                _, preds = torch.max(outputs.data, dim=1)
                acc = (preds == gt).sum().item() / gt.shape[0]
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}, Acc: {:.2f}%.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1), 100*acc))

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            valGT, valPred = [], []
            for batch_idx, val_sample in enumerate(self.val_dataloader):
                val_img=val_sample['img'].to(self.device)
                val_gts=val_sample['gt'].to(self.device)
                val_output= self.model(val_img)
                _, val_preds = torch.max(val_output.data, dim=1)
                valPred.extend(val_preds)
                valGT.extend(val_gts)

            val_metric = self.metric_ftns(torch.Tensor(valPred).type(torch.int64),torch.Tensor(valGT).type(torch.int64))
            log.update(**{'val_' + k: v for k, v in val_metric.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            testGT, testPred = [], []
            for batch_idx, test_sample in enumerate(self.test_dataloader):
                test_img=test_sample['img'].to(self.device)
                test_gts=test_sample['gt'].to(self.device)
                test_output= self.model(test_img)
                _, test_preds = torch.max(test_output.data, dim=1)
                testPred.extend(test_preds)
                testGT.extend(test_gts)

            test_metric = self.metric_ftns(torch.Tensor(testPred).type(torch.int64),torch.Tensor(testGT).type(torch.int64))
            log.update(**{'test_' + k: v for k, v in test_metric.items()})

        self.lr_scheduler.step()

        return log
