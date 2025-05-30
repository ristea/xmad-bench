import torch
import numpy as np
from utils.stats_manager import StatsManager
from utils.data_logs import save_logs_train, save_logs_eval
import os


class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer,
                 lr_scheduler, logs_writer, config):
        self.config = config
        self.network = network
        self.stats_manager = StatsManager(config)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logs_writer = logs_writer

        self.best_metric = 0.0

    def train_epoch(self, epoch):
        running_loss = []
        self.network.train()
        for idx, (inputs, labels) in enumerate(self.train_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).long()

            predictions = self.network(inputs)

            loss = self.criterion(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            if idx % self.config['print_loss'] == 0:
                running_loss = np.mean(np.array(running_loss))
                print(f'Training loss on iteration {idx} = {running_loss}')
                save_logs_train(os.path.join(self.config['exp_path'], self.config['exp_name']),
                                f'Training loss on iteration {idx} = {running_loss}')

                self.logs_writer.add_scalar('Training Loss', running_loss, epoch * len(self.train_dataloader) + idx)
                running_loss = []

    @torch.no_grad()
    def eval_net(self, epoch):
        stats_labels = []
        stats_predictions = []

        running_eval_loss = 0.0
        self.network.eval()
        for idx, (inputs, labels) in enumerate(self.eval_dataloader):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).long()

            predictions = self.network(inputs)
            eval_loss = self.criterion(predictions, labels)
            running_eval_loss += eval_loss.item()

            stats_predictions.append(predictions.detach().cpu().numpy())
            stats_labels.append(labels.detach().cpu().numpy())

        performance = self.stats_manager.get_stats(predictions=stats_predictions, labels=stats_labels)
        running_eval_loss = running_eval_loss / len(self.eval_dataloader)

        stats = f"### EVAL:: Epoch {epoch}:: Loss = {running_eval_loss} :: ACC = {performance[0]}, AUC = {performance[1]}, EER = {performance[2]}"
        print(stats)
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']), stats)

        if self.best_metric < performance[0]:
            self.best_metric = performance[0]
            self.save_net_state(epoch, best=True)

        self.logs_writer.add_scalar('Validation Loss', running_eval_loss, (epoch + 1) * len(self.train_dataloader))

    def train(self):
        if self.config['resume_training'] is True:
            checkpoint = torch.load(os.path.join(self.config['exp_path'],
                                                 self.config['exp_name'],
                                                 'latest_checkpoint.pkl'),
                                    map_location=self.config['device'])
            self.network.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            self.train_epoch(i)
            self.save_net_state(i, latest=True)

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

            self.lr_scheduler.step()

    def save_net_state(self, epoch, latest=False, best=False):
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'latest_checkpoint.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
        elif best is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'best_model.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)
        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'model_epoch_{epoch}.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)

    @torch.no_grad()
    def test_out_of_domain(self, dataloader):
        checkpoint = torch.load(os.path.join(self.config['exp_path'], self.config['exp_name'], 'best_model.pkl'),
                                map_location=self.config['device'])
        self.network.load_state_dict(checkpoint['model_weights'])
        self.network.eval()

        stats_labels = []
        stats_predictions = []

        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).long()

            predictions = self.network(inputs)
            stats_predictions.append(predictions.detach().cpu().numpy())
            stats_labels.append(labels.detach().cpu().numpy())

        performance = self.stats_manager.get_stats(predictions=stats_predictions, labels=stats_labels)

        stats = f"### Out of domain:: Best epoch {checkpoint['epoch']} :: ACC = {performance[0]}, AUC = {performance[1]}, EER = {performance[2]}"
        print(stats)
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']), stats)
