import torch
import torch.utils.data
from detection.data.base_dataset import BaseDataset
from detection.data.base_dataset_test import BaseDatasetTest


class DataManager:
    def __init__(self, config):
        self.config = config

    def get_dataloaders(self, ast_proc=False):
        train = BaseDataset(config=self.config, mode="train", ast_proc=ast_proc)
        test = BaseDataset(config=self.config, mode="test", ast_proc=ast_proc)

        train_loader = torch.utils.data.DataLoader(dataset=train,
                                                   batch_size=self.config['batch_size'],
                                                   shuffle=True, num_workers=10)
        test_loader = torch.utils.data.DataLoader(dataset=test,
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=False, num_workers=10)
        return train_loader, test_loader

    def get_dataloader_test(self, ast_proc=False):
        train = BaseDatasetTest(config=self.config, ast_proc=ast_proc)
        test_loader = torch.utils.data.DataLoader(dataset=train,
                                                   batch_size=self.config['batch_size'],
                                                   shuffle=False, num_workers=10, drop_last=False)
        return test_loader
