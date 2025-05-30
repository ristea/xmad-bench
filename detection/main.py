import json
import os
from pathlib import Path
import torch.optim as optim

from data.data_manager import DataManager
from trainer import Trainer
from utils.data_logs import save_logs_about
from torch.utils.tensorboard import SummaryWriter
from models_util import *


def main():
    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    Path(os.path.join(config['exp_path'], config['exp_name'])).mkdir(exist_ok=True, parents=True)

    logs_writer = SummaryWriter(os.path.join('runs', config['exp_name']))

    model_type = config.get("model_type", "resnet18")
    ast_proc = False
    if model_type == "resnet18":
        model = get_resnet18_model(config)
    elif model_type == "resnet50":
        model = get_resnet50_model(config)
    elif model_type == "septr":
        model = get_septr_model(config)
    elif model_type == "ast":
        model = get_ast_model(config)
        ast_proc = True

    # Save info about experiment
    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'])

    data_manager = DataManager(config)
    train_loader, validation_loader = data_manager.get_dataloaders(ast_proc)
    test_loader = data_manager.get_dataloader_test(ast_proc)

    trainer = Trainer(model, train_loader, validation_loader, criterion, optimizer, lr_scheduler, logs_writer, config)
    trainer.train()
    trainer.test_out_of_domain(test_loader)


if __name__ == "__main__":
    main()
