import torch.nn as nn
from transformers import ASTModel


class ASTModelLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        x = self.ast(x[:, 0]).pooler_output
        x = self.fc(x)
        return x

