import sys
import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_transformers import BertModel

"""
Archivo obtenido de Helsinki-NLP/prosody (MIT License 2019)
https://github.com/Helsinki-NLP/prosody
"""
class Bert(nn.Module):
    def __init__(self, device, labels=None):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, labels).to(device)
        self.device = device

    def forward(self, x, y):

        x = x.to(self.device)
        y = y.to(self.device)

        if self.training:
            self.bert.train()
            enc = self.bert(x)[0]
        else:
            self.bert.eval()
            with torch.no_grad():
                enc = self.bert(x)[0]
        logits = self.fc(enc).to(self.device)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

