# Torch & Lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# Utils
import einops

def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)

class Conv1D(nn.Module):

    def __init__(self, in_channels, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

        self.layer_in = nn.Conv1d(in_channels, 2*in_channels, kernel_size=1)
        self.norm_in = nn.BatchNorm1d(2*in_channels)
        self.layer1 = nn.Conv1d(2*in_channels, 4*in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(4*in_channels)
        self.layer2 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=5, padding=2)
        self.norm2 = nn.BatchNorm1d(4*in_channels)
        self.layer3 = nn.Conv1d(4*in_channels, 2*in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(2*in_channels)
        self.layer4 = nn.Conv1d(2*in_channels, in_channels, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm1d(in_channels)
        self.fc = nn.Linear(in_channels, 2)
    
    def forward(self, x):

        x = self.layer_in(x)
        x = self.norm_in(x)
        x = self.dropout(self.relu(x))

        x = self.layer1(x)
        x = self.norm1(x)
        x = self.dropout(self.relu(x))

        x = self.layer2(x)
        x = self.norm2(x)
        x = self.dropout(self.relu(x))

        x = self.layer3(x)
        x = self.norm3(x)
        x = self.dropout(self.relu(x))

        x = self.layer4(x)
        x = self.norm4(x)
        x = self.dropout(self.relu(x))

        x = einops.rearrange(x, 'B C L -> B L C')
        x = self.softmax(self.fc(x))

        return x

class LitConv1D(pl.LightningModule):

    def __init__(self, fasttext, in_channels, dropout=0.1, lr=1e-4):
        super().__init__()

        self.lr = lr
        self.fasttext = fasttext
        self.model = Conv1D(in_channels, dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):

        arrs, sentences = batch[0], batch[1]
        arrs = einops.rearrange(arrs, 'B L -> (B L)')
        outputs = self.model(sentences)
        outputs = einops.rearrange(outputs, 'B L C -> (B L) C')
        loss = self.criterion(outputs, arrs)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        arrs, sentences, len_arrs = batch[0], batch[1], batch[2]
        arrs_ = einops.rearrange(arrs, 'B L -> (B L)')
        outputs = self.model(sentences)
        outputs_ = einops.rearrange(outputs, 'B L C -> (B L) C')
        loss = self.criterion(outputs_, arrs_)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        score = 0
        outputs = torch.argmax(outputs, dim=2)
        outputs = outputs.cpu().numpy()
        arrs = arrs.cpu().numpy()
        for len_arr, output, arr in zip(len_arrs, outputs, arrs):
            gt_one_hot = [arr[i] for i in range(len(len_arr)) for _ in range(len_arr[i])]
            gt_seq = []
            for i in range(len(gt_one_hot)):
                if gt_one_hot[i] == 1:
                    gt_seq.append(i)
            out_one_hot = [output[i] for i in range(len(len_arr)) for _ in range(len_arr[i])]
            out_seq = []
            for i in range(len(out_one_hot)):
                if out_one_hot[i] == 1:
                    out_seq.append(i)
            score += f1(out_seq, gt_seq)
        score /= arrs.shape[0]
        self.log("f1_score", score, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

