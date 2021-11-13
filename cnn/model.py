# Torch & Lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# Utils
import einops
# F1 Score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from eval_tsd import f1

def weight_init(m):

    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Conv1D(nn.Module):

    def __init__(self, in_channels, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=2)

        self.layer_in = nn.Conv1d(in_channels, 4*in_channels, kernel_size=1)
        self.norm_in = nn.BatchNorm1d(4*in_channels)
        self.layer1 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(4*in_channels)
        self.layer2 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(4*in_channels)
        self.layer3 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(4*in_channels)
        self.layer4 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm1d(4*in_channels)
        # self.layer5 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=3, padding=1)
        # self.norm5 = nn.BatchNorm1d(4*in_channels)
        # self.layer6 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=3, padding=1)
        # self.norm6 = nn.BatchNorm1d(4*in_channels)
        # self.layer7 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=3, padding=1)
        # self.norm7 = nn.BatchNorm1d(4*in_channels)
        # self.layer8 = nn.Conv1d(4*in_channels, 4*in_channels, kernel_size=3, padding=1)
        # self.norm8 = nn.BatchNorm1d(4*in_channels)
        self.fc = nn.Linear(4*in_channels, 2)

        self.apply(weight_init)
    
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

        # x = self.layer5(x)
        # x = self.norm5(x)
        # x = self.dropout(self.relu(x))

        # x = self.layer6(x)
        # x = self.norm6(x)
        # x = self.dropout(self.relu(x))

        # x = self.layer7(x)
        # x = self.norm7(x)
        # x = self.dropout(self.relu(x))

        # x = self.layer8(x)
        # x = self.norm8(x)
        # x = self.dropout(self.relu(x))

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

    def forward(self, sentences, len_arrs):

        outputs = self.model(sentences)
        outputs = torch.argmax(outputs, dim=2)
        outputs = outputs.cpu().numpy()
        batch_result = []
        for len_arr, output in zip(len_arrs, outputs):
            out_one_hot = [output[i] for i in range(len(len_arr)) for _ in range(len_arr[i])]
            out_seq = []
            for i in range(len(out_one_hot)):
                if out_one_hot[i] == 1:
                    out_seq.append(i)
            batch_result.append(out_seq)
        return batch_result

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
            with open('prediction.txt', 'a') as f:
                f.write(f'{out_seq}\n')
        score /= arrs.shape[0]
        self.log("f1_score", score, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

