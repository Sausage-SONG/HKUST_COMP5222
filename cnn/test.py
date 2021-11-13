# Lightning
import pytorch_lightning as pl
# Model & Dataset
from dataset import LitTSDataset
from model import LitConv1D
# Utils
from pathlib import Path
from argparse import ArgumentParser

# Reset Prediction File
with open('prediction.txt', 'w') as f:
    pass

# Parse Arguments
parser = ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=Path)
parser.add_argument('-f', '--fold', type=int, help='the fold for testing')
args = parser.parse_args()

test_file = [Path(__file__).parent.parent / f'data/fold_{args.fold}.csv']
ds = LitTSDataset([], test_file, batch_size=100, fasttext='pretrained')
model = LitConv1D.load_from_checkpoint(args.checkpoint, fasttext=ds.fasttext, in_channels=300)

trainer = pl.Trainer()
trainer.validate(model=model, datamodule=ds)

Path('prediction.txt').rename(f'prediction_{args.fold}.txt')