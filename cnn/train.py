# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# Model & Dataset
from dataset import LitTSDataset
from model import LitConv1D
# Utils
from pathlib import Path
from argparse import ArgumentParser

# Parse Arguments
parser = ArgumentParser()
parser.add_argument('-r', '--resume', type=Path)
parser.add_argument('-g', '--gpu', type=str, default='1', help='the id of gpu(s) to be used')
parser.add_argument('--no-check', action='store_true', help='skip pytorch-lightning sanity check')
args = parser.parse_args()

train_files = [Path(__file__).parent.parent / f'data/fold_{i}.csv' for i in [2,3,4,5]]
test_file = [Path(__file__).parent.parent / 'data/fold_1.csv']

ds = LitTSDataset(train_files, test_file, batch_size=128)#, fasttext='pretrained')
model = LitConv1D(ds.fasttext, 100, lr=1e-3)

# Callbacks
checkpoint_callback = ModelCheckpoint(monitor='f1_score', save_top_k=1, mode='max')
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=150, mode="min")
callbacks = [checkpoint_callback, early_stop_callback]
# Additional Trainer Configs
kwargs = dict()
if args.resume:
    kwargs['resume_from_checkpoint'] = args.resume 
if args.no_check:
    kwargs['num_sanity_val_steps'] = 0
kwargs['default_root_dir'] = Path(__file__).parent
kwargs['progress_bar_refresh_rate'] = 1
kwargs['gpus'] = args.gpu
kwargs['callbacks'] = callbacks

# Training
trainer = pl.Trainer(**kwargs)
trainer.fit(model, ds)
# trainer.test(ckpt_path="best")
