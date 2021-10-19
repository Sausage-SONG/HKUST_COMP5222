# Torch & Lightning
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
# Utils
import re
import csv
import einops
import fasttext as ft
import fasttext.util as ftu
from pathlib import Path

class ToxicSpanDataset(Dataset):

    def __init__(self, csv_files, fasttext=None, mode='train'):
        super().__init__()

        self.mode = mode

        # Extract data from the csv files
        self.all_data = []
        for file in csv_files:
            with file.open(newline='') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                # Skip the first line (column headings)
                iterator = enumerate(reader)
                iterator.__next__()
                for idx, line in iterator:
                    arr = line[0].strip()
                    sentence = line[1].rstrip()

                    # Convert the string to python list
                    arr = eval(arr)
                    # To one-hot
                    arr = [1 if i in arr else 0 for i in range(len(sentence))]
                    
                    self.all_data.append([arr, sentence])
        
        if fasttext is None:
            self.fasttext = self.train_fasttext()
        elif isinstance(fasttext, str) and fasttext == 'pretrained':
            ftu.download_model('en', if_exists='ignore')
            self.fasttext = ft.load_model('cc.en.300.bin')
        else:
            self.fasttext = fasttext

    def __len__(self):

        return len(self.all_data)

    def __getitem__(self, idx):

        arr, sentence = self.all_data[idx]
        sentence = self.parse_sentence(sentence, arr)
        len_arr = list(len(word) for word in sentence)
        
        i = 0
        new_arr = []
        for word in sentence:
            new_arr.append(arr[i])
            i += len(word)
        arr = new_arr

        sentence = list(map(self.fasttext.get_word_vector, sentence))
        sentence = torch.tensor(sentence)

        if self.mode != 'train':
            return arr, sentence, len_arr
        else:
            return arr, sentence
    
    def train_fasttext(self):

        corpus_path = Path('corpus.tmp')
        with corpus_path.open('w') as f:
            for _, sentence in self.all_data:
                f.write(str(sentence)+'\n')

        model = ft.train_unsupervised(str(corpus_path))
        corpus_path.unlink()
        return model
    
    def parse_sentence(self, sentence, arr=None):

        p = 0
        fragments = []
        if arr is not None:
            for i in range(len(arr)-1):
                if arr[i] != arr[i+1]:
                    fragments.append(sentence[p:i+1])
                    p = i+1
        fragments.append(sentence[p:])

        words = list(map(lambda x:re.split('(\W)', x), fragments))
        words = [item for sublist in words for item in sublist]
        result = []
        for word in words:
            if len(word) > 0:
                result.append(word)
        assert ''.join(result) == sentence, 'Invalid Parse Sentence Operation!'
        return result

def SeqPad(batch):

    arrs = [torch.tensor(item[0]) for item in batch]
    arrs = pad_sequence(arrs, batch_first=True, padding_value=-1)
    
    sentences = [item[1] for item in batch]
    sentences = pad_sequence(sentences, batch_first=True)
    sentences = einops.rearrange(sentences, 'B L D -> B D L')

    if len(batch[0]) == 3:
        len_arrs = [item[2] for item in batch]
        return arrs, sentences, len_arrs

    return arrs, sentences
    

class LitTSDataset(pl.LightningDataModule):

    def __init__(self, train_files, test_files, batch_size=512, fasttext=None):
        super().__init__()

        self.train_ds = ToxicSpanDataset(train_files, fasttext=fasttext)
        self.fasttext = self.train_ds.fasttext
        self.val_ds = ToxicSpanDataset(test_files, self.fasttext, 'val')
        self.test_ds = ToxicSpanDataset(test_files, self.fasttext, 'test')

        self.batch_size = batch_size

    def prepare_data(self, *args, **kwargs):
        
        pass

    def setup(self, *args, **kwargs):
        
        pass

    def train_dataloader(self):

        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=SeqPad, num_workers=8)

    def val_dataloader(self):

        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=SeqPad, num_workers=8)

    def test_dataloader(self):

        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=SeqPad, num_workers=8)

