import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np

class CustomDatasetforTranslation :
    '''
    Creates a dataset for translation:

    src = english dataframe or array or list
    tgt = dialect dataframe or array or list
    tag = region tag
    src_tokenizer = trained src_tokenizer 
    tgt_tokenizer = trained tgt_tokenizer

    '''
    def __init__(self, src, tgt, tag = None, batch_size = 1, src_tokenizer = None, tgt_tokenizer = None, AsPieces = False) :
        self.src = np.array_split(src,len(src)//batch_size) # english
        self.tgt = np.array_split(tgt,len(src)//batch_size) # dialect
        self.tag = np.array_split('<'+tag+'>', len(src)//batch_size) # region tag
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.AsPieces = AsPieces

    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]
        
        if self.tag is not None:
            tag = self.tag[index]
            src = src + tag

        if self.src_tokenizer and self.tgt_tokenizer is not None:
            if self.AsPieces:
                src = [self.src_tokenizer.EncodeAsPieces(_) for _ in src]
                tgt = [self.tgt_tokenizer.EncodeAsPieces(_) for _ in tgt]
                return src, tgt
            else :
                src = [self.src_tokenizer.Encode(_) for _ in src]
                tgt = [self.tgt_tokenizer.Encode(_) for _ in tgt]
                return src, tgt
        else :
            return src, tgt

    def __len__(self):
        return len(self.src)