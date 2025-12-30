

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(
            self, 
            dataset, 
            src_tokenizer, 
            tgt_tokenizer, 
            src_lang='en', 
            tgt_lang='de', 
            max_length=128,
            ):
        """
        A PyTorch Dataset for bilingual text data suitable for training translation models.
        Args:
            data (list): A list of dictionaries containing bilingual text data.
            src_tokenizer: Tokenizer for the source language.
            tgt_tokenizer: Tokenizer for the target language.
            src_lang (str): Source language code.
            tgt_lang (str): Target language code.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
    
        self.sos_token_id = torch.Tensor([tgt_tokenizer.token_to_id("[CLS]")]).type(torch.int64)
        self.eos_token_id = torch.Tensor([tgt_tokenizer.token_to_id("[SEP]")]).type(torch.int64)
        self.pad_token_id = torch.Tensor([tgt_tokenizer.token_to_id("[PAD]")]).type(torch.int64)

        # new() received an invalid combination of arguments - got (list, dtype=torch.dtype), but expected one of:
        # * (*, torch.device device = None)
        #     didn't match because some of the keywords were incorrect: dtype
        # * (torch.Storage storage)
        # * (Tensor other)
        # * (tuple of ints size, *, torch.device device = None)
        # * (object data, *, torch.device device = None)

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.tgt_lang]

        src_encoding = self.src_tokenizer.encode(
            src_text,
            # max_length=self.max_length,
            # truncation=True,
            # padding='max_length',
            # return_tensors='pt'
        ).ids

        tgt_encoding = self.tgt_tokenizer.encode(
            tgt_text,
            # max_length=self.max_length,
            # truncation=True,
            # padding='max_length',
            # return_tensors='pt'
        ).ids

        # Convert to torch tensors
        src_ids = torch.tensor(src_encoding, dtype=torch.int64)
        tgt_ids = torch.tensor(tgt_encoding, dtype=torch.int64)

        # Add special tokens for source sequence
        src_padded_length = self.max_length - (len(src_ids) + 2)
        # argument 'fill_value' (position 2) must be Number, not Tensor
        src_padding = torch.full((src_padded_length,), self.pad_token_id.item()).type(torch.int64)
        src_input_ids = torch.cat([self.sos_token_id, src_ids, self.eos_token_id, src_padding], dim=0)

        # Add special tokens for target sequence
        tgt_padded_length = self.max_length - (len(tgt_ids) + 1)
        tgt_padding = torch.full((tgt_padded_length,), self.pad_token_id.item()).type(torch.int64)
        tgt_input_ids = torch.cat([self.sos_token_id, tgt_ids, tgt_padding], dim=0)
        tgt_labels = torch.cat([self.sos_token_id, tgt_ids], dim=0)

        # Add special tokens for target labels
        tgt_padded_length = self.max_length - (len(tgt_ids) + 1)
        tgt_padding = torch.full((tgt_padded_length,), self.pad_token_id.item(), dtype=torch.int64)
        tgt_labels = torch.cat([tgt_ids, self.eos_token_id, tgt_padding], dim=0)

        assert len(src_input_ids) == self.max_length, f"Source input ids length {len(src_input_ids)} does not match max length {self.max_length}"
        assert len(tgt_input_ids) == self.max_length, f"Target input ids length {len(tgt_input_ids)} does not match max length {self.max_length}"
        assert len(tgt_labels) == self.max_length, f"Target labels length {len(tgt_labels)} does not match max length {self.max_length}"

        # Create attention masks for encoder
        encoder_mask = (src_input_ids != self.pad_token_id)
        encoder_mask = encoder_mask.unsqueeze(0)
        encoder_mask = encoder_mask.unsqueeze(0).int()

        # Create decoder mask will be created in the model using subsequent masking
        decoder_mask = (tgt_input_ids != self.pad_token_id)
        decoder_mask = decoder_mask.unsqueeze(0)
        decoder_mask = decoder_mask.unsqueeze(0).int()
        decoder_mask = decoder_mask & causal_mask(tgt_input_ids.size(0))

        return {
            'encoder_input': src_input_ids,
            'decoder_input': tgt_input_ids,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': tgt_labels,
            'src_text': src_text,
            'tgt_text': tgt_text
            }


def causal_mask(size):
    """Create a causal mask for the decoder."""
    # diagonal=0 includes the main diagonal  -> not working, dont know why
    # diagonal=1 includes the main diagonal and the one above it ?? --> it works this way
    mask = torch.tril(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
