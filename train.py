import os
import pathlib
import re
import tqdm
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import Transformer 

import argparse

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from torch.optim import AdamW

data_root = (pathlib.Path(__file__).parent / 'data').resolve()
training_root = (pathlib.Path(__file__).parent / 'runs').resolve()

class TokenizerWrapper:
    def __init__(
            self,
            arguments: argparse.Namespace,
            dataset=None,
            lang: str = "en"
    ):
        self.tokenizer_path = data_root / arguments.tokenizer_path / arguments.tokenizer_file.format(lang=lang)
        self.vocab_size = arguments.vocab_size
        self.lang = lang

        if self.tokenizer_path.exists():
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        else:
            self.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            self.tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            self.trainer = WordLevelTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
                min_frequency=2
            )
            self.train_tokenizer(dataset)

    def get_all_texts(self, dataset):
        for sample in dataset:
            yield sample['translation'][self.lang]

    def train_tokenizer(self, dataset):
        self.tokenizer.train_from_iterator(
            self.get_all_texts(dataset),
            trainer=self.trainer
        )
        self.tokenizer.save(str(self.tokenizer_path))


    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--tokenizer-path', type=str, default="tokens", help='Path to save/load the tokenizer')
        parser.add_argument('--tokenizer-file', type=str, default='tokenizer_{lang}.json', help='Tokenizer file name pattern with {lang} placeholder')
        parser.add_argument('--vocab-size', type=int, default=30522, help='Vocabulary size for the tokenizer')

class DatasetWrapper:
    def __init__(
            self,
            args: argparse.Namespace

    ):
        self.max_length = args.max_length

        # Load dataset
        dataset = load_dataset(args.dataset_name, f'{args.src_lang}-{args.tgt_lang}', split='train')

        # Initialize and train tokenizer
        self.tokenizer_src = TokenizerWrapper(args, dataset=dataset, lang=args.src_lang)
        self.tokenizer_tgt = TokenizerWrapper(args, dataset=dataset, lang=args.tgt_lang)

        print(f"Source tokenizer vocab size: {len(self.tokenizer_src.tokenizer.get_vocab())}")

        validation_size = int(len(dataset) * args.validation_split)
        test_size = int(len(dataset) * args.test_split)
        train_size = len(dataset) - validation_size - test_size

        print(f"Dataset sizes - Train: {train_size}, Validation: {validation_size}, Test: {test_size}")
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, validation_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Create BilingualDataset instances
        self.bilingual_train_dataset = BilingualDataset(
            train_dataset,
            src_tokenizer=self.tokenizer_src.tokenizer,
            tgt_tokenizer=self.tokenizer_tgt.tokenizer,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_length=args.max_length
            
        )
        self.bilingual_val_dataset = BilingualDataset(
            val_dataset,
            src_tokenizer=self.tokenizer_src.tokenizer,
            tgt_tokenizer=self.tokenizer_tgt.tokenizer,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_length=args.max_length
        )

        max_src_len = max_length(dataset, self.tokenizer_src.tokenizer, args.src_lang)
        max_tgt_len = max_length(dataset, self.tokenizer_tgt.tokenizer, args.tgt_lang)
        
        print(f"Max source length: {max_src_len}")
        print(f"Max target length: {max_tgt_len}")

        assert max_src_len <= args.max_length, f"Source sequences exceed maximum length ({max_src_len} > {args.max_length})"
        assert max_tgt_len <= args.max_length, f"Target sequences exceed maximum length ({max_tgt_len} > {args.max_length})"

        self.train_dataloader = DataLoader(
            self.bilingual_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            # collate_fn=causal_mask
        )
        self.val_dataloader = DataLoader(
            self.bilingual_val_dataset,
            batch_size=1, # args.batch_size, # we use batch size 1 for evaluation
            shuffle=False,
            # collate_fn=causal_mask
        )

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--dataset-name', type=str, default='opus_books', help='Hugging Face dataset name')
        parser.add_argument('--batch-size', type=int, default=24, help='Batch size for training and evaluation')
        parser.add_argument('--src-lang', type=str, default='en', help='Source language for translation dataset')
        parser.add_argument('--tgt-lang', type=str, default='it', help='Target language for translation dataset')
        parser.add_argument('--max-length', type=int, default=350, help='Maximum sequence length for tokenization')
        # validation and test splits for tokenizer training
        parser.add_argument('--validation-split', type=float, default=0.1, help='Proportion of data for validation')
        parser.add_argument('--test-split', type=float, default=0.0, help='Proportion of data for testing')


def max_length(dataset, tokenizer, lang):
    max_len = 0
    for item in dataset:
        text = item['translation'][lang]
        encoding = tokenizer.encode(text)
        seq_len = len(encoding.ids)
        if seq_len > max_len:
            max_len = seq_len
    return max_len

class ModelWrapper:
    # def __init__(
    #         self,
    #         args: argparse.Namespace
    # ):
    #     from transformers import AutoModelForSeq2SeqLM

    #     self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # @staticmethod
    # def add_arguments(parser: argparse.ArgumentParser):
    #     parser.add_argument('--model-name', type=str, default='t5-small', help='Pretrained model name or path')
    def __init__(
            self,
            args: argparse.Namespace
    ):
        self.model = Transformer(
            src_vocab_size=args.vocab_size,
            tgt_vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_length=args.max_length

        )

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--d-model', type=int, default=512, help='Dimension of model embeddings')
        parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
        parser.add_argument('--num-encoder-layers', type=int, default=6, help='Number of encoder layers')
        parser.add_argument('--num-decoder-layers', type=int, default=6, help='Number of decoder layers')
        parser.add_argument('--dim-feedforward', type=int, default=2048, help='Dimension of feedforward layers')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--model-folder', type=str, default='weights', help='Folder to save model weights')
        parser.add_argument('--model-prefix', type=str, default='transmodel_', help='Prefix for saved model files')

def get_model_filepath(args, epoch):
    model_folder = training_root / args.model_folder
    model_folder.mkdir(parents=True, exist_ok=True)
    model_filename = f"{args.model_prefix}{epoch + 1:04d}.pt"
    return model_folder / model_filename

def get_latest_model_filepath(args):
    model_folder = training_root / args.model_folder
    if not model_folder.exists():
        return None
    model_files = list(model_folder.glob(f"{args.model_prefix}*.pt"))
    
    max_epoch = -1
    latest_file = None
    for file in model_files:
        regex = f"{args.model_prefix}(?P<epoch>\\d+)\\.pt"
        match = re.match(regex, file.name)
        if match:
            epoch = int(match.group('epoch')) -1
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = file

    return latest_file

def greedy_decode(model, dataset_wrapper, src, src_mask, max_length, device):
    model.eval()
    src_tokenizer = dataset_wrapper.tokenizer_src.tokenizer
    tgt_tokenizer = dataset_wrapper.tokenizer_tgt.tokenizer

    sos = tgt_tokenizer.token_to_id("[CLS]")
    eos = tgt_tokenizer.token_to_id("[SEP]")
    pad = tgt_tokenizer.token_to_id("[PAD]")

    # Pre-compute source encoding
    encoder_output = model.encode(src, src_mask)
    # Initialize decoder input with SOS token
    decoder_input = torch.cat([
        torch.tensor([[sos]], device=device),  # (1, 1)
    ], dim=1).type_as(src)  # (1, 2)
    while True:
        if decoder_input.size(1) >= max_length:
            break
        tgt_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)  # (1, tgt_seq_len, tgt_seq_len)
        decoder_output = model.decode(decoder_input, encoder_output, src_mask, tgt_mask)  # (1, tgt_seq_len, d_model)
        projection_output = model.project(decoder_output[:, -1:, :])

        _, next_token = torch.max(projection_output, dim=-1)  # (1, 1)
        next_token_id = next_token.item()
        decoder_input = torch.cat([
            decoder_input, 
            next_token,
            ], dim=1).type_as(src)  # (1, tgt_seq_len + 1)
        
        if next_token_id == eos:
            break

    return decoder_input  # (1, generated_seq_len)
    

def run_validation(model, dataset_wrapper, device, tqdm_print_fn):
    model.eval()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=dataset_wrapper.tokenizer_tgt.tokenizer.token_to_id("[PAD]"),
        label_smoothing=0.1
    ).to(device)

    source_texts = []
    target_texts = []
    predicted_texts = []

    with torch.no_grad():
        for batch in dataset_wrapper.val_dataloader:
            src = batch['encoder_input'].to(device) # (batch_size, src_seq_len)
            # tgt = batch['decoder_input'].to(device) # (batch_size, tgt_seq_len)
            src_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, src_seq_len)
            # tgt_mask = batch['decoder_mask'].to(device) # (batch_size, 1, tgt_seq_len, tgt_seq_len)

            assert src.size(0) == 1, "Validation batch size should be 1"

            encoder_output = greedy_decode(
                model,
                dataset_wrapper,
                src,
                src_mask,
                max_length=dataset_wrapper.max_length,
                device=device
            )

            source_text = batch['src_text'][0]
            source_texts.append(source_text)
            target_text = batch['tgt_text'][0]
            target_texts.append(target_text)

            output_text = dataset_wrapper.tokenizer_tgt.tokenizer.decode(encoder_output[0].detach().cpu().numpy(), skip_special_tokens=True)
            predicted_texts.append(output_text)

            tqdm_print_fn('-'*50)
            tqdm_print_fn(f"Source: {source_text}")
            tqdm_print_fn(f"Target: {target_text}")
            tqdm_print_fn(f"Predicted: {output_text}")

            break  # Remove this break to run on the entire validation set

    return total_loss / len(dataset_wrapper.val_dataloader)


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TokenizerWrapper.add_arguments(parser)
    DatasetWrapper.add_arguments(parser)
    ModelWrapper.add_arguments(parser)
    parser.add_argument('--experiment-name', type=str, default='transformer_experiment', help='Name of the experiment')
    parser.add_argument('--learning-rate', type=float, default=4e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--data-path', type=str, default=str(data_root), help='Path to the data directory')
    parser.add_argument('--training-path', type=str, default=str(training_root), help='Path to the training runs directory')

    args = parser.parse_args()

    if args.data_path:
        data_root = pathlib.Path(args.data_path).resolve()
        data_root.mkdir(parents=True, exist_ok=True)

    if args.training_path:
        training_root = pathlib.Path(args.training_path).resolve()
        training_root.mkdir(parents=True, exist_ok=True)

    dataset_wrapper = DatasetWrapper(args)
    model_wrapper = ModelWrapper(args)
    model = model_wrapper.model
    model.init_weights()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    writer = SummaryWriter(args.experiment_name)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        eps=1e-9
        )

    latest_model_info = get_latest_model_filepath(args)
    if latest_model_info is not None:
        print(f"Loading model from {latest_model_info}")
        state = torch.load(latest_model_info)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state.get('epoch', 0) + 1
        global_step = state.get('global_step', 0)
    else:
        initial_epoch = 0
        global_step = 0

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=dataset_wrapper.tokenizer_tgt.tokenizer.token_to_id("[PAD]"),
        label_smoothing=0.1
        ).to(device)

    for epoch in range(initial_epoch, args.num_epochs):

        model.train()
        
        total_loss = 0.0
        batch_iterator = tqdm.tqdm(dataset_wrapper.train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(batch_iterator):
            src = batch['encoder_input'].to(device) # (batch_size, src_seq_len)
            tgt = batch['decoder_input'].to(device) # (batch_size, tgt_seq_len)
            src_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, src_seq_len)
            tgt_mask = batch['decoder_mask'].to(device) # (batch_size, 1, tgt_seq_len, tgt_seq_len)
            src_texts = batch['src_text']
            tgt_texts = batch['tgt_text']

            encoder_output = model.encode(src, src_mask) # (batch_size, src_seq_len, d_model)
            decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask) # (batch_size, tgt_seq_len, d_model)
            projection_output = model.project(decoder_output) # (batch_size, tgt_seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch_size, tgt_seq_len)
            
            # (batch_size, tgt_seq_len, tgt_vocab_size) -> (batch_size * tgt_seq_len, tgt_vocab_size)
            loss = loss_fn(projection_output.view(-1, projection_output.size(-1)), label.view(-1))
            batch_iterator.set_postfix({'loss': f"{loss.item():6.3f}"})

            # Log outputs for loss computation
            writer.add_scalar('Batch Loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            total_loss += loss.item()

            # Run validation every 500 steps
            if (batch_idx + 1) % 100 == 0:
                val_loss = run_validation(
                    model,
                    dataset_wrapper,
                    device,
                    batch_iterator.write
                )
                writer.add_scalar('Validation Loss', val_loss, global_step)
                writer.flush()
                model.train()


        # Save model checkpoint
        model_filepath = get_model_filepath(args, epoch)
        torch.save(
            {'epoch': epoch,
             'global_step': global_step,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()
            }, model_filepath)
        print(f"Model saved to {model_filepath}")







