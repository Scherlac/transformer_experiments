# transformer model experiment
# src: https://www.youtube.com/watch?v=ISNdQcPhsts
# SRC: https://github.com/hkproj/pytorch-transformer
# implementing input embedding and positional encoding layers
# ![img](ModalNet-21.png)
# the paper "Attention is all you need" introduces these components in detail
# https://arxiv.org/abs/1706.03762

# other sources to verify the implementation:
# SRC: https://nlp.seas.harvard.edu/2018/04/03/attention.html
# SRC: https://jalammar.github.io/illustrated-transformer/
# SRC: https://github.com/harvardnlp/annotated-transformer
# on tokenizer:
# https://www.youtube.com/watch?v=zduSFxRajkE
# https://github.com/karpathy/minbpe/blob/master/exercise.md



import torch
import torch.nn as nn
import torch.optim as optim
import math

class InputEmbedding(nn.Module):
    def __init__(
            self, 
            d_model: int,
            vocab_size: int,
            ):
        """
        Input embedding layer for transformer model.
        Args:
            d_model (int): Dimension of the model input.
            embedding_dim (int): Dimension of the embedding output.
        """
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def init_weights(
        self,
        ):
        nn.init.xavier_uniform_(self.embedding.weight)
        

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            d_model: int,
            max_len: int = 5000,
            dropout: float = 0.1
            ):
        """
        Positional encoding layer for transformer model.
        Args:
            d_model (int): Dimension of the model input.
            max_len (int): Maximum length of the input sequence.
            dropout (float): Dropout rate.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # for positional encoding, we build a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add a batch dimension
        pe = pe.unsqueeze(0) # we might need to transpose - .transpose(0, 1)
        self.register_buffer('pe', pe)

    def init_weights(
        self,
        ):
        pass


    def forward(self, x):
        # x = x + self.pe[:x.size(0), :]
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            eps: float = 1e-6,
            ):
        """
        Layer normalization layer.
        Args:
            d_model (int): Dimension of the model input.
            eps (float): Epsilon value for numerical stability.
        """
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # scale parameter
        self.bias = nn.Parameter(torch.zeros(d_model)) # shift parameter (bias)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForward(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            d_ff: int, 
            dropout: float = 0.1,
            ):
        """
        Position-wise feedforward layer.
        Args:
            d_model (int): Dimension of the model input.
            d_ff (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
        """
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff) # this is the W_1 matrix and b_1 bias (by default bias is True)
        self.linear2 = nn.Linear(d_ff, d_model) # this is the W_2 matrix and b_2 bias

    def init_weights(
        self,
        ):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(self.linear1(x).relu()))
        # x = self.linear1(x)
        # x = x.relu()
        # x = self.dropout(x)
        # x = self.linear2(x)
        # return x

class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            dropout: float = 0.1,
            ):
        """
        Multi-head attention layer.
        Args:
            d_model (int): Dimension of the model input.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        self.linear_q = nn.Linear(d_model, d_model, bias=False) # W_q ...
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def init_weights(
        self,
        ):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Scaled dot-product attention. 
        Args:
            query: Query tensor of shape (..., seq_len_q, d_k).
            key: Key tensor of shape (..., seq_len_k, d_k).
            value: Value tensor of shape (..., seq_len_v, d_k).
            mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k).
            dropout: Optional dropout layer.
        Returns:
            Output tensor of shape (..., seq_len_q, d_k) and attention weights. 

        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model).
            key: Key tensor of shape (batch_size, seq_len, d_model).
            value: Value tensor of shape (batch_size, seq_len, d_model).
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len).
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).

        Note: mask is typically used to mask out certain positions (e.g., padding or future positions in decoder).
        """
        batch_size = query.size(0)

        # linear projections
        # ( batch_size, seq_len, d_model ) --> ( batch_size, seq_len, num_heads, d_k ) --> ( batch_size, num_heads, seq_len, d_k )
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot-product attention
        x, attn = MultiHeadAttention.attention(query, key, value, mask=mask, dropout=self.dropout)

        # concatenate heads and put through final linear layer
        output = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(output)
        del query, key, value
        return output

class ResidualConnection(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1):
        """
        Residual connection layer with dropout.
        Args:
            size (int): Dimension of the model input.
            dropout (float): Dropout rate.
        """
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(size)

    def init_weights(
        self,
        ):
        pass

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        Args:
            x: Input tensor.
            sublayer: A function or layer to apply to the input tensor.
        Returns:
            Output tensor after applying the sublayer and adding the input tensor.
        """
        x_n = self.norm(x)
        return x + self.dropout(sublayer(x_n))

class EncoderBlock(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float = 0.1,
            ):
        """
        Single block of the transformer encoder.
        Args:
            d_model (int): Dimension of the model input.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = ResidualConnection(d_model, dropout)
        self.sublayer2 = ResidualConnection(d_model, dropout)

    def init_weights(
        self,
        ):
        self.self_attn.init_weights()
        self.feed_forward.init_weights()

    def forward(self, x, mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer2(x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(
            self, 
            num_layers: int, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float = 0.1,
            ):
        """
        Transformer encoder consisting of multiple encoder blocks.
        Args:
            num_layers (int): Number of encoder blocks.
            d_model (int): Dimension of the model input.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(d_model)

    def init_weights(
        self,
        ):
        for layer in self.layers:
            layer.init_weights()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float = 0.1,
            ):
        """
        Single block of the transformer decoder.
        Args:
            d_model (int): Dimension of the model input.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.src_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = ResidualConnection(d_model, dropout)
        self.sublayer2 = ResidualConnection(d_model, dropout)
        self.sublayer3 = ResidualConnection(d_model, dropout)

    def init_weights(
        self,
        ):
        self.self_attn.init_weights()
        self.src_attn.init_weights()
        self.feed_forward.init_weights()

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.src_attn(x, memory, memory, src_mask))
        x = self.sublayer3(x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    def __init__(
            self, 
            num_layers: int, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float = 0.1,
            ):
        """
        Transformer decoder consisting of multiple decoder blocks.
        Args:
            num_layers (int): Number of decoder blocks.
            d_model (int): Dimension of the model input.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(d_model)

    def init_weights(
        self,
        ):
        for layer in self.layers:
            layer.init_weights()

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            vocab_size: int,
            ):
        """
        Projection layer to map decoder output to vocabulary size.
        Args:
            d_model (int): Dimension of the model input.
            vocab_size (int): Size of the vocabulary.
        """
        super(ProjectionLayer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def init_weights(
        self,
        ):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # return torch.log_softmax(self.linear(x), dim=-1)
        return self.linear(x)
    
class Transformer(nn.Module):
    def __init__(
            self, 
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int = 512,
            num_heads: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            max_length: int = 5000,
            ):
        """
        Transformer model.
        Args:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            d_model (int): Dimension of the model input.
            num_heads (int): Number of attention heads.
            num_encoder_layers (int): Number of encoder blocks.
            num_decoder_layers (int): Number of decoder blocks.
            dim_feedforward (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of the input sequence.
        """
        super(Transformer, self).__init__()
        # Encoder components
        self.src_embedding = InputEmbedding(d_model, src_vocab_size)
        self.src_pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, dim_feedforward, dropout)
        # Decoder components
        self.tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, dim_feedforward, dropout)
        self.projection = ProjectionLayer(d_model, tgt_vocab_size)

    def init_weights(
        self,
        ):
        self.src_embedding.init_weights()
        self.tgt_embedding.init_weights()
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.projection.init_weights()

    def encode(self, src, src_mask=None):
        src_emb = self.src_embedding(src) 
        src_emb = self.src_pos_encoding(src_emb)
        memory = self.encoder(src_emb, src_mask)
        return memory   

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return output    

    def project(self, x):
        return self.projection(x)   

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        output = self.projection(output)
        return output
        

def main():
    pass


if __name__ == "__main__":
    main()
