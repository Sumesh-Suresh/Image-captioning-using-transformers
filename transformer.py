# Credit to the CS-231n course at Stanford, from which this assignment is adapted
import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
       
        super().__init__()
        self.embed_dim = embed_dim
        # projectin layer for query, key and value is embed_dim
        self.query_proj = torch.nn.Linear(in_features=embed_dim,out_features=embed_dim,bias=False)
        self.key_proj = torch.nn.Linear(in_features=embed_dim,out_features=embed_dim,bias=False)
        self.value_proj = torch.nn.Linear(in_features=embed_dim,out_features=embed_dim,bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape
       
        # Computing attention
        #project query, key and value 
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        #compute dot-product attention. 
        #Expected shape of dot_product is (N, S, T)
        dot_product = torch.bmm(self.query_proj,torch.transpose(self.key_proj,dim0=1,dim1=2))
        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            # If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # multiplying with a large negative number (-1e6) in inputs to make softmax 0.
            additive_mask = (1 - attn_mask) * (-1e6)
            dot_product += additive_mask
        
        # apply softmax, dropout, and use value
        y = self.dropout(torch.matmul(F.softmax(dot_product/torch.sqrt(D)),value))
        return y  

class MultiHeadAttentionLayer(AttentionLayer):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
       
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads

        # Initializing the layers and parameters to perform attention
        self.head_proj = torch.nn.Linear(embed_dim,embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        

        assert key.shape == value.shape

        # Computing multi-head attention
 
        #project query, key and value
        #after projection, split the embedding across num_heads
        #eg - expected shape for value is (N, H, T, D/H)
        # N, S, D
        # (N, S, D/H), (N, S, D/H), (N, S, D/H), ..... H times
        # Concatenate on dim=1, N, H, S, D/H

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)


        query_split = torch.split(query, split_size_or_sections=D//self.num_heads, dim=2)
        key_split = torch.split(key, split_size_or_sections=D//self.num_heads, dim=2)
        value_split = torch.split(value, split_size_or_sections=D//self.num_heads, dim=2)

        query = torch.stack(query_split,dim=1)
        value = torch.stack(value_split,dim=1)
        key   = torch.stack(key_split,dim=1)

        dot_product = (query@torch.transpose(key,dim0=2,dim1=3))

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            # Hint : If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # multiplying with a large negative number (-1e6) in inputs to make softmax 0.
            additive_mask = (1 - attn_mask) * (-1e6)
            dot_product += additive_mask.to(dot_product.device)
        
        # apply softmax, dropout, and use value
        denominator = torch.tensor(D/self.num_heads)

        y = self.dropout(torch.matmul(F.softmax(dot_product/torch.sqrt(denominator)),value))
        Y_tuples = torch.split(y,split_size_or_sections=1,dim=1)
        output = torch.cat(Y_tuples,dim=3).squeeze(1)
        output =self.head_proj(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoding = torch.nn.Embedding(num_embeddings=max_len,embedding_dim=embed_dim) 
        self.dropout = torch.nn.Dropout(dropout)
      
    def forward(self, x):
        N, S, D = x.shape
        # adding the encoding to x
        
        output = x + self.encoding(torch.arange(S).to(x.device))
        output = self.dropout(output)
   
        return output


class SelfAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
    
        self.self_attn = MultiHeadAttentionLayer(embed_dim=input_dim,num_heads=num_heads,dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm = torch.nn.LayerNorm(input_dim)
        
       
    def forward(self, seq, mask):
       
        x = self.self_attn(query = seq,key =seq,value=seq,attn_mask=mask)
        x= self.dropout(x)
        x= seq+x
        out = self.layernorm(x)
        return out

class CrossAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()

        self.cross_attn = MultiHeadAttentionLayer(embed_dim=input_dim,num_heads=num_heads,dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        
        self.norm = torch.nn.LayerNorm(input_dim)
       
    def forward(self, seq, cond):
        
        x = self.cross_attn(seq, cond, cond) # check order
        x = self.dropout(x)
        x= seq + x
        out = self.norm(x)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        
        
        self.mlp = torch.nn.Sequential(nn.Linear(input_dim,dim_feedforward),nn.ReLU(),nn.Dropout(dropout),nn.Linear(dim_feedforward,input_dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
       

    def forward(self, seq):
         
        x = self.mlp(seq)
        x= self.dropout(x)
        x = seq + x
        out = self.norm(x)

        
        return out

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, cond, mask):
        out = self.self_atn_block(seq, mask)
        
        out = self.cross_atn_block(out, cond)
        return self.feedforward_block(out)
       
class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4,
                 num_layers=2, max_length=50, device = 'cuda'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension of input image feature vectors.
        - embed_dim: Embedding dimension of the transformer.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        print('null index : ',word_to_idx["<NULL>"])
        self._start = word_to_idx.get("<START>", None)
        self.idx_to_word = idx_to_word
        
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, captions):
        
        feature_embedding = self.feature_embedding(features)
        feature_embedding  = torch.unsqueeze(feature_embedding,dim=1)

        caption_embedding = self.caption_embedding(captions)
        caption_embedding = self.positional_encoding(caption_embedding)

        return feature_embedding, caption_embedding

    def get_causal_mask(self, _len):
        
        a = torch.ones((_len,_len))
        mask = torch.tril(a)
        return mask
                                      
    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.
        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)
        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        features_embed, captions_embed = self.get_data_embeddings(features, captions)
        mask = self.get_causal_mask(captions_embed.shape[1])
        mask.to(captions_embed.dtype)
        
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, mask=mask)

        scores = self.score_projection(output)
        return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.
        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


