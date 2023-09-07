#Transformer model 
import torch
from torch import nn
import torch.nn.functional as F
import math


#Layers 

class LayerNorm_seasonal(nn.Module):
    def __init__(self, channels):
        super(LayerNorm_seasonal, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class DecompositionLayer(nn.Module): 
    def __init__(self, kernel_size = 5):
        super(DecompositionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)  # moving average 

    def forward(self, x):
        # padding on both ends of time series
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, :, 0:1].repeat(1, 1, num_of_pads)
        end = x[:, :, -1:].repeat(1, 1, num_of_pads)
        x_padded = torch.cat([front, x, end], dim=2)

        # calculate the trend and seasonal part of the series
        batch_size, n_ROIs, _ = x.shape
        x_trend = torch.zeros_like(x)

        # Loop over the n_ROIs dimension and apply avg pooling to each time series
        for i in range(n_ROIs):
            x_trend[:, i, :] = self.avg(x_padded[:, i, :].unsqueeze(1)).squeeze(1)
        x_seasonal = x - x_trend

        return x_seasonal, x_trend 
      
class MultiheadAttention(nn.Module): #########Standard layers
    def __init__(self, embed_size, heads):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        scores = scores / (self.embed_size ** (1 / 2))
        if mask is not None:
            scores.masked_fill_(mask == 0, float("-1e20"))

        attention = F.softmax(scores, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)

class seasonal_NormalizationLayer(nn.Module):
    """Normalization layer for seasonal part of the series"""
    def __init__(self, channels):
        super(seasonal_NormalizationLayer, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return (x_hat - bias)

class AutoCorrelation(nn.Module):
    """
    This block replaces the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # Find top k autorcorrelation
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # Update correlations
        tmp_corr = torch.softmax(weights, dim=-1)
        # Aggregate
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)

class AutoCorrelationLayer(nn.Module): 
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class Autocorr_TransformerBlock(nn.Module): 
    def __init__(self, embed_size, heads, forward_expansion, dim_output, kernel_size = 5, dropout=0.5):
        super(Autocorr_TransformerBlock, self).__init__()
        self.correlation = AutoCorrelationLayer(
            correlation=AutoCorrelation(),
            d_model=embed_size,
            n_heads=heads
        )
        self.decomposition1 = DecompositionLayer(kernel_size=kernel_size)
        self.decomposition2 = DecompositionLayer(kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        intermediate_dim = (forward_expansion * embed_size + dim_output)//2
        padding = (kernel_size - 1) // 2

        self.feed_forward_projection = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size), # Dimension: [forward_expansion * embed_size]
            #Permute(0, 2, 1), # Permute dimensions if needed
            #nn.Conv1d(forward_expansion * embed_size, intermediate_dim, kernel_size=kernel_size, bias=False, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            #nn.Conv1d(intermediate_dim, dim_output, kernel_size=kernel_size, bias=False, padding=padding), 
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #Permute(0, 2, 1), # Permute back to original dimensions if needed
            nn.Linear(forward_expansion * embed_size, dim_output) # Dimension: [dim_output]
        )
        #self.norm = LayerNorm_seasonal(embed_size)

    def forward(self, x, mask):
        correlation_out, _  = self.correlation(x, x, x, mask)
        x = x + self.dropout(correlation_out)
        x, _ = self.decomposition1(x)
        y = self.feed_forward(x)
        sum = x+y
        sum, _ = self.decomposition2(sum)

        out = self.feed_forward_projection(sum)
        return out

class Autocorr_TransformerEncoder(nn.Module):
    def __init__(self, input_dim, intermediate_dims, heads, dropout, forward_expansion):
        super(Autocorr_TransformerEncoder, self).__init__()

        # Create a list of dimensions including the input and final output dimensions
        dims = [input_dim] + intermediate_dims 
        
        # Create the stacked Transformer blocks
        self.blocks = nn.ModuleList([
            Autocorr_TransformerBlock(
                embed_size=dims[i],
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                dim_output=dims[i+1]
            )
            for i in range(len(dims) - 1)
        ])

    def forward(self, x, mask = None):
        for block in self.blocks:
            x = block(x, mask)
        return x

class Autocorr_TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, intermediate_dims, heads, dropout, forward_expansion, kernel_size = 5):
        super(Autocorr_TransformerAutoencoder, self).__init__()

        # Split the intermediate_dims into two equal parts for the encoder and decoder

        encoder_dims = intermediate_dims
        decoder_dims = intermediate_dims[::-1]+[input_dim]  # Reverse the decoder dimensions

        self.encoder = Autocorr_TransformerEncoder(
            input_dim=input_dim,
            intermediate_dims=encoder_dims,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )

        # Note that the input dimension for the decoder is the last dimension of the encoder
        self.decoder = Autocorr_TransformerEncoder(
            input_dim=encoder_dims[1::][-1], #Every dimension but the first one - dimension of the latent space
            intermediate_dims=decoder_dims,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )

        self.latent_dim = intermediate_dims[-1]
        self.decomposition = DecompositionLayer(kernel_size=kernel_size)

        #Physical loss parameters 
        self.a = nn.Parameter(torch.tensor(0.1))  
        self.b = nn.Parameter(torch.tensor(0.1))
        self.c = nn.Parameter(torch.tensor(0.1))
        self.K = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, mask = None):
        encoded_x = self.encoder(x, mask) #encoding
        output = self.decoder(encoded_x, mask) #decoding
        return output, encoded_x
    
    def encode(self, x, mask = None):
        encoded_x = self.encoder(x, mask) # Passing data through encoder
        return encoded_x

class TransformerBlock(nn.Module): 
    def __init__(self, embed_size, heads, forward_expansion, dim_output, kernel_size = 5, dropout=0.5):
        super(TransformerBlock, self).__init__()

        self.attention = MultiheadAttention(embed_size, heads)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.normalization = nn.LayerNorm(embed_size)

        self.feed_forward_projection = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size), # Dimension: [forward_expansion * embed_size]
            #Permute(0, 2, 1), # Permute dimensions if needed
            #nn.Conv1d(forward_expansion * embed_size, intermediate_dim, kernel_size=kernel_size, bias=False, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            #nn.Conv1d(intermediate_dim, dim_output, kernel_size=kernel_size, bias=False, padding=padding), 
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #Permute(0, 2, 1), # Permute back to original dimensions if needed
            nn.Linear(forward_expansion * embed_size, dim_output) # Dimension: [dim_output]
        )
        #self.norm = LayerNorm_seasonal(embed_size)

    def forward(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        x = x + self.dropout(attention_out)
        y = x
        y = self.feed_forward(y)
        sum = x+y
        sum = self.normalization(sum)
        out = self.feed_forward_projection(sum)
        return out

class Transformer_encoder(nn.Module):
    def __init__(self, input_dim, intermediate_dims, heads, dropout, forward_expansion):
        super(Transformer_encoder, self).__init__()

        # Create a list of dimensions including the input and final output dimensions
        dims = [input_dim] + intermediate_dims 
        
        # Create the stacked Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_size=dims[i],
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                dim_output=dims[i+1]
            )
            for i in range(len(dims) - 1)
        ])

    def forward(self, x, mask = None):
        for block in self.blocks:
            x = block(x, mask)
        return x

class Standard_Transformer_Autoencoder(nn.Module):
    def __init__(self, input_dim, intermediate_dims, heads, dropout, forward_expansion, kernel_size = 5):
        super(Standard_Transformer_Autoencoder, self).__init__()

        # Split the intermediate_dims into two equal parts for the encoder and decoder

        encoder_dims = intermediate_dims
        decoder_dims = intermediate_dims[::-1]+[input_dim]  # Reverse the decoder dimensions

        self.encoder = Transformer_encoder(
            input_dim=input_dim,
            intermediate_dims=encoder_dims,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )

        # Note that the input dimension for the decoder is the last dimension of the encoder
        self.decoder = Transformer_encoder(
            input_dim=encoder_dims[1::][-1], #Every dimension but the first one - dimension of the latent space
            intermediate_dims=decoder_dims,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )

        self.latent_dim = intermediate_dims[-1]

        #Physical loss parameters 
        self.a = nn.Parameter(torch.tensor(0.1))  
        self.b = nn.Parameter(torch.tensor(0.1))
        self.c = nn.Parameter(torch.tensor(0.1))
        self.K = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, mask):
        encoded_x = self.encoder(x, mask) #encoding
        output = self.decoder(encoded_x, mask) #decoding
        return output, encoded_x
    
    def encode(self, x, mask):
        encoded_x = self.encoder(x, mask) # Passing data through encoder
        return encoded_x

class Decomposition_TransformerBlock(nn.Module): 
    def __init__(self, embed_size, heads, forward_expansion, dim_output, kernel_size = 5, dropout=0.5):
        super(Decomposition_TransformerBlock, self).__init__()

        self.attention = MultiheadAttention(embed_size, heads)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.decomposition = DecompositionLayer(kernel_size=kernel_size)
        self.feed_forward_projection = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size), # Dimension: [forward_expansion * embed_size]
            #Permute(0, 2, 1), # Permute dimensions if needed
            #nn.Conv1d(forward_expansion * embed_size, intermediate_dim, kernel_size=kernel_size, bias=False, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            #nn.Conv1d(intermediate_dim, dim_output, kernel_size=kernel_size, bias=False, padding=padding), 
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #Permute(0, 2, 1), # Permute back to original dimensions if needed
            nn.Linear(forward_expansion * embed_size, dim_output) # Dimension: [dim_output]
        )
        #self.norm = LayerNorm_seasonal(embed_size)

    def forward(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        x = x + self.dropout(attention_out)
        y, _ = self.decomposition(x)
        z = y
        z = self.feed_forward(z)
        sum = y+z
        sum, _ = self.decomposition(sum)
        out = self.feed_forward_projection(sum)
        return out

class Decomposition_Transformer_encoder(nn.Module):
    def __init__(self, input_dim, intermediate_dims, heads, dropout, forward_expansion):
        super(Decomposition_Transformer_encoder, self).__init__()

        # Create a list of dimensions including the input and final output dimensions
        dims = [input_dim] + intermediate_dims 
        
        # Create the stacked Transformer blocks
        self.blocks = nn.ModuleList([
            Decomposition_TransformerBlock(
                embed_size=dims[i],
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                dim_output=dims[i+1]
            )
            for i in range(len(dims) - 1)
        ])

    def forward(self, x, mask = None):
        for block in self.blocks:
            x = block(x, mask)
        return x
    
class Decomposition_Transformer_Autoencoder(nn.Module):
    def __init__(self, input_dim, intermediate_dims, heads, dropout, forward_expansion, kernel_size = 5):
        super(Decomposition_Transformer_Autoencoder, self).__init__()

        # Split the intermediate_dims into two equal parts for the encoder and decoder

        encoder_dims = intermediate_dims
        decoder_dims = intermediate_dims[::-1]+[input_dim]  # Reverse the decoder dimensions

        self.encoder = Decomposition_Transformer_encoder(
            input_dim=input_dim,
            intermediate_dims=encoder_dims,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )

        # Note that the input dimension for the decoder is the last dimension of the encoder
        self.decoder = Decomposition_Transformer_encoder(
            input_dim=encoder_dims[1::][-1], #Every dimension but the first one - dimension of the latent space
            intermediate_dims=decoder_dims,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )

        self.latent_dim = intermediate_dims[-1]

        #Physical loss parameters 
        self.a = nn.Parameter(torch.tensor(0.1))  
        self.b = nn.Parameter(torch.tensor(0.1))
        self.c = nn.Parameter(torch.tensor(0.1))
        self.K = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, mask):
        encoded_x = self.encoder(x, mask) #encoding
        output = self.decoder(encoded_x, mask) #decoding
        return output, encoded_x
    
    def encode(self, x, mask):
        encoded_x = self.encoder(x, mask) # Passing data through encoder
        return encoded_x
