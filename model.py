import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class GridRefineBlock(nn.Module):
    """
    A lightweight 2D residual block for refining grid logits/features.
    Input/Output: [B, Cg, L, L]
    """
    def __init__(self, channels: int, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        # x: [B, Cg, L, L]
        residual = x
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.norm(x + residual)
        return x


class GridRefiner(nn.Module):
    """
    Stack multiple GridRefineBlocks.
    """
    def __init__(self, channels: int, num_layers: int = 2, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            GridRefineBlock(channels, dropout=dropout, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        x: [B, Cg, L, L]
        mask: optional [B, L] boolean mask indicating valid tokens.
              If provided, we will zero-out invalid rows/cols after each layer.
        """
        if mask is not None:
            # mask2d: [B, 1, L, L]
            mask2d = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1).to(x.dtype)

        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask2d
        return x


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xavier', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class EXCEEDS(nn.Module):
    def __init__(self, config):
        super(EXCEEDS, self).__init__()

        lstm_input_size = config.bert_hid_size
        self.lstm_hid_size = config.lstm_hid_size

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/")

        self.lstm = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.grid_channels = config.grid_channels

        # position embedding
        self.dis_embs = nn.Embedding(20, config.dist_emb_size)

        # pair representation
        self.pair_mlp_feat = MLP(
            n_in=2 * config.lstm_hid_size + config.dist_emb_size,
            n_out=self.grid_channels,
            dropout=config.dropout_rate
        )

        # conditional layerNorm
        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)
        self.mlp0 = MLP(config.lstm_hid_size, config.label_num, config.dropout_rate)

        # refinement
        refine_layers = config.grid_refine_layers
        refine_dropout = config.grid_refine_dropout
        refine_kernel = config.grid_refine_kernel
        self.grid_refiner = GridRefiner(
            channels=self.grid_channels,
            num_layers=refine_layers,
            dropout=refine_dropout,
            kernel_size=refine_kernel
        )
        
        # final classifier
        self.classifier = nn.Linear(self.grid_channels, config.label_num)
        

    def forward(self, bert_inputs, pieces2word, dist_inputs, document_length):
        '''
        :param bert_inputs: [B, L']
        :param pieces2word: [B, L, L']
        :param dist_inputs: [B, L, L]
        :param document_length: [B]
        :return:
        '''
        
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)  # replace each token with max piece representation
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # Bi-LSTM
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, document_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.lstm(packed_embs)  # [B, L, H_lstm]
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=document_length.max())
        
        # Conditional LayerNorm
        word_reps = self.cln(word_reps, word_reps)  # self.cln: LayerNorm, cln: [B, L, H_lstm]
        B, L, H = word_reps.shape
        mask = (torch.arange(L, device=word_reps.device)[None, :] < document_length[:, None])  # [B,L] bool
        word_reps = word_reps * mask.unsqueeze(-1)
        
        # Pair Feature
        x_i = word_reps.unsqueeze(2).expand(-1, -1, length, -1)
        x_j = word_reps.unsqueeze(1).expand(-1, length, -1, -1)
        pair_reps = torch.cat([x_i, x_j], dim=-1)
        dist = self.dis_embs(dist_inputs)
        pair_reps = torch.cat([pair_reps, dist], dim=-1)
        grid_feat = self.pair_mlp_feat(pair_reps)  # [B, L, L, Cg]

        # build valid mask for refinement
        max_len = grid_feat.size(1)
        device = grid_feat.device
        arange = torch.arange(max_len, device=device).unsqueeze(0)
        valid_mask = arange < document_length.unsqueeze(1)  # [B, L]

        # refine with 2D CNN: [B, Cg, L, L]
        grid_feat_2d = grid_feat.permute(0, 3, 1, 2).contiguous()
        grid_feat_2d = self.grid_refiner(grid_feat_2d, mask=valid_mask)
        grid_feat = grid_feat_2d.permute(0, 2, 3, 1).contiguous()  # [B, L, L, Cg]

        # classify to logits
        logits = self.classifier(grid_feat)        # [B, L, L, label_num]
        return logits

