import math
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertLayer, BertEncoder, BertModel, BertForSequenceClassification

from utils.utils import toeplitz, toeplitz_multiplication


class ToeplitzLayer():
    def __init__(
        self,
        toeplitz_dropout: float,
        merge_weights: bool,
        trainable_alpha : bool,

    ):
        """
        Simple Toeplitz layer
        """

        # Optional dropout
        if toeplitz_dropout > 0.:
            self.toeplitz_dropout = nn.Dropout(p=toeplitz_dropout)
        else:
            self.toeplitz_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.trainable_alpha = trainable_alpha



class Linear(nn.Linear, ToeplitzLayer):
    # Toeplitz implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        toep_alpha: int = 1,
        toeplitz_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        trainable_alpha : bool = False,
        use_toep : bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        ToeplitzLayer.__init__(self, toeplitz_dropout=toeplitz_dropout,
                           merge_weights=merge_weights, trainable_alpha = trainable_alpha,
                           )

        self.fan_in_fan_out = fan_in_fan_out
        self.trainable_alpha = trainable_alpha
        self.use_toep = use_toep
        # Actual trainable parameters

        assert in_features == out_features

        self.scaling = toep_alpha 

        if self.trainable_alpha :
          self.scaling = nn.Parameter(torch.Tensor([0.0]))

        self.toeplitz_init_A_r = nn.Parameter(self.weight.new_zeros(in_features))
        self.toeplitz_init_B_r = nn.Parameter(self.weight.new_zeros(in_features))

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.init_parameters()


        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def init_parameters(self):
        nn.Linear.reset_parameters(self)
        self.toeplitz_init_A_r.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.toeplitz_init_B_r)



    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.use_toep :
                    self.weight.data -= T(toeplitz(self.toeplitz_init_B_r, self.toeplitz_init_B_r) @ toeplitz(self.toeplitz_init_A_r, self.toeplitz_init_A_r)) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.use_toep:
                    self.weight.data += T(toeplitz(self.toeplitz_init_B_r, self.toeplitz_init_B_r) @ toeplitz(self.toeplitz_init_A_r, self.toeplitz_init_A_r)) * self.scaling

                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.use_toep and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (toeplitz_multiplication(self.toeplitz_init_B_r, self.toeplitz_init_B_r, \
                                                     toeplitz_multiplication(self.toeplitz_init_A_r,  self.toeplitz_init_A_r, self.toeplitz_dropout(x)))) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
# for a wide useable standpoint, it would be better to inject our method into the OG transformers code instead of creating our custom classes

class ToeplitzBertSelfAttention(BertSelfAttention):
    def __init__(self, config, apply_toep_q, apply_toep_k, apply_toep_v, toep_alpha,
                 trainable_alpha, use_toep, toeplitz_dropout=0.0,
                 position_embedding_type=None, **kwargs):
        super().__init__(config)
        self.apply_toep_q = apply_toep_q
        self.apply_toep_k = apply_toep_k
        self.apply_toep_v = apply_toep_v
        self.toep_alpha = toep_alpha
        self.toeplitz_dropout = toeplitz_dropout
        self.trainable_alpha = trainable_alpha
        self.use_toep = use_toep


        if self.apply_toep_q:
            self.query = Linear(config.hidden_size, self.all_head_size, toep_alpha=self.toep_alpha,
                                trainable_alpha=self.trainable_alpha, use_toep=self.use_toep,
                                toeplitz_dropout=self.toeplitz_dropout, **kwargs)
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)

      # for even more parameter efficiency do not use adaptation to key
        if self.apply_toep_k:
            self.key = Linear(config.hidden_size, self.all_head_size, toep_alpha=self.toep_alpha,
                                trainable_alpha=self.trainable_alpha, use_toep=self.use_toep,
                                toeplitz_dropout=self.toeplitz_dropout, **kwargs)
        else :
            self.key = nn.Linear(config.hidden_size, self.all_head_size)

        if self.apply_toep_v:
            self.value = Linear(config.hidden_size, self.all_head_size, toep_alpha=self.toep_alpha,
                                trainable_alpha=self.trainable_alpha, use_toep=self.use_toep,
                                toeplitz_dropout=self.toeplitz_dropout, **kwargs)
        else:
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

class ToeplitzBertAttention(BertAttention):
    def __init__(self, config, apply_toep_q, apply_toep_k, apply_toep_v, toep_alpha,
                 trainable_alpha, use_toep, toeplitz_dropout=0.0,
                 position_embedding_type=None, **kwargs):
        super().__init__(config)
        self.self = ToeplitzBertSelfAttention(config, apply_toep_q=apply_toep_q, apply_toep_k=apply_toep_k,
                                               apply_toep_v=apply_toep_v, toep_alpha=toep_alpha,
                                               trainable_alpha=trainable_alpha, use_toep=use_toep,
                                               toeplitz_dropout=toeplitz_dropout,
                                               position_embedding_type=position_embedding_type,
                                              **kwargs
                                               )


class ToeplitzBertLayer(BertLayer):
    def __init__(self, config, apply_toep_q, apply_toep_k, apply_toep_v, toep_alpha,
                 trainable_alpha, use_toep, toeplitz_dropout=0.0,  **kwargs):
        super().__init__(config)
        self.attention = ToeplitzBertAttention(config, apply_toep_q=apply_toep_q, apply_toep_k=apply_toep_k,
                                               apply_toep_v=apply_toep_v, toep_alpha=toep_alpha,
                                               trainable_alpha=trainable_alpha, use_toep=use_toep,
                                               toeplitz_dropout=toeplitz_dropout,
                                               **kwargs
                                                )

class ToeplitzBertEncoder(BertEncoder):
    def __init__(self, config, apply_toep_q, apply_toep_k, apply_toep_v, toep_alpha,
                 trainable_alpha, use_toep, toeplitz_dropout=0.0, **kwargs):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([ToeplitzBertLayer(config, apply_toep_q=apply_toep_q, apply_toep_k=apply_toep_k,
                                               apply_toep_v=apply_toep_v, toep_alpha=toep_alpha,
                                               trainable_alpha=trainable_alpha, use_toep=use_toep,
                                               toeplitz_dropout=toeplitz_dropout,
                                               **kwargs
                                                       )
                                        for _ in range(config.num_hidden_layers)]
                                   )


class ToeplitzBertModel(BertModel):
    def __init__(self, config, apply_toep_q, apply_toep_k, apply_toep_v, toep_alpha,
                 trainable_alpha, use_toep, toeplitz_dropout=0.0,
                 add_pooling_layer=True, **kwargs):
        super().__init__(config)
        self.config = config
        self.encoder = ToeplitzBertEncoder(config, apply_toep_q=apply_toep_q, apply_toep_k=apply_toep_k,
                                            apply_toep_v=apply_toep_v, toep_alpha=toep_alpha,
                                               trainable_alpha=trainable_alpha, use_toep=use_toep,
                                               toeplitz_dropout=toeplitz_dropout,
                                               **kwargs,
                                            )

class ToeplitzBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, apply_toep_q, apply_toep_k, apply_toep_v, toep_alpha,
                 trainable_alpha, use_toep, toeplitz_dropout=0.0, **kwargs
                 ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = ToeplitzBertModel(config, apply_toep_q=apply_toep_q, apply_toep_k=apply_toep_k,
                                       apply_toep_v=apply_toep_v, toep_alpha=toep_alpha,
                                               trainable_alpha=trainable_alpha, use_toep=use_toep,
                                               toeplitz_dropout=toeplitz_dropout, **kwargs
                              )

