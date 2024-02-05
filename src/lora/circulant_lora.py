#TODO: Add support for GPT2 style transformer layers

import math
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertLayer, BertEncoder, BertModel, BertForSequenceClassification

from utils.utils import circulant_multiply, circulant

class CirculantLayer():
    def __init__(
        self,
        circulant_dropout: float,
        merge_weights: bool,
        trainable_alpha : bool,
        use_prod : bool
    ):
        """
        Simple Circulant layer
        """

        # Optional dropout
        if circulant_dropout > 0.:
            self.circulant_dropout = nn.Dropout(p=circulant_dropout)
        else:
            self.circulant_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.trainable_alpha = trainable_alpha
        self.use_prod = use_prod



class Linear(nn.Linear, CirculantLayer):
    # Circulant implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        circ_alpha: float = 1.,
        circulant_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        trainable_alpha : bool = False,
        use_prod : bool = True,
        init_zero : bool = False,
        use_circ : bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CirculantLayer.__init__(self, circulant_dropout=circulant_dropout,
                           merge_weights=merge_weights, trainable_alpha = trainable_alpha,
                           use_prod = use_prod)

        self.fan_in_fan_out = fan_in_fan_out
        self.trainable_alpha = trainable_alpha
        self.use_prod = use_prod
        self.init_zero = init_zero
        self.use_circ = use_circ
        # Actual trainable parameters

        assert in_features == out_features
        self.circulant_A = nn.Parameter(self.weight.new_zeros(in_features))
        self.scaling = circ_alpha # I do not know if we need scaling in this case
        if self.trainable_alpha :
          self.scaling = nn.Parameter(torch.Tensor([0.0]))
        if self.use_prod :
          self.circulant_B = nn.Parameter(self.weight.new_zeros(out_features))
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.init_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def init_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'circulant_B'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but does not affect performance
            nn.init.normal_(self.circulant_A) # bert style initialization mean =0, std = .02
            nn.init.zeros_(self.circulant_B)
        elif self.trainable_alpha is True :
            nn.init.normal_(self.circulant_A)
        elif self.init_zero is True :
            nn.init.zeros_(self.circulant_A)
        elif self.init_zero is False :
            nn.init.normal_(self.circulant_A, a=math.sqrt(5))
        else :
          raise NotImplementedError('Unsupported combination of hyperparameters')



    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.use_circ :
                    if self.use_prod :
                      self.weight.data -= T(circulant(self.circulant_B * self.circulant_A).transpose(0,1)) * self.scaling
                    else :
                      self.weight.data -= T(circulant(self.circulant_A).transpose(0,1)) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.use_circ:
                    if self.use_prod :
                      self.weight.data += T(circulant(self.circulant_B * self.circulant_A).transpose(0,1)) * self.scaling
                    else :
                      self.weight.data += T(circulant(self.circulant_A).transpose(0,1)) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.use_circ and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.use_prod :
              result += (circulant_multiply(self.circulant_B * self.circulant_A, self.circulant_dropout(x))) * self.scaling
            else :
              result += (circulant_multiply(self.circulant_A, self.circulant_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
# for a wide useable standpoint, it would be better to inject our method into the OG transformers code instead of creating our custom classes

class CirculantBertSelfAttention(BertSelfAttention):
    def __init__(self, config, apply_circ_q, apply_circ_k, apply_circ_v, circ_alpha,
                 trainable_alpha, use_prod, init_zero, use_circ, circulant_dropout=0.0,
                 position_embedding_type=None):
        super().__init__(config)
        self.apply_circ_q = apply_circ_q
        self.apply_circ_k = apply_circ_k
        self.apply_circ_v = apply_circ_v
        self.circ_alpha = circ_alpha
        self.circulant_dropout = circulant_dropout
        self.trainable_alpha = trainable_alpha
        self.use_prod = use_prod
        self.init_zero = init_zero
        self.use_circ = use_circ

        if self.apply_circ_q:
            self.query = Linear(config.hidden_size, self.all_head_size, circ_alpha=self.circ_alpha,
                                trainable_alpha=self.trainable_alpha, use_prod=self.use_prod,
                                init_zero=self.init_zero, use_circ=self.use_circ,
                                circulant_dropout=self.circulant_dropout)
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)

      # for even more parameter efficiency do not use adaptation to key
        if self.apply_circ_k:
            self.key = Linear(config.hidden_size, self.all_head_size, circ_alpha=self.circ_alpha,
                                trainable_alpha=self.trainable_alpha, use_prod=self.use_prod,
                                init_zero=self.init_zero, use_circ=self.use_circ,
                                circulant_dropout=self.circulant_dropout)
        else :
            self.key = nn.Linear(config.hidden_size, self.all_head_size)

        if self.apply_circ_v:
            self.value = Linear(config.hidden_size, self.all_head_size, circ_alpha=self.circ_alpha,
                                trainable_alpha=self.trainable_alpha, use_prod=self.use_prod,
                                init_zero=self.init_zero, use_circ=self.use_circ,
                                circulant_dropout=self.circulant_dropout)
        else:
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

class CirculantBertAttention(BertAttention):
    def __init__(self, config, apply_circ_q, apply_circ_k, apply_circ_v, circ_alpha,
                 trainable_alpha, use_prod, init_zero, use_circ, circulant_dropout=0.0,
                 position_embedding_type=None):
        super().__init__(config)
        self.self = CirculantBertSelfAttention(config, apply_circ_q=apply_circ_q, apply_circ_k=apply_circ_k,
                                               apply_circ_v=apply_circ_v, circ_alpha=circ_alpha,
                                               trainable_alpha=trainable_alpha, use_prod=use_prod,
                                               init_zero=init_zero, use_circ=use_circ,
                                               circulant_dropout=circulant_dropout,
                                               position_embedding_type=position_embedding_type
                                               )


class CirculantBertLayer(BertLayer):
    def __init__(self, config, apply_circ_q, apply_circ_k, apply_circ_v, circ_alpha,
                 trainable_alpha, use_prod, init_zero, use_circ, circulant_dropout=0.0):
        super().__init__(config)
        self.attention = CirculantBertAttention(config, apply_circ_q=apply_circ_q, apply_circ_k=apply_circ_k,
                                               apply_circ_v=apply_circ_v, circ_alpha=circ_alpha,
                                               trainable_alpha=trainable_alpha, use_prod=use_prod,
                                               init_zero=init_zero, use_circ=use_circ,
                                               circulant_dropout=circulant_dropout,
                                                )

class CirculantBertEncoder(BertEncoder):
    def __init__(self, config, apply_circ_q, apply_circ_k, apply_circ_v, circ_alpha,
                 trainable_alpha, use_prod, init_zero, use_circ, circulant_dropout=0.0):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([CirculantBertLayer(config, apply_circ_q=apply_circ_q, apply_circ_k=apply_circ_k,
                                               apply_circ_v=apply_circ_v, circ_alpha=circ_alpha,
                                               trainable_alpha=trainable_alpha, use_prod=use_prod,
                                               init_zero=init_zero, use_circ=use_circ,
                                               circulant_dropout=circulant_dropout,
                                                       )
                                        for _ in range(config.num_hidden_layers)]
                                   )


class CirculantBertModel(BertModel):
    def __init__(self, config, apply_circ_q, apply_circ_k, apply_circ_v, circ_alpha,
                 trainable_alpha, use_prod, init_zero, use_circ, circulant_dropout=0.0,
                 add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = CirculantBertEncoder(config, apply_circ_q=apply_circ_q, apply_circ_k=apply_circ_k,
                                            apply_circ_v=apply_circ_v, circ_alpha=circ_alpha,
                                               trainable_alpha=trainable_alpha, use_prod=use_prod,
                                               init_zero=init_zero, use_circ=use_circ,
                                               circulant_dropout=circulant_dropout,
                                            )

class CirculantBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, apply_circ_q, apply_circ_k, apply_circ_v, circ_alpha,
                 trainable_alpha, use_prod, init_zero, use_circ, circulant_dropout=0.0
                 ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = CirculantBertModel(config, apply_circ_q=apply_circ_q, apply_circ_k=apply_circ_k,
                                       apply_circ_v=apply_circ_v, circ_alpha=circ_alpha,
                                               trainable_alpha=trainable_alpha, use_prod=use_prod,
                                               init_zero=init_zero, use_circ=use_circ,
                                               circulant_dropout=circulant_dropout,
                              )
