import math
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertAttention,
    BertLayer,
    BertEncoder,
    BertModel,
    BertForSequenceClassification,
)

from utils.utils import batch_kron


class KroneckerLayer:
    def __init__(
        self,
        r: int,
        kronecker_dropout: float,
        merge_weights: bool,
        tie_weights: bool,
    ):
        """
        Simple Kronecker product layer, as a first step we are only implementing pure tensors i.e. a \otimes b
        """
        # TODO : Add support for general tensors
        # note r must divide the dimension
        # This is the simple case but we can allow r to be a list[r_1,r_2] satisfying the required conditions.

        self.r = r
        # Optional dropout
        if kronecker_dropout > 0.0:
            self.kronecker_dropout = nn.Dropout(p=kronecker_dropout)
        else:
            self.kronecker_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.tie_weights = tie_weights


class Linear(nn.Linear, KroneckerLayer):
    # Kronecker implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        kronecker_alpha: int = 1,
        kronecker_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        tie_weights: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        KroneckerLayer.__init__(
            self,
            r=r,
            kronecker_dropout=kronecker_dropout,
            merge_weights=merge_weights,
            tie_weights=tie_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.tie_weights = tie_weights
        # Actual trainable parameters
        if r > 0:
            assert in_features % r == 0 and out_features % r == 0
            self.kronecker_A = nn.Parameter(
                self.weight.new_zeros((r, int(in_features / r)))
            )
            self.kronecker_B = nn.Parameter(
                self.weight.new_zeros((int(out_features / r), r))
            )
            self.scaling = kronecker_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "kronecker_A"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but does not affect performance
            if self.tie_weights is False:
                nn.init.kaiming_uniform_(self.kronecker_A, a=math.sqrt(5))
                nn.init.zeros_(self.kronecker_B)
            else:
                assert self.kronecker_B.shape == self.kronecker_A.T.shape
                nn.init.zeros_(self.kronecker_B)
                nn.init.zeros_(
                    self.kronecker_A
                )  # this will likely not work but I do not of a good initialization

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (
                        T(torch.kron(self.kronecker_B, self.kronecker_A)) * self.scaling
                    )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (
                        T(torch.kron(self.kronecker_B, self.kronecker_A)) * self.scaling
                    )
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                self.kronecker_dropout(x)
                @ (torch.kron(self.kronecker_B, self.kronecker_A)).transpose(0, 1)
            ) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, KroneckerLayer):
    # Kronecker implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        kronecker_alpha: int = 1,
        kronecker_dropout: float = 0.0,
        enable_kronecker: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        tie_weights: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        KroneckerLayer.__init__(
            self,
            r=r,
            kronecker_dropout=kronecker_dropout,
            merge_weights=merge_weights,
            tie_weights=tie_weights,
        )
        assert (
            out_features % len(enable_kronecker) == 0
        ), "The length of enable_lora must divide out_features"
        self.enable_kronecker = enable_kronecker
        self.fan_in_fan_out = fan_in_fan_out
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.kronecker_alpha = kronecker_alpha
        self.tie_weights = tie_weights

        # Actual trainable parameters
        if r > 0 and any(enable_kronecker):
            self.kronecker_A = self.weight.new_zeros(
                (r * sum(enable_kronecker), int(in_features / r))
            )
            self.kronecker_B = self.weight.new_zeros(
                ((out_features // len(enable_kronecker) * sum(enable_kronecker)) // r),
                r,
            )

            self.kronecker_A = nn.Parameter(
                self.kronecker_A.reshape(
                    sum(self.enable_kronecker), self.r, int(self.in_features / self.r)
                )
            )
            self.kronecker_B = nn.Parameter(
                self.kronecker_B.reshape(sum(self.enable_kronecker), -1, self.r)
            )
            self.scaling = self.kronecker_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.kronecker_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_kronecker), -1)
            self.kronecker_ind[enable_kronecker, :] = True
            self.kronecker_ind = self.kronecker_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "kronecker_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            if self.tie_weights is False:
                nn.init.kaiming_uniform_(self.kronecker_A, a=math.sqrt(5))
                nn.init.zeros_(self.kronecker_B)
            else:
                assert self.kronecker_B.shape == self.kronecker_A.transpose(1, 2)
                nn.init.zeros_(
                    self.kronecker_A
                )  # this will likely not work. I do not know of a proper initialization scheme
                nn.init.zeros_(self.kronecker_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.kronecker_ind), *x.shape[1:]))
        result[self.kronecker_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = batch_kron(self.kronecker_A, self.kronecker_B).reshape(
            -1, self.in_features
        )
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_kronecker):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_kronecker):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                    self.kronecker_dropout(x) @ T(self.merge_AB().T) * self.scaling
                )
            return result


# for a wide useable standpoint, it would be better to inject our method into the OG transformers code instead of creating our custom classes


class KroneckerBertSelfAttention(BertSelfAttention):
    def __init__(
        self,
        config,
        apply_kron_q,
        apply_kron_k,
        apply_kron_v,
        r,
        kron_alpha,
        kronecker_dropout=0.0,
        tie_weights=False,
        position_embedding_type=None,
    ):
        super().__init__(config)
        self.apply_kron_q = apply_kron_q
        self.apply_kron_k = apply_kron_k
        self.apply_kron_v = apply_kron_v
        self.r = r
        self.kron_alpha = kron_alpha
        self.tie_weights = tie_weights
        self.kronecker_dropout = kronecker_dropout

        if self.apply_kron_q:
            self.query = Linear(
                config.hidden_size,
                self.all_head_size,
                self.r,
                kronecker_alpha=self.kron_alpha,
                tie_weights=self.tie_weights,
                kronecker_dropout=self.kronecker_dropout,
            )
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)

        # for even more parameter efficiency do not use adaptation to key
        if self.apply_kron_k:
            self.key = Linear(
                config.hidden_size,
                self.all_head_size,
                self.r,
                kronecker_alpha=self.kron_alpha,
                tie_weights=self.tie_weights,
                kronecker_dropout=self.kronecker_dropout,
            )
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)

        if self.apply_kron_v:
            self.value = Linear(
                config.hidden_size,
                self.all_head_size,
                self.r,
                kronecker_alpha=self.kron_alpha,
                tie_weights=self.tie_weights,
                kronecker_dropout=self.kronecker_dropout,
            )
        else:
            self.value = nn.Linear(config.hidden_size, self.all_head_size)


class KroneckerBertAttention(BertAttention):
    def __init__(
        self,
        config,
        apply_kron_q,
        apply_kron_k,
        apply_kron_v,
        r,
        kron_alpha,
        kronecker_dropout=0.0,
        tie_weights=False,
        position_embedding_type=None,
    ):
        super().__init__(config)
        self.self = KroneckerBertSelfAttention(
            config,
            apply_kron_q=apply_kron_q,
            apply_kron_k=apply_kron_k,
            apply_kron_v=apply_kron_v,
            r=r,
            kron_alpha=kron_alpha,
            kronecker_dropout=kronecker_dropout,
            tie_weights=tie_weights,
            position_embedding_type=position_embedding_type,
        )


class KroneckerBertLayer(BertLayer):
    def __init__(
        self,
        config,
        apply_kron_q,
        apply_kron_k,
        apply_kron_v,
        r,
        kron_alpha,
        kronecker_dropout=0.0,
        tie_weights=False,
    ):
        super().__init__(config)
        self.attention = KroneckerBertAttention(
            config,
            apply_kron_q=apply_kron_q,
            apply_kron_k=apply_kron_k,
            apply_kron_v=apply_kron_v,
            r=r,
            kron_alpha=kron_alpha,
            kronecker_dropout=kronecker_dropout,
            tie_weights=tie_weights,
        )


class KroneckerBertEncoder(BertEncoder):
    def __init__(
        self,
        config,
        apply_kron_q,
        apply_kron_k,
        apply_kron_v,
        r,
        kron_alpha,
        kronecker_dropout=0.0,
        tie_weights=False,
    ):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList(
            [
                KroneckerBertLayer(
                    config,
                    apply_kron_q=apply_kron_q,
                    apply_kron_k=apply_kron_k,
                    apply_kron_v=apply_kron_v,
                    r=r,
                    kron_alpha=kron_alpha,
                    kronecker_dropout=kronecker_dropout,
                    tie_weights=tie_weights,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )


class KroneckerBertModel(BertModel):
    def __init__(
        self,
        config,
        apply_kron_q,
        apply_kron_k,
        apply_kron_v,
        r,
        kron_alpha,
        tie_weights=False,
        kronecker_dropout=0.0,
        add_pooling_layer=True,
    ):
        super().__init__(config)
        self.config = config
        self.encoder = KroneckerBertEncoder(
            config,
            apply_kron_q=apply_kron_q,
            apply_kron_k=apply_kron_k,
            apply_kron_v=apply_kron_v,
            r=r,
            kron_alpha=kron_alpha,
            kronecker_dropout=kronecker_dropout,
            tie_weights=tie_weights,
        )


class KroneckerBertForSequenceClassification(BertForSequenceClassification):
    def __init__(
        self,
        config,
        apply_kron_q,
        apply_kron_k,
        apply_kron_v,
        r,
        kron_alpha,
        kronecker_dropout=0.0,
        tie_weights=False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = KroneckerBertModel(
            config,
            apply_kron_q=apply_kron_q,
            apply_kron_k=apply_kron_k,
            apply_kron_v=apply_kron_v,
            r=r,
            kron_alpha=kron_alpha,
            kronecker_dropout=kronecker_dropout,
            tie_weights=tie_weights,
        )
