import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import ( # type: ignore
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)

from utils.utils import circulant_multiply, compute_root, skew_mult


class MixedLayer(nn.Module):
    def __init__(
        self,
        circulant_dropout: float,
        trainable_alpha : bool,
        feats : int,
        root_vec = None,
        scaling : float = 1.
    ):
        super().__init__()
        """
        Simple Circulant+SkewCirculant layer
        """

        # Optional dropout
        if circulant_dropout > 0.:
            self.circulant_dropout = nn.Dropout(p=circulant_dropout)
        else:
            self.circulant_dropout = lambda x: x
        self.trainable_alpha = trainable_alpha
        self.feats = feats
        self.root_vec = root_vec

        self.circulant = nn.Parameter(torch.rand(self.feats))
        self.scaling = scaling 
        if self.trainable_alpha :
          self.scaling = nn.Parameter(torch.Tensor([0.0]))
        self.skew_circulant = nn.Parameter(torch.rand(self.feats))

    def forward(self, x):
        return self.scaling * circulant_multiply(self.circulant, \
                                  skew_mult(self.skew_circulant, self.circulant_dropout(x), self.root_vec)
        )



class MixedLinear(nn.Linear):
    # Circulant implemented in a dense layer
    def __init__(
        self,
        in_features,
        out_features,
        num_adapters : int = 0,
        circ_alpha: float = 1.,
        circulant_dropout: float = 0.,
        trainable_alpha: bool = False,
        root_vec = None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.trainable_alpha = trainable_alpha

        self.root_vec = root_vec
        self.circulant_dropout = circulant_dropout

        # Actual trainable parameters
        self.in_features = in_features
        self.out_features = out_features
        assert in_features == out_features

        self.num_adapters = num_adapters
        # Freezing the pre-trained weight matrix
        # self.linear = nn.Linear(self.features, self.features, bias=self.bias_term )
        self.root_vec = nn.Parameter(root_vec, requires_grad=False)

        # if self.num_adapters > 0 and self.root_vec is not None :
        #   self.root_vec = self.root_vec.to(self.weight.device)

        self.num_adapters = num_adapters
        if self.num_adapters > 0 and self.root_vec is None:
          self.root = compute_root(self.in_features+1)
          self.root_vec = torch.zeros((self.in_features,), dtype=torch.complex64, device=self.weight.device)
          self.root_vec[0] = 1.
          self.root_vec[1] = torch.tensor(np.array([self.root]), dtype=torch.cfloat, device=self.weight.device)
          for i in range(2,len(self.root_vec)):
            self.root_vec[i] = self.root_vec[1]**i


        if self.num_adapters > 0 :
          self.layers = nn.ModuleDict({f'circulant_skew_circ_layer_{i}': \
                                     MixedLayer(circulant_dropout=self.circulant_dropout,
                                      trainable_alpha=self.trainable_alpha,
                                      feats=self.in_features,
                                      root_vec=self.root_vec,
                                    ) for i in range(self.num_adapters)}
                                      )
          self.weight.requires_grad=False

        self.init_parameters()



    def init_parameters(self):
        self.reset_parameters()
        with torch.no_grad():
          for i in range(self.num_adapters):
            nn.init.normal_(self.layers[f'circulant_skew_circ_layer_{i}'].skew_circulant, std=.02)
            nn.init.zeros_(self.layers[f'circulant_skew_circ_layer_{i}'].circulant)

    def forward(self, x: torch.Tensor):
        if self.num_adapters > 0:
            result = F.linear(x, self.weight, bias=self.bias)
            Y = self.layers['circulant_skew_circ_layer_0'](x)
            for i in range(1,self.num_adapters):
              Y += self.layers[f'circulant_skew_circ_layer_{i}'](x)
            result += Y
            del Y
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)
        
### We show an implementation of this layer in Llama-2

class CirculantLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx, apply_circ_q, apply_circ_k, apply_circ_v, apply_circ_o, circ_alpha,
                 trainable_alpha, circulant_dropout, num_adapters, root_vec
                 ):
        super().__init__(config, layer_idx)
        self.apply_circ_q = apply_circ_q
        self.apply_circ_k = apply_circ_k
        self.apply_circ_v = apply_circ_v
        self.apply_circ_o = apply_circ_o
        self.circ_alpha = circ_alpha
        self.circulant_dropout = circulant_dropout
        self.trainable_alpha = trainable_alpha
        self.num_adapters = num_adapters
        self.root_vec = root_vec

        if self.apply_circ_q:
            self.q_proj = MixedLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias,
                                 circ_alpha=self.circ_alpha,
                                trainable_alpha=self.trainable_alpha,
                                circulant_dropout=self.circulant_dropout,
                                 num_adapters=self.num_adapters,
                                 root_vec=self.root_vec
                 )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)

        if self.apply_circ_k:
            self.k_proj = MixedLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias,
                                 circ_alpha=self.circ_alpha,
                                trainable_alpha=self.trainable_alpha,
                                circulant_dropout=self.circulant_dropout,
                                 num_adapters=self.num_adapters,
                                 root_vec=self.root_vec
                 )
        else :
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)

        if self.apply_circ_v:
            self.v_proj = MixedLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias,
                                 circ_alpha=self.circ_alpha,
                                trainable_alpha=self.trainable_alpha,
                                circulant_dropout=self.circulant_dropout,
                                 num_adapters=self.num_adapters,
                                 root_vec=self.root_vec
                 )
        else:
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)

        if self.apply_circ_o:
            self.o_proj = MixedLinear(self.hidden_size, self.hidden_size, bias=config.attention_bias,
                                 circ_alpha=self.circ_alpha,
                                trainable_alpha=self.trainable_alpha,
                                circulant_dropout=self.circulant_dropout,
                                 num_adapters=self.num_adapters,
                                 root_vec=self.root_vec)
        else:
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

class CirculantLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, apply_circ_q, apply_circ_k, apply_circ_v, apply_circ_o,
                  circ_alpha,
                 trainable_alpha, circulant_dropout, num_adapters, root_vec
                ):
        super().__init__(config, layer_idx)

        self.self_attn = CirculantLlamaAttention(config=config, layer_idx=layer_idx, apply_circ_q=apply_circ_q,
                                                 apply_circ_k=apply_circ_k, apply_circ_v=apply_circ_v,
                                                 apply_circ_o=apply_circ_o, circ_alpha=circ_alpha,
                                                 trainable_alpha=trainable_alpha,
                                                 circulant_dropout=circulant_dropout, num_adapters=num_adapters,
                                                 root_vec=root_vec)


class CirculantLlamaModel(LlamaModel):
    def __init__(self, config, apply_circ_q, apply_circ_k, apply_circ_v, apply_circ_o,
                 circ_alpha, trainable_alpha,
                  circulant_dropout,
                 num_adapters, root_vec):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CirculantLlamaDecoderLayer(config=config, layer_idx=layer_idx,
                                        apply_circ_q=apply_circ_q,
                                        apply_circ_k=apply_circ_k,
                                        apply_circ_v=apply_circ_v,
                                        apply_circ_o=apply_circ_o,
                                        circ_alpha=circ_alpha,
                                        trainable_alpha=trainable_alpha,
                                        circulant_dropout=circulant_dropout,
                                        num_adapters=num_adapters,
                                        root_vec=root_vec)
            for layer_idx in range(config.num_hidden_layers)
            ]
        )

class CirculantLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, apply_circ_q, apply_circ_k, apply_circ_v, apply_circ_o,
                  circ_alpha,
                 trainable_alpha, circulant_dropout, num_adapters, root_vec):
        super().__init__(config)

        self.root_vec = nn.Parameter(root_vec, requires_grad=False) # hack for keeping track of devices
        self.model = CirculantLlamaModel(config=config, apply_circ_q=apply_circ_q,
                                        apply_circ_k=apply_circ_k,
                                        apply_circ_v=apply_circ_v,
                                        apply_circ_o=apply_circ_o,
                                        circ_alpha=circ_alpha,
                                        trainable_alpha=trainable_alpha,
                                        circulant_dropout=circulant_dropout,
                                        num_adapters=num_adapters, root_vec=self.root_vec)


