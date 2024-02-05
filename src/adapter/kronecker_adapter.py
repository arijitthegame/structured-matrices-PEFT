from typing import Optional, List, Dict
import torch
import torch.nn as nn

from transformers import BertLayer, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertAttention, BertEncoder
from transformers.modeling_outputs import SequenceClassifierOutput

from utils.utils import ACT2CLS, ClassInstantier, batch_kron

ACT2FN = ClassInstantier(ACT2CLS)


class KroneckerLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        r: list,
        kronecker_dropout: float,
        tie_weights: bool,
        activation: str,
        **kwargs
    ):
        """
        Simple Kronecker product layer, as a first step we are only implementing pure tensors i.e. a \otimes b
        """
        # TODO : Add support for general tensors
        # note In this general case, we allow r to be a list[r_1,r_2] satisfying the required conditions :
        # A is a matrix of size r_1 \times r_2 then B is such that B \otimes A is of the right size
        super().__init__()
        self.r = r
        self.tie_weights = tie_weights
        self.activation = activation
        self.activation_fn = ACT2FN[self.activation]
        self.kronecker_bias = nn.Parameter(torch.zeros(out_features))
        # Optional dropout
        if kronecker_dropout > 0.0:
            self.kronecker_dropout = nn.Dropout(p=kronecker_dropout)
        else:
            self.kronecker_dropout = lambda x: x

        if r[0] > 0 or r[-1] > 0:
            assert in_features % r[0] == 0 and out_features % r[-1] == 0
            self.kronecker_A = nn.Parameter(torch.randn((r[0], r[-1])))
            if self.tie_weights:
                self.kronecker_B = self.kronecker_A.t()
            else:
                self.kronecker_B = nn.Parameter(
                    torch.randn((int(in_features / r[0]), out_features // r[-1]))
                )

        self.init_parameters()

    def init_parameters(self):

        # initialize B the same way as the default for nn.Linear and A to zero
        # this is different than what is described in the paper but does not affect performance
        with torch.no_grad():
            if self.tie_weights is False:
                self.kronecker_A.data.normal_(mean=0.0, std=0.02)
                nn.init.zeros_(self.kronecker_B)
            else:
                nn.init.zeros_(self.kronecker_A)
                nn.init.zeros_(
                    self.kronecker_B
                )  # this will likely not work but I do not of a good initialization

    def forward(self, x: torch.Tensor):
        x = (
            self.activation_fn(
                self.kronecker_dropout(x)
                @ (torch.kron(self.kronecker_B, self.kronecker_A))
            )
            + self.kronecker_bias
        )
        return x


class CustomAdapter(nn.Module):
    def __init__(
        self,
        input_size,
        r,
        activation,
        tie_weights,
        ln_before: bool = False,
        ln_after: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = ln_before
        self.add_layer_norm_after = ln_after
        self.r = r
        self.dropout = dropout
        self.activation = activation
        self.tie_weights = tie_weights

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        seq_list.append(
            KroneckerLayer(
                self.input_size,
                self.input_size,
                self.r,
                kronecker_dropout=self.dropout,
                activation=self.activation,
                tie_weights=self.tie_weights,
                **kwargs,
            )
        )
        self.adapter_down = nn.Sequential(*seq_list)

        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:  # False
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

    def forward(self, x):
        rep = self.adapter_down(x)
        output = x + rep
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)
        return output


class CustomAdapterBertSelfOutput(nn.Module):
    def __init__(self, config, r, kron_dropout, activation, kron_tie_weights, **kwargs):
        super().__init__()
        self.config = config
        self.r = r
        self.kron_dropout = kron_dropout
        self.activation = activation
        self.kron_tie_weights = kron_tie_weights

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mh_adapter = CustomAdapter(
            input_size=config.hidden_size,
            r=self.r,
            dropout=self.kron_dropout,
            activation=self.activation,
            tie_weights=self.kron_tie_weights,
            **kwargs,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.mh_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CustomAdapterBertOutput(nn.Module):
    def __init__(self, config, r, kron_dropout, activation, kron_tie_weights, **kwargs):
        super().__init__()
        self.config = config
        self.r = r
        self.kron_dropout = kron_dropout
        self.activation = activation
        self.kron_tie_weights = kron_tie_weights

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_adapter = CustomAdapter(
            input_size=config.hidden_size,
            r=self.r,
            kron_dropout=self.kron_dropout,
            activation=self.activation,
            tie_weights=self.kron_tie_weights,
            **kwargs,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CustomAdapterBertAttention(BertAttention):
    def __init__(
        self,
        config,
        r,
        kron_dropout,
        activation,
        kron_tie_weights,
        position_embedding_type=None,
        **kwargs
    ):
        super().__init__(config)

        self.config = config
        self.r = r
        self.kron_dropout = kron_dropout
        self.activation = activation
        self.kron_tie_weights = kron_tie_weights
        self.output = CustomAdapterBertSelfOutput(
            config=self.config,
            r=self.r,
            kron_dropout=self.kron_dropout,
            activation=self.activation,
            kron_tie_weights=self.kron_tie_weights,
            **kwargs,
        )


class CustomAdapterBertLayer(BertLayer):
    def __init__(self, config, r, kron_dropout, activation, kron_tie_weights, **kwargs):
        super().__init__(config)
        self.config = config
        self.r = r
        self.kron_dropout = kron_dropout
        self.activation = activation
        self.kron_tie_weights = kron_tie_weights

        self.attention = CustomAdapterBertAttention(
            config=self.config,
            r=self.r,
            kron_dropout=self.kron_dropout,
            activation=self.activation,
            kron_tie_weights=self.kron_tie_weights,
            **kwargs,
        )
        self.output = CustomAdapterBertOutput(
            config=self.config,
            r=self.r,
            kron_dropout=self.kron_dropout,
            activation=self.activation,
            kron_tie_weights=self.kron_tie_weights,
            **kwargs,
        )


class CustomAdapterBertEncoder(BertEncoder):
    # note this custom BERT do not support gradient checkpointing
    def __init__(self, config, r, kron_dropout, activation, kron_tie_weights, **kwargs):
        super().__init__(config)
        self.config = config
        self.r = r
        self.kron_dropout = kron_dropout
        self.activation = activation
        self.kron_tie_weights = kron_tie_weights

        self.layer = nn.ModuleList(
            [
                CustomAdapterBertLayer(
                    config=self.config,
                    r=self.r,
                    kron_dropout=self.kron_dropout,
                    activation=self.activation,
                    kron_tie_weights=self.kron_tie_weights,
                    **kwargs,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )


class CustomAdapterBertModel(BertModel):
    def __init__(self, config, r, kron_dropout, activation, kron_tie_weights, **kwargs):
        super().__init__(config)

        self.config = config
        self.r = r
        self.kron_dropout = kron_dropout
        self.activation = activation
        self.kron_tie_weights = kron_tie_weights

        self.encoder = CustomAdapterBertEncoder(
            config=self.config,
            r=self.r,
            kron_dropout=self.kron_dropout,
            activation=self.activation,
            kron_tie_weights=self.kron_tie_weights,
            **kwargs,
        )


class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, config, r, kron_dropout, activation, kron_tie_weights, **kwargs):
        super().__init__()

        self.config = config
        self.r = r
        self.kron_dropout = kron_dropout
        self.activation = activation
        self.kron_tie_weights = kron_tie_weights

        self.bert = CustomAdapterBertModel.from_pretrained(
            "bert-base-uncased",
            config=self.config,
            r=self.r,
            kron_dropout=self.kron_dropout,
            activation=self.activation,
            kron_tie_weights=self.kron_tie_weights,
            **kwargs,
        )
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
