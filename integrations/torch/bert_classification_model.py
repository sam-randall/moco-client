

from typing import Callable, Literal, Optional
import numpy as np
from transformers import BertForSequenceClassification
import torch
from src.early_exit_model import EarlyExitModel
from src.get_fast_models import get_fast_rules

def get_extended_attention_mask(attention_mask, device):
    # (batch_size, 1, 1, seq_len)
    extended = attention_mask[:, None, None, :].to(dtype=torch.float, device=device)
    extended = (1.0 - extended) * -10000.0
    return extended


class EarlyExitTextClassificationModel(BertForSequenceClassification, EarlyExitModel):
    def __init__(self, model: BertForSequenceClassification):
        super().__init__(model.config)
        self.embeddings = model.bert.embeddings
        self.encoder = model.bert.encoder
        self.pooler = model.bert.pooler
        self.dropout = model.dropout
        self.classifier = model.classifier
        self.n_classes = self.classifier.out_features

        self.embedding_rule = None
        self.encoder_rule = None
        self.encoder_rule_metadata = None
        self.prediction_value = None

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        x = self.embeddings(input_ids = input_ids)
        outputs = np.zeros((len(input_ids), self.n_classes))
        mask = None
        if self.embedding_rule is not None:
            mask = self.embedding_rule(x)
            outputs[mask, self.prediction_value] = 1
            x = x[~mask]

        for i, l in enumerate(self.encoder.layer):

            # are these the right inputs?
            extended_attention_mask = get_extended_attention_mask(attention_mask, input_ids.device)
            attn_output = l.attention(
                hidden_states=x,
                attention_mask=extended_attention_mask,
                head_mask=None
            )

            x = attn_output[0]

            if self.encoder_rule is not None and self.encoder_rule_metadata['layer_index'] == i and self.encoder_rule_metadata['module'] == 'attention':
                N = x.shape[0]
                mask = self.encoder_rule(x.detach().numpy().reshape((N, -1)))
                outputs[mask, self.prediction_value] = 1
                x = x[~mask]
                attention_mask = attention_mask[~mask]

            intermediate_output = l.intermediate(x)

            if self.encoder_rule is not None and self.encoder_rule_metadata['layer_index'] == i and self.encoder_rule_metadata['module'] == 'intermediate':
                mask = self.encoder_rule(x.detach().numpy().reshape((N, -1)))
                outputs[mask, self.prediction_value] = 1
                x = x[~mask]
            x = l.output(intermediate_output, x)

            if self.encoder_rule is not None and self.encoder_rule_metadata['layer_index'] == i and self.encoder_rule_metadata['module'] == 'output':
                mask = self.encoder_rule(x.detach().numpy().reshape((N, -1)))
                outputs[mask, self.prediction_value] = 1
                x = x[~mask]
        x = self.pooler(x)
        x = self.classifier(x)
        if mask is not None:
            outputs[~mask] = x.detach().numpy()
        return outputs
    
    def add_rule(self,
                 classifier: Callable,
                 prediction_given_classifier: int,
                 layer: Literal['inputs', 'embeddings', 'encoder', 'pooler'],
                 encoder_layer: Optional[int] = None,
                 encoder_layer_module: Optional[Literal['attention', 'intermediate', 'output']] = None):
        if layer == 'embeddings':
            self.embedding_rule = classifier
            self.prediction_value = prediction_given_classifier
        elif layer == 'encoder':
            self.encoder_rule = classifier
            self.prediction_value = prediction_given_classifier
            self.encoder_rule_metadata = {
                'layer_index': encoder_layer,
                'module': encoder_layer_module
            }
        else:
            raise NotImplementedError()
        
    def get_and_apply_rule(self, dataset, predictions=None):
        if predictions is None:
            
            raise NotImplementedError("Need to implement model evaluation in this case. ")
        rule_response = get_fast_rules(dataset, predictions)
        
        if rule_response is not None:
            print(rule_response)

