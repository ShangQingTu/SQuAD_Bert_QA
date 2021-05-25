import torch
import torch.nn as nn
from transformers import *


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertPasReader(BertPreTrainedModel):
    """Implementation of the Paragraph Reader."""

    def __init__(self, basemodel, config):
        super().__init__(config)
        self.config = config

        self.bert = basemodel
        self.qa_outputs = nn.Linear(self.config.hidden_size, self.config.num_labels)

        # initialize weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.2)
        if self.qa_outputs.bias is not None:
            self.qa_outputs.bias.data.zero_()


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


