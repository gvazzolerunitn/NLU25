# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch.nn as nn
from transformers import BertModel, BertConfig

class JointBertForNLU(nn.Module):
    def __init__(self, pretrained_name, num_intents, num_slots, dropout=0.1):
        super().__init__()
        self.config = BertConfig.from_pretrained(pretrained_name)
        self.bert   = BertModel.from_pretrained(pretrained_name, config=self.config)
        H = self.config.hidden_size

        self.dropout          = nn.Dropout(dropout)
        self.intent_classifier = nn.Linear(H, num_intents)
        self.slot_classifier   = nn.Linear(H, num_slots)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        seq_h = self.dropout(out.last_hidden_state)    # (B, S, H)
        cls_h = self.dropout(out.pooler_output)        # (B, H)

        intent_logits = self.intent_classifier(cls_h)  # (B, num_intents)
        slot_logits   = self.slot_classifier(seq_h)    # (B, S, num_slots)
        return intent_logits, slot_logits
