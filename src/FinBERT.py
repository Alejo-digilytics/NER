import os
import torch
import torch.nn as nn
import src.config_data_loader as config
import transformers

# paths
output_dir = os.path.join(os.path.split(os.getcwd())[0], "Data", "FinBERT_base")
output_model_file = os.path.join(output_dir, "finbert_vocab_uncased.bin")
output_config_file = os.path.join(output_dir, "config_fb.json")


def loss_function(output, target, mask, num_labels):
    # Cross entropy for classification
    lfn = nn.CrossEntropyLoss()
    # Just for those tokens which are not padding ---> active
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class finbert_model(nn.Module):
    def __init__(self):
        super(finbert_model, self).__init__()
        self.finbert = torch.load(output_model_file)

    def forward(self, ids, mask, tokens_type_ids, target_pos, target_tag):
        pass


class BERT_entities(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(BERT_entities, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH_B)
        # NER parameters
        self.num_tag = num_tag
        self.num_pos = num_pos
        # TODO: Check the best architecture after the base model for fine tuning.
        # Extra layers for fine-tuning
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        # 768 (BERT) composed with a linear function
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, tokens_type_ids, target_pos, target_tag):
        """
        This method if the extra fine tuning NN for both, tags and pos
        """
        # Since this model is for NER we need to take the sequence output
        # We don't want to get a value as output but a sequence of outputs, one per token
        # BERT output: o1
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=tokens_type_ids)
        # Add dropouts
        output_tag = self.bert_drop_1(o1)
        output_pos = self.bert_drop_2(o1)
        # We add the linear outputs
        tag = self.out_tag(output_tag)
        pos = self.out_pos(output_pos)
        # loss for each task! TODO: freeze the other layers
        loss_tag = loss_function(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_function(pos, target_pos, mask, self.num_pos)
        # Compute the accumulative loss
        loss = (loss_tag + loss_pos) / 2
        return tag, pos, loss
