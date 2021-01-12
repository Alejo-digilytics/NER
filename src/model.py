import torch.nn as nn
import src.config as config
from src.train_val_loss import loss_function
from pytorch_pretrained_bert import BertModel, BertConfig


class BERT_NER(nn.Module):
    def __init__(self, num_tag,
                 num_pos,
                 num_ner=0,
                 base_model="bert-base-uncased",
                 tag_dropout=0.3,
                 pos_dropout=0.3,
                 ner_dropout=None,
                 tag_dropout_2=0.3,
                 pos_dropout_2=0.3,
                 ner_dropout_2=None,
                 architecture="simple",
                 ner=False,
                 middle_layer=100):
        super(BERT_NER, self).__init__()
        self.base_model = base_model
        self.architecture = architecture
        self.ner = ner
        if base_model == "bert-base-uncased":
            self.base_model_path = config.BERT_UNCASED_PATH
        elif base_model == "bert_base_cased":
            self.base_model_path = config.BERT_CASED_PATH
        elif base_model == "finbert-vocab-cased":
            self.base_model_path = config.FINBERT_CASED_PATH
        elif base_model == "finbert-vocab-uncased":
            self.base_model_path = config.FINBERT_UNCASED_PATH

        if base_model == "bert-base-uncased":
            # self.model = BertModel.from_pretrained("bert-base-uncased")
            # self.model = transformers.BertModel.from_pretrained(config.BERT_UNCASED_PATH)
            self.model = BertModel.from_pretrained(config.BERT_UNCASED_PATH)
            self.config = BertConfig(vocab_size_or_config_json_file=30522,
                                     hidden_size=768,
                                     num_hidden_layers=12,
                                     num_attention_heads=12,
                                     intermediate_size=3072,
                                     hidden_act='gelu',
                                     hidden_dropout_prob=0.1,
                                     attention_probs_dropout_prob=0.1,
                                     max_position_embeddings=512,
                                     type_vocab_size=2,
                                     initializer_range=0.02

                                     )
        if base_model == "finbert-uncased":
            self.model = BertModel.from_pretrained(config.FINBERT_UNCASED_PATH)
            self.config = BertConfig(vocab_size_or_config_json_file=30873,
                                     hidden_size=768,
                                     num_hidden_layers=12,
                                     num_attention_heads=12,
                                     intermediate_size=3072,
                                     hidden_act='gelu',
                                     hidden_dropout_prob=0.1,
                                     attention_probs_dropout_prob=0.1,
                                     max_position_embeddings=512,
                                     type_vocab_size=2,
                                     initializer_range=0.02
                                     )

        # NER parameters
        self.num_tag = num_tag
        self.num_pos = num_pos
        if self.ner:
            self.num_ner = num_ner

        # Extra layers for fine-tuning FeedForward layer with dropout
        self.bert_drop_1 = nn.Dropout(tag_dropout)
        self.bert_drop_2 = nn.Dropout(pos_dropout)
        if self.ner:
            self.bert_drop_3 = nn.Dropout(ner_dropout)

        # Architecture
        if self.architecture == "simple":
            # 768 (BERT) composed with a linear function
            self.out_tag = nn.Linear(self.config.hidden_size, self.num_tag)
            self.out_pos = nn.Linear(self.config.hidden_size, self.num_pos)
            if ner:
                self.out_pos = nn.Linear(768, self.num_ner)

        if self.architecture == "complex":
            # 768 (BERT) composed with a linear function
            self.tag_mid = nn.Linear(self.config.hidden_size, middle_layer)
            self.bert_drop_1_2 = nn.Dropout(tag_dropout_2)
            self.out_tag = nn.Linear(middle_layer, self.num_tag)
            self.pos_mid = nn.Linear(self.config.hidden_size,  middle_layer)
            self.bert_drop_2_2 = nn.Dropout(pos_dropout_2)
            self.out_pos = nn.Linear(middle_layer, self.num_pos)
            if self.ner:
                self.ner_mid = nn.Linear(self.config.hidden_size,  middle_layer)
                self.bert_drop_3_2 = nn.Dropout(ner_dropout_2)
                self.out_ner = nn.Linear(middle_layer, self.num_ner)

    def forward(self, ids, mask, tokens_type_ids, target_pos, target_tag, target_ner=None):
        """
        This method if the extra fine tuning NN for both, tags and pos
        """
        # Since this model is for NER we need to take the sequence output
        # We don't want to get a value as output but a sequence of outputs, one per token
        # BERT sequence output is the first output. Here o1
        o1, _ = self.model(input_ids=ids, token_type_ids=tokens_type_ids, attention_mask=mask)
        o1t = o1[-1]

        if self.architecture == "simple":
            # Add dropouts
            output_tag = self.bert_drop_1(o1t)
            output_pos = self.bert_drop_2(o1t)
            if self.ner:
                output_ner = self.bert_drop_3(o1t)

        if self.architecture == "complex":
            # Add dropouts
            output_tag1 = self.bert_drop_1(o1t)
            output_pos1 = self.bert_drop_2(o1t)
            if self.ner:
                output_ner1 = self.bert_drop_3(o1t)
            # Add middle layer
            output_tag_2 = self.tag_mid(output_tag1)
            output_pos_2 = self.pos_mid(output_pos1)
            if self.ner:
                output_ner_2 = self.ner_mid(output_ner1)
            output_tag = self.bert_drop_1_2(output_tag_2)
            output_pos = self.bert_drop_2_2(output_pos_2)
            if self.ner:
                output_ner = self.bert_drop_3_2(output_ner_2)

        # We add the linear outputs
        tag = self.out_tag(output_tag)
        pos = self.out_pos(output_pos)
        if self.ner:
            ner = self.out_pos(output_ner)

        # loss for each task
        loss_tag = loss_function(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_function(pos, target_pos, mask, self.num_pos)
        if self.ner:
            loss_ner = loss_function(ner, target_ner, mask, self.num_ner)

        # Compute the accumulative loss
        if not self.ner:
            loss = (loss_tag + loss_pos) / 2
            return tag, pos, loss
        if self.ner:
            loss = (loss_tag + loss_pos + loss_ner) / 3
            return tag, pos, ner, loss
