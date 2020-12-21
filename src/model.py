import torch.nn as nn
import src.config as config
import transformers
from src.train_val_loss import loss_function
from pytorch_pretrained_bert import BertModel, BertConfig


class BERT_NER(nn.Module):
    def __init__(self, num_tag, num_pos, base_model="bert_base_uncased"):
        super(BERT_NER, self).__init__()

        if base_model == "bert_base_uncased":
            # self.model = BertModel.from_pretrained("bert-base-uncased")
            self.model = transformers.BertModel.from_pretrained(config.BERT_PATH)
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
            self.model = BertModel.from_pretrained(config.FINBERT_PATH)
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

        # Extra layers for fine-tuning FeedFordward layer with 30% of dropout in both
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
        o1, _ = self.model(ids, attention_mask=mask, token_type_ids=tokens_type_ids)

        # Add dropouts
        output_tag = self.bert_drop_1(o1)
        output_pos = self.bert_drop_2(o1)

        # We add the linear outputs
        tag = self.out_tag(output_tag)
        pos = self.out_pos(output_pos)

        # loss for each task
        loss_tag = loss_function(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_function(pos, target_pos, mask, self.num_pos)

        # Compute the accumulative loss
        loss = (loss_tag + loss_pos) / 2
        return tag, pos, loss
