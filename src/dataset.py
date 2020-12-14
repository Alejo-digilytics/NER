import src.config_data_loader as config
import torch


class Entities_dataset:
    """
    This class produces the input for the model as masked language model with its special tokens
    , ids, masks and padding, creating a getitem method that produces the batches.
    Data must be preprocessed before using this class as a list of words to be tokenized
    Input:
        - text (list): list of words
        - pos (list): list of pos associated
        - tag (list): list of tags associated
    Output:
        - ids:
        - masks:
        - tokens:
        - pos:
        - tags:
    """
    def __init__(self, texts, pos, tags):
        # text = [["hi","I", "am", ...], ["And", ...]...]
        # pos/tags = [[1,2,3,4, ...], [...]...]
        self.texts = texts
        self.pos = pos
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        target_pos = []
        target_tag = []

        for i, s in enumerate(text):  # i = position, s = words
            # token id from Bert tokenizer
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            # words had become tokens and the size increase.
            # So, the pos and tags too
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tags[i]] * input_len)

            # Adding spacy for the special tokens
            ids = ids[:config.MAX_LEN - 2]
            target_pos = target_pos[:config.MAX_LEN - 2]
            target_tag = target_tag[:config.MAX_LEN - 2]

            # CSL = 101 in BERT!!! and SEP is 102 in BERT!!
            ids = [101] + ids + [102]
            target_pos = [0] + target_pos + [0]
            target_tag = [0] + target_tag + [0]

            # Prepare masks: 1 means token
            mask = [1] * len(ids)
            tokens_type_ids = [0] * len(ids)

            # PADDING FIXED, NOT DYNAMIC
            padding_len = config.MAX_LEN - len(ids)  # TODO: switch to dynamic padding
            ids = ids + ([0] * padding_len)
            tokens_type_ids = tokens_type_ids + ([0] * padding_len)
            mask = mask + ([0] * padding_len)
            target_pos = target_pos + ([0] * padding_len)
            target_tag = target_tag + ([0] * padding_len)

            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "tokens_type_ids": torch.tensor(tokens_type_ids, dtype=torch.long),
                "target_pos": torch.tensor(target_pos, dtype=torch.long),
                "target_tag": torch.tensor(target_tag, dtype=torch.long),
            }
