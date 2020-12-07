import transformers
import os

base_path = os.getcwd()
# Hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH_FB = os.path.join(base_path, "Data", "FinBERT_base", "finbert_vocab_uncased.bin")
BASE_MODEL_PATH_B = 'bert-base-uncased'
MODEL_PATH = os.path.join(base_path, "Data", "FinBERT_base", "bert_ner.bin")
TRAINING_FILE = os.path.join(base_path, "Data", "NER_data", "ner_dataset.csv")
TOKENIZER = transformers.BertTokenizer.from_pretrained(BASE_MODEL_PATH_B, do_lower_case=True)
