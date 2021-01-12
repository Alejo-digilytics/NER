from os.path import join
import os

# Hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10

# PATHS
BASE_PATH = os.getcwd()
BASE_DATA_PATH = join(BASE_PATH, "Data")
MODELS_PATHS = join(BASE_PATH, "models")

# MODELS
FINBERT_UNCASED_PATH = join(MODELS_PATHS, "finbert-uncased")
FINBERT_CASED_PATH = join(MODELS_PATHS, "finbert-cased")
BERT_CASED_PATH = join(MODELS_PATHS, "bert-base-cased")
BERT_UNCASED_PATH = join(MODELS_PATHS, "bert-base-uncased")

# Vocabularies
FINBERT_UNCASED_VOCAB = join(FINBERT_UNCASED_PATH, "vocab.txt")
FINBERT_CASED_VOCAB = join(FINBERT_CASED_PATH, "vocab.txt")
BERT_CASED_VOCAB = join(BERT_CASED_PATH, "vocab.txt")
BERT_UNCASED_VOCAB = join(BERT_UNCASED_PATH, "vocab.txt")

# DATA PATHS
TRAINING_FILE = join(BASE_DATA_PATH, "NER_data", "ner_dataset.csv")
ACC_FILE = join(BASE_DATA_PATH, "NER_data")
CHECKPOINTS_META_PATH = join(BASE_DATA_PATH, "Checkpoints", "std_data.bin")
CHECKPOINTS_MODEL_PATH = join(BASE_DATA_PATH, "Checkpoints", "model.bin")

