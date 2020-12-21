from os.path import join
import os

# Hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10

# PATHS
BASE_PATH = os.getcwd()
BASE_DATA_PATH = join(BASE_PATH, "Data")
BASE_MODELS_PATH = join(BASE_PATH, "models")

# DATA PATHS
TRAINING_FILE = join(BASE_DATA_PATH, "NER_data", "ner_dataset.csv")
CHECKPOINTS_PATH = join(BASE_DATA_PATH, "Checkpoints", "std_data.bin")

# MODELS
BERT_PATH = join(BASE_MODELS_PATH, "bert-base-uncased")
FINBERT_UNCASED_PATH = join(BASE_MODELS_PATH, "finbert-uncased")
FINBERT_CASED_PATH = join(BASE_MODELS_PATH, "finbert-cased")
