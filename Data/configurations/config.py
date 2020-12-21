from os.path import join

# Hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10

# PATHS
BASE_PATH = os.getcwd()
BASE_DATA_PATH = join(BASE_PATH, "Data")
MODELS_PATH = join(BASE_DATA_PATH, "base_models")
DIC_PATH = join(BASE_DATA_PATH, "distionaries")
CHECKPOINTS_PATH = join(BASE_DATA_PATH, "Checkpoints")

# DATA PATHS
TRAINING_FILE = join(BASE_DATA_PATH, "NER_data", "ner_dataset.csv")

# VOCABULARIES
BERT_UNCASED_VOCABULARY_PATH = join(DIC_PATH, "bert_vocab_uncased.txt")
FINBERT_UNCASED_VOCABULARY_PATH = join(DIC_PATH, "finbert_vocab_uncased.txt")
FINBERT_CASED_VOCABULARY_PATH = join(DIC_PATH, "finbert_vocab_cased.txt")

# MODELS
BERT_PATH = join(MODELS_PATH, "bert_base_uncased.bin")
FINBERT_PATH = join(MODELS_PATH, "finbert_base_uncased.bin")
