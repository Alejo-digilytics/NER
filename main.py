from NER_model import NER
import logging
logger = logging.getLogger()

if __name__ == '__main__':
    model = NER()
    model.data_preprocess()
    model.hyperparameters()
    model.train()