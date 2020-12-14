from NER_model import NER
import logging
logger = logging.getLogger()

if __name__ == '__main__':
    model = NER(encoding="latin-1")
    model.train()