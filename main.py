from NER_model import NER

if __name__ == '__main__':
    model = NER(encoding="latin-1")
    model.training()