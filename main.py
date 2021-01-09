from NER_model import NER

if __name__ == '__main__':
    model = NER(encoding="UTF-8", base_model="bert-base-uncased")
    model.training()