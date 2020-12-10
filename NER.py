from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup
import src.config_data_loader as config
from src import train_val_loss
from src.NER_dataset import NER_dataset
from src.tools import check_device, preprocess_data_BERT
from src.model import BERT_NER
import joblib
import logging


class NER:
    def __init__(self):
        self.config = config
        self.loss = [[], []]
        self.pos_std = None
        self.tag_std = None
        self.device = None


    def data_preprocess(self, saving=True):
        logging.info("Loading data")
        sentences, pos, tag, self.pos_std, self.tag_std = preprocess_data_BERT(self.config.TRAINING_FILE)

        if saving:
            # Check point for the standardized pos and tag
            data_check_pt = {
                "pos_std": pos_std,
                "tag_std": tag_std
            }
            joblib.dump(data_check_pt, "std_data.bin")
        else:
            pass

        # Save the number of cases per class
        num_tag = len(list(tag_std.classes_))
        num_pos = len(list(pos_std.classes_))

        # Split training set with skl
        self.train_sentences, self.test_sentences, self.train_pos, self.test_pos, self.train_tag, self.test_tag \
            = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.2)

        # Format based on NER_dataset; they are pd.DF
        self.train = NER_dataset(texts=train_sentences, pos=train_pos, tags=train_tag)
        self.test = NER_dataset(texts=test_sentences, pos=test_pos, tags=test_tag)

        # Loaders from torch: it formats the data for pytorch and fixes the batch and the num of kernels
        self.train_data_loader = DataLoader(self.train, batch_size=self.config.TRAIN_BATCH_SIZE,
                                            num_workers=4)  # 4 subprocess
        self.test_data_loader = DataLoader(self.test, batch_size=self.config.VALID_BATCH_SIZE, num_workers=4)

    def model_device(self, phase):
        # Use GPU, load model and move it there -- device or cpu if cuda is not available
        self.device = check_device()
        self.model = BERT_NER(num_tag=num_tag, num_pos=num_pos)
        if phase == "train":
            self.model.to(device)
        else:
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH))
            self.model.to(device)

    def hyperparameters(self):
        # nn.module list of parameters: all parameters from BERT plus the pos and tag layer
        self.param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {"params": [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.001},
            {"params": [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]

        num_train_steps = int(len(train_sentences) / self.config.TRAIN_BATCH_SIZE * self.config.EPOCHS)
        self.optimizer = AdamW(optimizer_parameters, lr=3e-5)

        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                         num_training_steps=num_train_steps)

    def train(self):
        # Loss
        best_loss = np.inf
        for epoch in range(self.config.EPOCHS):
            train_loss = train_val_loss.train(self.train_data_loader, self.model, self.optimizer,
                                              self.device, self.scheduler)
            test_loss = train_val_loss.validation(self.test_data_loader, self.model, self.device)
            print("Train Loss = {} test Loss = {}".format(train_loss, test_loss))
            self.loss[0].extend(train_loss)
            self.loss[1].extend(test_loss)
            if test_loss < best_loss:
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                best_loss = test_loss
        return best_loss

    def predict(self, text):
        # check pos and tag
        if self.pos_std == None:
            std_data = joblib.load("std_data.bin")
            pos_std = std_data["pos_std"]
            tag_std = std_data["tag_std"]
        else:
            pass

        # preprocessing
        text = self.config.TOKENIZER.encode(text)
        sentence = text.split()
        tets_text = NER_dataset(texts=sentence, pos=[[0]* len(sentence)], tags=[[0] * len(sentence)])

        self.model_device()

        with torch.no_grad():
            data = test_dataset[0]
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag, pos, _ = model(**data)

            print(
                enc_tag.inverse_transform(
                    tag.argmax(2).cpu().numpy().reshape(-1)
                )[:len(tokenized_sentence)]
            )
            print(
                enc_pos.inverse_transform(
                    pos.argmax(2).cpu().numpy().reshape(-1)
                )[:len(tokenized_sentence)]
            )