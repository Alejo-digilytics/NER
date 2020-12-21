# DS libraries
from sklearn.model_selection import train_test_split
import numpy as np
# NLP and DL libraries
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import transformers
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
# Internal calls
import src.config as config
from src import train_val_loss, dataset
from src.tools import check_device, preprocess_data_BERT
from src.model import BERT_NER
# coding libraries
import joblib
import logging



class NER:
    def __init__(self, encoding, base_model="bert_base_uncased"):
        """ There are only two base_model options allowed: "bert_base_uncased" and "finbert-uncased" """
        self.config = config
        self.loss = [[], []]
        self.pos_std = None
        self.tag_std = None
        self.device = None
        self.encoding = encoding
        self.base_model = base_model

        # Fix the tokenizer
        if base_model == "bert_base_uncased":
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        elif base_model == "finbert-uncased":
            self.tokenizer = BertTokenizer(vocab_file=FINBERT_VOCABULARY_PATH,
                                           do_lower_case=True,
                                           do_basic_tokenize=True)

    def train(self, saving=True):
        logging.info("preprocessing data ...")

        # We preprocess and normalize the data and output it as np.arrays/ pd.series
        sentences, pos, tag, self.pos_std, self.tag_std = preprocess_data_BERT(self.config.TRAINING_FILE,
                                                                               self.encoding)

        logging.info("data has been preprocessed")

        # Checkpoint for the standardized pos and tag
        logging.info("Making checkpoint for the preprocessed data ...")
        if saving:
            data_check_pt = {
                "pos_std": self.pos_std,
                "tag_std": self.tag_std
            }
            joblib.dump(value=data_check_pt, filename=config.CHECKPOINTS_PATH)
        else:
            pass

        # Save the number of cases per class
        num_tag = len(list(self.tag_std.classes_))
        num_pos = len(list(self.pos_std.classes_))

        # Split training set with skl
        logging.info(" Splitting data and creating data sets ...")
        self.train_sentences, self.test_sentences, self.train_pos, self.test_pos, self.train_tag, self.test_tag \
            = train_test_split(sentences, pos, tag, random_state=42, test_size=0.2)

        # Format based on Entities_dataset: getitem outputs pandas dataframes
        self.train = dataset.Entities_dataset(texts=self.train_sentences,
                                              pos=self.train_pos,
                                              tags=self.train_tag,
                                              tokenizer=self.tokenizer
                                              )

        self.test = dataset.Entities_dataset(texts=self.test_sentences,
                                             pos=self.test_pos,
                                             tags=self.test_tag,
                                             tokenizer=self.tokenizer
                                             )

        # Loaders from torch: it formats the data for pytorch and fixes the batch and the num of kernels
        # "workers" means subprocess no gpus in the cuda
        self.train_data_loader = DataLoader(self.train,
                                            batch_size=self.config.TRAIN_BATCH_SIZE,
                                            num_workers=4)
        self.test_data_loader = DataLoader(self.test,
                                           batch_size=self.config.VALID_BATCH_SIZE,
                                           num_workers=4)

        # Load tensor to device and hyperparameters
        logging.info("Moving model to cuda ...")
        self.model_device(phase="train", num_tag=num_tag, num_pos=num_pos)
        self.hyperparameters()

        # initialize the loss
        best_loss = np.inf

        # EPOCHS
        logging.info("Starting Fine-tuning ...")
        for epoch in range(self.config.EPOCHS):
            train_loss = train_val_loss.train(self.train_data_loader, self.model,
                                              self.optimizer, self.device, self.scheduler)
            test_loss = train_val_loss.validation(self.test_data_loader, self.model,
                                                  self.device)
            logging.info("Train Loss = {} test Loss = {}".format(train_loss, test_loss))
            self.loss[0].extend(train_loss)
            self.loss[1].extend(test_loss)
            if test_loss < best_loss:
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                best_loss = test_loss
        logging.info("Fine-tuning finished")
        return best_loss

    def predict(self, text):
        # check pos and tag
        if self.pos_std is None:
            std_data = joblib.load("std_data.bin")
            self.pos_std = std_data["pos_std"]
            self.tag_std = std_data["tag_std"]
        else:
            pass

        # preprocessing
        sentence = text.split()
        text = self.config.TOKENIZER.encode(text)
        tets_text = dataset.Entities_dataset(texts=[sentence], pos=[[0] * len(sentence)], tags=[[0] * len(sentence)])

        self.model_device(phase="predict", num_tag=num_tag, num_pos=num_pos)

        with torch.no_grad():
            data = tets_text[0]
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag, pos, _ = model(**data)

            print(tag_std.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(text)])
            print(pos_std.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(text)])

    def model_device(self, phase, num_tag, num_pos):
        """ Use GPU, load model and move it there -- device or cpu if cuda is not available """

        self.device = check_device()
        self.model = BERT_NER(num_tag=num_tag, num_pos=num_pos)
        if phase == "train":
            self.model.to(self.device)
        elif phase == "predict":
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH))
            self.model.to(self.device)
        else:
            pass

    def hyperparameters(self):
        """ This method fix the parameters and makes a filter over to exclude LayerNorm and biases """

        # nn.module list of parameters: all parameters from BERT plus the pos and tag layer
        self.param_optimizer = list(self.model.named_parameters())

        #  exclude LayerNorm and biases
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {"params": [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.001},
            {"params": [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]

        num_train_steps = int(len(self.train_sentences) / self.config.TRAIN_BATCH_SIZE * self.config.EPOCHS)
        self.optimizer = AdamW(optimizer_parameters, lr=3e-5)

        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=num_train_steps)
