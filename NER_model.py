# DS libraries
from sklearn.model_selection import train_test_split
import numpy as np

# NLP and DL libraries
from torch.utils.data import DataLoader
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import transformers
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

# Internal calls
import src.config as config
from src import train_val_loss, dataset
from src.tools import check_device, preprocess_data_BERT, special_tokens_dict, ploter
from src.model import BERT_NER

# coding libraries
from os.path import join
import joblib
import logging
import sys


formatter = logging.Formatter('%(asctime)s %(levelname)s_%(name)s: %(message)s')
logging.basicConfig(filename='fine_tune.log', level=logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
global logger
logger = logging.getLogger("NER")
logger.addHandler(handler)


class NER:
    def __init__(self, encoding, base_model="bert-base-uncased",
                 num_ner=0, tag_dropout=0.3, pos_dropout=0.3, ner_dropout=None,
                 architecture="simple", ner=False, middle_layer=None
                 ):
        """ There are only two base_model options allowed: "bert-base-uncased" and "finbert-uncased" """
        # Fine Tuning parameters
        self.ner = ner
        self.num_ner = num_ner
        self.ner_dropout = ner_dropout
        self.architecture = architecture
        self.middle_layer = middle_layer
        self.tag_dropout = tag_dropout
        self.pos_dropout = pos_dropout

        # configuration
        self.config = config
        self.list_train_losses = []
        self.list_test_losses = []
        self.pos_std = None
        self.tag_std = None
        self.device = None
        self.encoding = encoding
        self.base_model = base_model

        # Fix the tokenizer and special tokens
        if base_model == "bert-base-uncased":
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
            self.special_tokens_dict = special_tokens_dict(config.BERT_UNCASED_VOCAB)
        elif base_model == "finbert-uncased":
            self.tokenizer = BertTokenizer(vocab_file=config.FINBERT_UNCASED_VOCAB,
                                           do_lower_case=True,
                                           do_basic_tokenize=True)
            self.special_tokens_dict = special_tokens_dict(config.FINBERT_UNCASED_VOCAB)

    def training(self, saving=True):
        logger.info("preprocessing data ...")

        # We preprocess and normalize the data and output it as np.arrays/ pd.series
        sentences, pos, tag, self.pos_std, self.tag_std = preprocess_data_BERT(self.config.TRAINING_FILE,
                                                                               self.encoding)

        logger.info("data has been preprocessed")

        # Checkpoint for the standardized pos and tag
        logger.info("Making checkpoint for the preprocessed data ...")
        if saving:
            data_check_pt = {
                "pos_std": self.pos_std,
                "tag_std": self.tag_std
            }
            joblib.dump(value=data_check_pt, filename=config.CHECKPOINTS_META_PATH)
        else:
            pass

        # Save the number of cases per class
        num_tag = len(list(self.tag_std.classes_))
        num_pos = len(list(self.pos_std.classes_))

        # Split training set with skl
        logger.info(" Splitting data and creating data sets ...")
        self.train_sentences, self.test_sentences, self.train_pos, self.test_pos, self.train_tag, self.test_tag \
            = train_test_split(sentences, pos, tag, random_state=42, test_size=0.2)

        # Format based on Entities_dataset: getitem outputs pandas dataframes
        self.train = dataset.Entities_dataset(texts=self.train_sentences,
                                              pos=self.train_pos,
                                              tags=self.train_tag,
                                              tokenizer=self.tokenizer,
                                              special_tokens=self.special_tokens_dict
                                              )

        self.test = dataset.Entities_dataset(texts=self.test_sentences,
                                             pos=self.test_pos,
                                             tags=self.test_tag,
                                             tokenizer=self.tokenizer,
                                             special_tokens=self.special_tokens_dict
                                             )

        # Loaders from torch: it formats the data for pytorch and fixes the batch and the num of kernels
        # "workers" means subprocess no gpus in the cuda
        self.train_data_loader = DataLoader(self.train,
                                            batch_size=self.config.TRAIN_BATCH_SIZE,
                                            num_workers=4
                                            )
        self.test_data_loader = DataLoader(self.test,
                                           batch_size=self.config.VALID_BATCH_SIZE,
                                           num_workers=4
                                           )

        # Load tensor to device and hyperparameters
        logger.info("Moving model to cuda ...")
        self.model_device(phase="train", num_tag=num_tag, num_pos=num_pos)
        self.hyperparameters()

        # initialize the loss
        best_loss = np.inf

        # EPOCHS
        logger.info("Starting Fine-tuning ...")
        for epoch in range(self.config.EPOCHS):
            logger.info("Start epoch {}".format(epoch+1))
            train_loss = train_val_loss.train(self.train_data_loader,
                                              self.model,
                                              self.optimizer,
                                              self.device,
                                              self.scheduler)
            test_loss = train_val_loss.validation(self.test_data_loader,
                                                  self.model,
                                                  self.device)
            logger.info("Train Loss = {}".format(train_loss))
            logger.info("Test Loss = {}".format(test_loss))
            self.list_train_losses.append(float(train_loss))
            self.list_test_losses.append(float(test_loss))
            logger.info("End epoch {}".format(epoch+1))
            logger.info("Testing epoch {}".format(epoch+1))
            if test_loss < best_loss:
                torch.save(self.model.state_dict(), self.config.CHECKPOINTS_MODEL_PATH)
                best_loss = test_loss
            logger.info("End epoch {} with loss {} asnd best loss {}".format(epoch, test_loss, best_loss))
        logger.info("Fine-tuning finished")
        logger.info("With training losses: {}".format(self.list_train_losses))
        logger.info("With test losses: {}".format(self.list_test_losses))

        # plotting
        losses = {"train": self.list_train_losses, "test": self.list_test_losses}
        ploter(self.config.EPOCHS, **losses)

        # Saving results
        data1 = np.array(self.list_train_losses)
        np.savez(join(config.BASE_DATA_PATH, "list_train_losses"), data1)
        data2 = np.array(self.list_test_losses)
        np.savez(join(config.BASE_DATA_PATH, "list_test_losses"), data2)
        data4 = np.array(num_pos)
        np.savez(join(config.BASE_DATA_PATH, "num_pos"), data4)
        data3 = np.array(num_tag)
        np.savez(join(config.BASE_DATA_PATH, "num_tag"), data3)
        return best_loss

    def predict(self, text):

        # Loading the results
        num_tag = np.load("num_tag.npz")
        num_pos = np.load("num_pos.npz")

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
        tets_text = dataset.Entities_dataset(texts=[sentence],
                                             pos=[[0] * len(sentence)],
                                             tags=[[0] * len(sentence)],
                                             tokenizer=self.tokenizer,
                                             special_tokens=self.special_tokens_dict
                                             )

        self.model_device(phase="predict", num_tag=num_tag, num_pos=num_pos)

        with torch.no_grad():
            data = tets_text[0]
            for k, v in data.items():
                data[k] = v.to(self.device).unsqueeze(0)
            tag, pos, _ = self.model(**data)

            print(self.tag_std.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(text)])
            print(self.pos_std.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(text)])

    def model_device(self, phase, num_tag, num_pos):
        """ Use GPU, load model and move it there -- device or cpu if cuda is not available """

        self.device = check_device()
        self.model = BERT_NER(num_tag=num_tag,
                              num_pos=num_pos,
                              num_ner=self.num_ner,
                              base_model=self.base_model,
                              tag_dropout=self.tag_dropout,
                              pos_dropout=self.pos_dropout,
                              ner_dropout=self.ner_dropout,
                              architecture=self.architecture,
                              ner=self.ner,
                              middle_layer=self.middle_layer)
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
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=num_train_steps)
