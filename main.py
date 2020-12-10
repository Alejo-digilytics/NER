import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup
import src.config_data_loader as config
from src import NER_dataset, train_val_loss
from src.tools import check_device, preprocess_data_BERT
from src.model import BERT_NER

if __name__ == '__main__':
    sentences, pos, tag, pos_std, tag_std = preprocess_data_BERT(config.TRAINING_FILE)
    # Check point for the standardized pos and tag
    check_pt = {
        "pos_std": pos_std,
        "tag_std": tag_std
    }
    # Save the number of cases per class
    num_tag = len(list(tag_std.classes_))
    num_pos = len(list(pos_std.classes_))

    # Split training set with skl
    train_sentences, test_sentences, train_pos, test_pos, train_tag, test_tag \
        = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.2)

    # Format based on NER_dataset; they are pd.DF
    train = NER_dataset.NER_dataset(texts=train_sentences, pos=train_pos, tags=train_tag)
    test = NER_dataset.NER_dataset(texts=test_sentences, pos=test_pos, tags=test_tag)

    # Loaders from torch: it formats the data for pytorch and fixes the batch and the num of kernels
    train_data_loader = DataLoader(train, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)  # 4 subprocess
    test_data_loader = DataLoader(test, batch_size=config.VALID_BATCH_SIZE, num_workers=4)

    # Use GPU, load model and move it there -- device or cpu if cuda is not available
    device = check_device()
    model = BERT_NER(num_tag=num_tag, num_pos=num_pos)
    model.to(device)

    # Optimizer
    # nn.module list of parameters: all parameters from BERT plus the pos and tag layer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.001},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    # Scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    # Loss
    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = train_val_loss.train(train_data_loader, model, optimizer, device, scheduler)
        test_loss = train_val_loss.validation(test_data_loader, model, device)
        print("Train Loss = {} test Loss = {}".format(train_loss, test_loss))
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
