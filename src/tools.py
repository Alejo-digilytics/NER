import torch
import pandas as pd
from sklearn import preprocessing as prep
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import random


def check_device():
    """ This function checks the cuda's setting. It prints the setting. """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        print("Cuda architectures lis{}".format(torch.cuda.get_arch_list()))
        print("Device capability {}".format(torch.cuda.get_device_capability()))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def preprocess_data_BERT(data_path, my_encoding="utf-8"):
    """
    This function standardizes the pos and tag columns from the dataframe
    Input:
        - data_path (str): path to the DF : Sentence, word, pos, tag
    Output:
        - sentences (ndarray): contains sentences
        - pos (ndarray): contains pos
        - tag (ndarray): contains tag
        - pos_std (ndarray): contains pos_std standardized
        - tag_std (ndarray): contains tag_std standardized
    """

    # The data from the df comes with 4 columns> Sentence, word, pos, tag
    df = pd.read_csv(data_path, encoding=my_encoding)

    # Get columns names and fill possible Nan values
    cols = df.columns.tolist()
    df[cols[0]] = df[cols[0]].fillna(method="ffill")
    # df.to_csv("processed_df_K.csv")

    # Encoding tags and pos for preprocessing, which will be added as columns
    tag_std = prep.LabelEncoder()
    pos_std = prep.LabelEncoder()

    # using fit_transform we standardize the distributions of POS and tag
    df["POS"] = pos_std.fit_transform(df["POS"])
    df["Tag"] = tag_std.fit_transform(df["Tag"])

    # Convert into lists of lists and group by sentence
    sentences = df.groupby(cols[0])["Word"].apply(list).values
    pos = df.groupby(cols[0])["POS"].apply(list).values
    tag = df.groupby(cols[0])["Tag"].apply(list).values

    return sentences, pos, tag, pos_std, tag_std


def special_tokens_dict(vocab_path):
    """ This function creates a dictionary of special tokens based on the vocab.txt from each model """

    special_tokens_dict = {}
    position = 0
    with open(vocab_path, "r") as vocab:
        for line in vocab:
            if "[PAD]" in line:
                special_tokens_dict["[PAD]"] = position
            elif "[UNK]" in line:
                special_tokens_dict["[UNK]"] = position
            elif "[MASK]" in line:
                special_tokens_dict["[MASK]"] = position
            elif "[SEP]" in line:
                special_tokens_dict["[SEP]"] = position
            elif "[CLS]" in line:
                special_tokens_dict["[CLS]"] = position
            elif len(special_tokens_dict) == 5:
                break
            position += 1
    vocab.close()
    special_tokens_path = os.path.join(os.path.dirname(vocab_path), "special_tokens.json")
    with open(special_tokens_path, "w+") as outfile:
        json.dump(special_tokens_dict, outfile)
    outfile.close()
    return special_tokens_dict


def ploter(output_path, name, num_epochs, **losses_accuracies):
    """ Given a dictionary of losses and accuracies plots and saves the graphics for both with random colours.
    Inputs:
        - output_path (str): path to save the plots
        - name (str): name of the plots containing the model's name and hyperparameters
        - num_epochs (int): number of epochs for the x axis
        - losses_accuracies (dict): dictionary with the names of the curves and their values
    """
    epochs = range(num_epochs)
    losses = {k: v for k, v in losses_accuracies.items() if "los" in k.lower()}
    accuracies = {k: v for k, v in losses_accuracies.items() if "acc" in k.lower()}

    # Losses
    for key, value in losses.items():
        value = np.asarray(value)
        new_color = "#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        plt.plot(epochs, value, marker='o', color=new_color, label=key)
    plt.ylabel(" ".join(list(losses.keys())) + " Losses")
    plt.xlabel("Number of epochs")
    plt.legend()
    plt.title(" vs ".join([key for key in losses.keys()]))
    plt.savefig(os.path.join(output_path, name + "_losses_" + '.png'))
    plt.show()

    # Accuracies
    for key, value in accuracies.items():
        value = np.asarray(value)
        new_color = "#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        plt.plot(epochs, value, marker='o', color=new_color, label=key)
    plt.ylabel(" ".join(list(accuracies.keys())) + " Accuracies")
    plt.xlabel("Number of epochs")
    plt.legend()
    plt.title(" vs ".join([key for key in accuracies.keys()]))
    plt.savefig(os.path.join(output_path, name + "_accuracy_" + '.png'))
    plt.show()
