import torch
import pandas as pd
from sklearn import preprocessing as prep

def check_device():
    """
    This function checks the cuda's setting
    """
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


def preprocess_data_BERT(data_path, my_encoding="utf8"):
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
    # read df
    df = pd.read_csv(data_path, encoding=my_encoding)
    # Get columns names
    cols = df.columns.tolist()
    df[cols[0]] = df[cols[0]].fillna(method="ffill")

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