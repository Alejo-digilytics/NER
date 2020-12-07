import torch

def check_device():
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


def preprocess_data_BERT(data_path):

    # The data from the df comes with 4 columns> Sentence, word, pos, tag
    # read df
    df = pd.read_csv(data_path, encoding="latin-1")
    # Get columns names
    cols = df.columns.tolist()
    df[cols[0]] = df[cols[0]].fillna(method="ffill")

    # Encoding tags and pos, which will be added as columns
    tag_enc = prep.LabelEncoder()
    pos_enc = prep.LabelEncoder()

    # We need to preserve the class LabelEncoder
    df["POS"] = pos_enc.fit_transform(df["POS"])
    df["Tag"] = tag_enc.fit_transform(df["Tag"])

    # Convert into lists of lists and group by sentence
    sentences = df.groupby(cols[0])["Word"].apply(list).values
    pos = df.groupby(cols[0])["POS"].apply(list).values
    tag = df.groupby(cols[0])["Tag"].apply(list).values
    return sentences, pos, tag, pos_enc, tag_enc