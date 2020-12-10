from tqdm import tqdm
import torch.nn as nn
import torch


def train(data_loader, model, optimizer, device, scheduler):
    """
        -  data_loader: NER_dataset object
        -  model: BERT or another
        -  optimizer: optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        -  device: cuda
        -  scheduler: learning rate scheduler (torch.optim.lr_scheduler.StepLR()
    """
    model.train()
    # Fix a top for the loss
    final_loss = 0
    # loop over the data items and print nice with tqdm
    for data in tqdm(data_loader, total=len(data_loader)):
        for key, value in data.items():
            data[key] = value.to(device)
        # Always clear any previously calculated gradients before performing a BP
        # PyTorch doesn't do this automatically because accumulating the gradients is
        # "convenient while training RNNs"
        model.zero_grad()
        # Take care that they use the same names that in data_loader:
        # "ids" "mask" "tokens_type_ids" "target_pos" "target_tag"
        _, _, loss = model(**data)  # Output tag pos loss
        loss.backward()
        optimizer.step()
        # Prior to PyTorch 1.1.0, scheduler of the lr was before the optimizer, now after
        scheduler.step()
        # accumulate the loss for the BP
        final_loss += loss.item()
    return final_loss / len(data_loader)


def loss_function(output, target, mask, num_labels):
    """
    
    output:
        - output: 
        - target: 
        - mask: 
        - num_labels:
    Input:
        - loss:
    """
    # Cross entropy for classification
    lfn = nn.CrossEntropyLoss()
    # Just for those tokens which are not padding ---> active
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

def validation(data_loader, model, device):
    """
        -  data_loader: NER_dataset object
        -  model: BERT or another
        -  device: cuda
    """
    model.eval()
    # Fix a top for the loss
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for key, val in data.items():
            data[key] = val.to(device)
            # Take care that they use the same names that in data_loader:
            # "ids" "mask" "tokens_type_ids" "target_pos" "target_tag"
            _, _, loss = model(**data)
            final_loss += loss.item()
    return final_loss / len(data_loader)
