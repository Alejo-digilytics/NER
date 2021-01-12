from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
from src.config import MAX_LEN


def train(data_loader, model, optimizer, device, scheduler):
    """
        -  data_loader: pytorch.DataLoader object
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
        # Move value to device
        for key, value in data.items():
            data[key] = value.to(device)

        # Initialize the gradients
        # Always clear all previously calculated gradients before performing a BP
        # PyTorch doesn't do it automatically because accumulating the gradients is "convenient while training RNNs"
        model.zero_grad()

        # Forward Propagation (loss inside the model)
        # Take care that they use the same names that in data_loader (or **data)
        # "ids" "mask" "tokens_type_ids" "target_pos" "target_tag"
        _, _, loss = model(**data)  # Output tag pos loss
        # Back propagation
        loss.backward()

        # Update the gradients with adams
        optimizer.step()

        # Update learning rate
        # Prior to PyTorch 1.1.0, scheduler of the lr was before the optimizer, now after
        scheduler.step()
        # accumulate the loss for the BP
        final_loss += loss.item()
    return final_loss / len(data_loader)


def loss_function(output, target, mask, num_labels):
    """
    This loss function is a Cross Entropy function since there is no entity overlapping
    Input:
        - output: torch tensor, output of the last layer of the network
        - target: the tensor representing the correct class
        - mask: 
        - num_labels: Number of labels in the sequence. Maximum minus the padding
    Output:
        - loss: the result of the loss function, tensor of floats?
    """
    # Cross entropy for classification
    loss_funct = nn.CrossEntropyLoss()

    # Just for those tokens which are not padding ---> active
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(loss_funct.ignore_index).type_as(target)
    )
    loss = loss_funct(active_logits, active_labels)
    return loss


def validation(data_loader, model, device):
    """
        -  data_loader: pytorch.DataLoader object
        -  model: BERT or another
        -  device: cuda if possible, also gpu or cpu
    """
    model.eval()

    # Fix a top for the loss
    final_loss = 0
    total_tag_acc1 = []
    total_pos_acc1 = []
    for data in tqdm(data_loader, total=len(data_loader)):
        # Load data
        for key, val in data.items():
            data[key] = val.to(device)

        # FP and loss
        _tag, _pos, loss = model(**data)

        # Accuracy
        print(len(data))
        target_pos = data["target_pos"].detach().cpu().numpy().reshape(-1)[:MAX_LEN]
        target_tag = data["target_tag"].detach().cpu().numpy().reshape(-1)[:MAX_LEN]
        pred_pos = _pos.argmax(2).cpu().numpy().reshape(-1)[:MAX_LEN]
        pred_tag = _tag.argmax(2).cpu().numpy().reshape(-1)[:MAX_LEN]
        padding_len = 0
        no_padding = 0
        special = 0
        for token in target_pos:
            if token == 0 and special >= 2:
                padding_len += 1
            elif token == 0 and special < 2:
                special += 1
            else:
                no_padding += 1
        comparison_tag = pred_tag[:no_padding] == target_tag[:no_padding]
        comparison_pos = pred_pos[:no_padding] == target_pos[:no_padding]
        matching_tag = 0
        matching_pos = 0
        for tagg in comparison_tag:
            if tagg:
                matching_tag += 1
            else:
                pass
        for poss in comparison_pos:
            if poss:
                matching_pos += 1
            else:
                pass
        total_tag_acc1.append((matching_tag / no_padding) * 100)
        total_pos_acc1.append((matching_pos / no_padding) * 100)
        final_loss += loss.item()
    tag_acc = np.array(total_tag_acc1).mean()
    pos_acc = np.array(total_pos_acc1).mean()
    return final_loss / len(data_loader), tag_acc, pos_acc
