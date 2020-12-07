from tqdm import tqdm


def train(data_loader, model, optimizer, device, scheduler):
    """
        -  data_loader: EntityDataset object
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


def validation(data_loader, model, device):
    """
        -  data_loader: EntityDataset object
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
