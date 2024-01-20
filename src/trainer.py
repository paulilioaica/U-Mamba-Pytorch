import torch 
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm 
from torch.utils.data import DataLoader

# set up train and eval from test.ipynb

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def eval(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def train_on_dataset(model, train_dataloader, test_dataloader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        test_loss = eval(model, test_dataloader, criterion, device)
        print(f"epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}")
    return model
