import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train(model: nn.Module, dataloader: DataLoader, 
        loss_fn, opt: optim, device) -> None:
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 10 == 0:
            current = (batch+1)*len(X)
            print(f"Params: {model.state_dict()}, Loss: {loss.item():>2f}, [{current}/{size}]")

def test(dataloader: DataLoader, model: nn.Module, loss_fn, device) -> None:
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred,y)
            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_test_loss = total_loss / num_batches
    accuracy = correct / size

    print(f"Test Error: \n Accuracy: {100*accuracy:>0.1f}%, Avg loss: {avg_test_loss:>8f}\n")
