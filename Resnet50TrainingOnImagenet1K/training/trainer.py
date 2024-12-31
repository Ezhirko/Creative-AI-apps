import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import os

def train(dataloader, model, device, loss_fn, optimizer,scheduler,epoch):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    start = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_size = len(X)
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], {(current/size * 100):>4f}%")
            step = epoch * size + current
            new_start = time.time()
            delta = new_start - start
            start = new_start
            if batch != 0:
                print("Done in ", delta, " seconds")
                remaining_steps = size - current
                speed = 100 * batch_size / delta
                remaining_time = remaining_steps / speed
                print("Remaining time (seconds): ", remaining_time)
        optimizer.zero_grad()
    print("Entire epoch done in ", time.time() - start0, " seconds")

def test(dataloader, model, device, loss_fn, epoch, train_dataloader, calc_acc5=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if calc_acc5:
                _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()
    test_loss /= num_batches
    step = epoch * len(train_dataloader.dataset)
    
    correct /= size
    correct_top5 /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if calc_acc5:
        print(f"Test Error: \n Accuracy-5: {(100*correct_top5):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def start_training(model,device,loss_fn,train_loader,test_loader,optimizer,scheduler,params,EPOCHS):
   best_test_accuracy = 0
   #checkpoint_path = os.path.join("/home/ubuntu/Session9/checkpoint", "ImageNet1k", f"checkpoint.pth")
   for epoch in range(EPOCHS):
        print(f" ***** EPOCH:{epoch} ***** ")
        train(train_loader,model,device,loss_fn,optimizer,scheduler,epoch)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "params": params
        }
        torch.save(checkpoint, os.path.join("/home/ubuntu/Session9/checkpoint", "ImageNet1k", f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("/home/ubuntu/Session9/checkpoint", "ImageNet1k", f"checkpoint.pth"))

        test(test_loader,model,device,loss_fn,epoch,train_loader,True)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate after epoch {epoch}: {current_lr:.6f}")
   print(f"Best Test Accuracy:{best_test_accuracy}")
    
        