#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import os
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import smdebug.pytorch as smd
from smdebug.pytorch import get_hook
from smdebug import modes
from smdebug.profiler.utils import str2bool


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1,keepdim=True)
            correct = correct+pred.eq(target.view_as(pred)).sum().item()

        print(f"Test set Accuracy: {100*(correct/len(test_loader.dataset))}")
    pass

def train(model, train_loader, val_loader,criterion, optimizer,epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(smd.modes.TRAIN)
    model.train()
    for e in range(epochs):
        training_loss = 0
        correct = 0
        for data,target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred,target)
            training_loss = training_loss+loss
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim=1,keepdim=True)
            correct = correct+pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e} : TrainingLoss {training_loss/len(train_loader.dataset)}, \
              Train Accuracy {100*(correct/len(train_loader.dataset))}%")

        hook.set_mode(smd.modes.EVAL)
        model.eval()
        val_loss = 0
        correct = 0 
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss = val_loss+loss
                pred = outputs.argmax(dim=1,keepdim=True)
                correct = correct+pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e} : ValLoss {val_loss/len(val_loader.dataset)},Validation Accuracy {100*(correct/len(val_loader.dataset))}%")
    return model.to(device)
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    num_classes = 133
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features,num_classes))
    return model


def create_data_loaders(data,batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    traindata = os.path.join(data, "train")
    validdata = os.path.join(data, "valid")
    testdata = os.path.join(data, "test")
    
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
         transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    
    trainset = torchvision.datasets.ImageFolder(traindata, transform=transform)
    validset = torchvision.datasets.ImageFolder(validdata, transform=transform)
    testset = torchvision.datasets.ImageFolder(testdata, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    return trainloader, validloader, testloader




def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    global hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    hook.register_loss(loss_criterion)
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, val_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    model=train(model, train_loader, val_loader,loss_criterion, optimizer,args.epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), path)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    
    args=parser.parse_args()
    
    main(args)
    
    
