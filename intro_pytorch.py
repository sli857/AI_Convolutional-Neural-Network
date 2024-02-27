##################################################
## Author: David Li
## NetID: sli857
## CS Login: sli857
## Email: sli857@wisc.edu
## External Sources:
## 1: https://www.w3schools.com/python/ref_string_format.asp
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set=datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    return torch.utils.data.DataLoader(train_set, batch_size=64) if training else torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()  # Set the model to training mode
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(T):
        total_loss = 0
        correct_count = 0
        total_count = 0

        for inputs, labels in train_loader:
            opt.zero_grad()  # Zero the gradients
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(output.data, 1)
            correct_count += (predicted == labels).sum().item()
            total_count += inputs.size(0)

        accuracy = correct_count / total_count
        average_loss = total_loss / total_count

        print("Train Epoch: {} Accuracy: {}/{} ({:.2%}) Loss: {:.3f}".format(epoch, correct_count, total_count, accuracy, average_loss))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    total_loss = 0
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for inputs, lables in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, lables)
            total_loss += loss.item() * inputs.size(0)

            _,predicted = torch.max(outputs.data, 1)
            correct_count += (predicted == lables).sum().item()
            total_count += lables.size(0)

        accuracy = correct_count / total_count
        average_loss = total_loss / total_count
    
    if show_loss:
        print("Average loss: {:.4f}\nAccuracy: {:.2%}".format(average_loss, accuracy))
    else:
        print("Accuracy: {:.2%}".format(average_loss))

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    with torch.no_grad():
        log = model(test_images[index])
    prob = F.softmax(log, dim = 1)
    top3P, top3C = torch.topk(prob, 3, dim = 1)
    for i in range(3):
        print("{}: {:.2%}".format(class_names[top3C[0][i]], top3P[0][i]))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()

    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion)
    evaluate_model(model, test_loader, criterion, show_loss = False)
    predict_label(model, next(iter(test_loader))[0], 1)