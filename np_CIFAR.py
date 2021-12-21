import ot
import time
import random
import pickle
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special
from model import CNN,CIFARConvNet,MNISTConvNet
from data import Data
from accountant import Accounting
from utils import renyi_divergence
from wdpsgd import SGD


parser = argparse.ArgumentParser(description='Wasserstein Differential Privacy')

parser.add_argument('--dataset', type=str, nargs='?', action = 'store', default='CIFAR10', 
                    help='MNIST | CIFAR10. Default: CIFAR10')
parser.add_argument('--data_dir', type=str, nargs='?', action = 'store', default='dataset', 
                    help='path for datasets')
parser.add_argument('--batch_size_train', type=int, nargs='?', action = 'store', default=1000, 
                    help='batchsize of data')

parser.add_argument('--batch_size_test', type=int, nargs='?', action = 'store', default=1000, 
                    help='batchsize of data')

parser.add_argument('--image_size', type=int, nargs='?', action = 'store', default=32, 
                    help='the height / width of the input image to network')


parser.add_argument('--epochs', type=int, nargs='?', action='store', default=25,
                    help='How many epochs to train. Default: 30.')
parser.add_argument('--learning_rate', type=float, nargs='?', action='store', default=1e-2,
                    help='learning rate for model training.  Default: 1e-3.')
parser.add_argument('--sensitivity', type=float, nargs='?', action='store', default=1,
                    help='sensitivity parameter for Laplace and Gaussian. Default: 1.')
parser.add_argument('--scale', type=float, nargs='?', action='store', default =1,
                    help='scale parameter for Laplace and Gaussian. Default: 1.')
parser.add_argument('--order', type=float, nargs='?', action='store', default = 1,
                    help='order for Wasserstein DP and Bayesian DP. Default=1.')
parser.add_argument('--lambda_', type=float, nargs='?', action='store', default = 1,
                    help='hyperparameter of moment generating function. Default=1.')
parser.add_argument('--delta', type=float, nargs='?', action='store', default=1e-2,
                    help='delta for Wasserstein DP, Bayesian DP and MA. Default: 1e-2.')

parser.add_argument('--grad_clip', type=float, nargs='?', action='store', default=1,
                    help='threshold of gradient clipping. Default: 1.')

args = parser.parse_args()

print(args)



original_data = Data(dataset = args.dataset, batch_size=args.batch_size_train)
train_loader,valid_loader,test_loader = original_data.download_data()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('==========device==========', device)

model = CIFARConvNet() 
#if torch.cuda.device_count() >1:
#    model = nn.DataParallel(model, device_ids=[0,1])

model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = SGD(params=model.parameters(), grad_clip=args.grad_clip, 
        C=args.grad_clip, scale=args.scale, lr=args.learning_rate, batch_size=args.batch_size_train, device=device)


#optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
#privacy_cost=[[] for epoch in range(args.epochs+1)]
#privacy_cost=[]

def train(model, n_epochs, device, criterion, optimizer, train_loader, valid_loader):

    train_loss_list = []
    valid_loss_list = []
    time_consume_history = []


    for epoch in range(1, n_epochs+1):
        train_loss = 0.0 
        valid_loss = 0.0
        train_acc =0.0
        param_ = []
        param_prime_ = []


        model.train()
        time_start = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) 
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            #gradient_norm_history.append(p.grad.norm())

            optimizer.step()
            train_loss += loss.item()*data.size(0)      
            pred = output.argmax(dim=1, keepdim=True) 

            correct = pred.eq(target.view_as(pred)).sum().item()
            train_acc += correct
            
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_acc / len(train_loader.dataset)

            #gradient_norm_history.append(np.linalg.norm(p.grad.numpy()))

        model.eval()
        #privacy_cost.append(ot.wasserstein_1d(x_a=X2,x_b=Y2, p=1).item())


        for batch_idx, (data, target) in enumerate(valid_loader): 
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target) 
            valid_loss += loss.item()*data.size(0)

        #train_loss = train_loss/len(train_loader.sampler) 
        valid_loss = valid_loss/len(valid_loader.sampler)
        
        time_elapsed = time.time() - time_start


        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        time_consume_history.append(time_elapsed)


        print(f'| Epoch: {epoch: 02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Time_comsume: {time_elapsed:.3f} |')
        print(f'| Train Accuracy: {train_accuracy: .3f}% |')
        
    return train_loss_list, valid_loss_list, time_consume_history



train_loss_list, valid_loss_list, time_consume_history = train(
    model = model, n_epochs = args.epochs, device = device, 
    criterion = criterion, optimizer = optimizer, 
    train_loader = train_loader, valid_loader = valid_loader
    )


print('Total Time comsume: ', np.sum(time_consume_history))


def save_as_pickle(save_dir, save_data):
    if type(save_dir) == str and type(save_data)==str:
        save_path = save_dir + '/' + save_data  +'.pickle'
        save_file = open(save_path, 'wb')
        pickle.dump(eval(save_data), save_file)
        save_file.close()
    else:
        print('Please enter the correct variable type')



'''
Renyi_file = open('privacy_dir/Renyi_privacy_list_concat.pickle', 'wb')
Wasser_file = open('privacy_dir/Wasser_privacy_list_concat.pickle', 'wb')
pickle.dump(Renyi_privacy_list_concat, Renyi_file)
Renyi_file.close()
pickle.dump(Wasser_privacy_list_concat, Wasser_file)
Wasser_file.close()
'''

def test(model, device, test_loader):
    test_loss = 0                           
    correct = 0                             
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)            
            test_loss += criterion(output, target).item() 
            pred = output.max(1, keepdim=True)[1]       
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

test(model = model, device = device, test_loader = test_loader)
