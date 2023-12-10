
import time
import random
import argparse
import itertools

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable, Function
import numpy as np
from model import CNN,CIFARConvNet,MNISTConvNet

from bayesian_privacy_accountant import BayesianPrivacyAccountant
from wasserstein_privacy_accountant import WassersteinPrivacyAccountant


parser = argparse.ArgumentParser(description='Wasserstein Differential Privacy')

parser.add_argument('--dataset', type=str, nargs='?', action = 'store', default='MNIST', 
                    help='MNIST | CIFAR10. Default: CIFAR10')

parser.add_argument('--batch_size_train', type=int, nargs='?', action = 'store', default=1024, 
                    help='batchsize of data')

parser.add_argument('--batch_size_test', type=int, nargs='?', action = 'store', default=1024, 
                    help='batchsize of data')

parser.add_argument('--n_epochs', type=int, nargs='?', action='store', default=25,
                    help='How many epochs to train. Default: 30.')
parser.add_argument('--learning_rate', type=float, nargs='?', action='store', default=0.01,
                    help='learning rate for model training.  Default: 1e-3.')

parser.add_argument('--scale', type=float, nargs='?', action='store', default =0.01,
                    help='scale parameter for Laplace and Gaussian. Default: 1.')
parser.add_argument('--order', type=float, nargs='?', action='store', default = 16,
                    help='order for Wasserstein DP and Bayesian DP. Default=1.')

parser.add_argument('--delta', type=float, nargs='?', action='store', default=1e-2,
                    help='delta for Wasserstein DP, Bayesian DP and MA. Default: 1e-2.')

parser.add_argument('--grad_clip', type=float, nargs='?', action='store', default=1.0,
                    help='threshold of gradient clipping. Default: 1.')

parser.add_argument('--accountant', type=str, nargs='?', action='store', default='Wasserstein')

parser.add_argument('--beta', type=float, nargs='?', action='store', default=32)

args = parser.parse_args()




train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('MNIST', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=args.batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('MNIST', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=args.batch_size_test, shuffle=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('==========device==========', device)


model = MNISTConvNet() 
#if torch.cuda.device_count() >1:
#    model = nn.DataParallel(model, device_ids=[0,1])

model = model.to(device)



criterion = nn.CrossEntropyLoss(reduction='none')



optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
#privacy_cost=[[] for epoch in range(args.epochs+1)]
#privacy_cost=[]



def sparsify_update(params, p, use_grad_field=True):
    init = True
    for param in params:
        if param is not None:
            if init:
                idx = torch.zeros_like(param, dtype=torch.bool)
                idx.bernoulli_(1 - p)
            if use_grad_field:
                if param.grad is not None:
                    idx = torch.zeros_like(param, dtype=torch.bool)
                    idx.bernoulli_(1 - p)
                    param.grad.data[idx] = 0
            else:
                init = False
                param.data[idx] = 0
    return idx





def test(test_loader, model, device):                      
    correct = 0                             
    with torch.no_grad():
        for feature, target in test_loader:
            feature, target = feature.to(device), target.to(device)
            output = model(feature)            
            #test_loss += criterion(output, target).item() 
            pred = output.max(1, keepdim=True)[1]       
            correct += pred.eq(target.view_as(pred)).sum().item()

    datasize = len(test_loader.dataset)

    #test_loss /= len(test_loader.dataset)
    print('\nTest Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, 
            datasize,
            100. * correct / datasize
            )
    )
    return 100. * correct / datasize

#test(model = model, device = device, test_loader = test_loader)




def train(train_loader, model, optimizer, criterion, n_epochs, device, accountant):


    if type(accountant)==list:
        print('Multiple Accounting Methods')
    else:
        accountant = list([accountant])
        print('Single Accounting Method')


    # for account in accountant:
    #     print(account.__class__.__name__)



    accuracies = []
    n_batches = len(train_loader.dataset) / args.batch_size_train + 1
    sampling_prob = 0.1
    max_grad_norm = args.grad_clip
    

    for epoch in range(1, n_epochs+1):
        running_loss = 0.0 
        train_acc =0.0


        model.train()
        time_start = time.time()

        for batch_idx, (feature, target) in enumerate(train_loader):
            #print('='*20, batch_idx)
            feature, target = feature.to(device), target.to(device) 

            #feature_variable = Variable(feature)
            #target_variable = Variable(target.long()).flatten()
            batch_size = float(len(feature))

            optimizer.zero_grad()

            
            output = model(feature)
            loss = criterion(output, target)

            # accountant--gradient subsampling
            if accountant:
                grads_est = []
                n_subbatch = 8
                for j in range(n_subbatch):
                    grad_sample = torch.autograd.grad(
                        loss[np.delete(range(int(batch_size)), j)].mean(),
                        [p for p in model.parameters() if p.requires_grad], 
                        retain_graph=True
                    )

                    with torch.no_grad():
                        grad_sample = torch.cat([g.view(-1) for g in grad_sample])
                        grad_sample /= max(1.0, grad_sample.norm().item() / max_grad_norm)
                        grads_est += [grad_sample]

                with torch.no_grad():
                    grads_est = torch.stack(grads_est)
                    sparsify_update(grads_est, p=sampling_prob, use_grad_field=False)

            (loss.mean()).backward()
            running_loss += loss.mean().item()


            if accountant:
                with torch.no_grad():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                p.grad += torch.randn_like(p.grad) * (args.scale * max_grad_norm)
                    sparsify_update(model.parameters(), p=sampling_prob)
                
            optimizer.step()


            if accountant:
                with torch.no_grad():
                    q = batch_size / len(train_loader.dataset)
                    # NOTE: 
                    # Using combinations within a set of gradients (like below)
                    # does not actually produce samples from the correct distribution
                    # (for that, we need to sample pairs of gradients independently).
                    # However, the difference is not significant, and it speeds up computations.
                    pairs = list(zip(*itertools.combinations(grads_est, 2)))


                    lgrad = torch.stack(pairs[0])
                    rgrad = torch.stack(pairs[1])

                    # print("len_pairs",len(pairs[0]), len(pairs[1]))  #len_pairs 28 28
                    # print("lgrad", lgrad.shape)     #lgrad torch.Size([28, 573834])
                    # print("rgrad", rgrad.shape)     #rgrad torch.Size([28, 573834])



                    #if accountant.__class__.__name__ == "WassersteinPrivacyAccountant":
                    accountant[0].accumulate(
                        scale_param = args.scale*max_grad_norm,
                        subsampling_rate = q,
                        lgrad = lgrad,
                        rgrad = rgrad, 
                    )

                    if len(accountant)>1:
                        accountant[1].accumulate(
                            ldistr=(lgrad, args.scale*max_grad_norm),
                            rdistr=(rgrad, args.scale*max_grad_norm),
                            q=q, 
                            steps=1,
                        )

                    if len(accountant)>2:
                        accountant[2].accumulate(
                            ldistr=(max_grad_norm, args.scale*max_grad_norm),
                            rdistr=(0, args.scale*max_grad_norm),
                            q=q, 
                            steps=1,
                        )



        # print training stats every epoch
        wasser_running_eps = accountant[0].get_privacy(target_delta=1e-5) if accountant else None
        if len(accountant)>1:
            bayes_running_eps = accountant[1].get_privacy(target_delta=1e-5) if accountant else None
        if len(accountant)>2:
            moments_running_eps = accountant[2].get_privacy(target_delta=1e-5) if accountant else None


        print("Epoch: {}/{}. Loss: {:.3f}. Wasserstein Accountant Privacy ({},{}): ({:.6f},{})".format(
                epoch, 
                n_epochs, 
                running_loss / len(train_loader), 
                chr(949),
                chr(948),
                wasser_running_eps[0],wasser_running_eps[1]
                )
        )

        if len(accountant)>1:
            print("Epoch: {}/{}. Loss: {:.3f}. Bayesian Accountant Privacy ({},{}): ({:.6f},{})".format(
                epoch, 
                n_epochs, 
                running_loss / len(train_loader), 
                chr(949),
                chr(948),
                bayes_running_eps[0],bayes_running_eps[1]
                )
            )
        if len(accountant)>2:
            print("Epoch: {}/{}. Loss: {:.3f}. Moments Accountant Privacy ({},{}): ({:.6f},{})".format(
                epoch, 
                n_epochs, 
                running_loss / len(train_loader), 
                chr(949),
                chr(948),
                moments_running_eps[0],moments_running_eps[1]
                )
            )


                
        acc = test(test_loader = test_loader, model = model, device = device)
        accuracies += [acc]
        #print("Test accuracy is %d %%" % acc)
        
    print('Finished Training')
    return model, accuracies









total_steps = args.n_epochs * len(train_loader)


if args.accountant == 'Wasserstein':
    wasser_accountant = WassersteinPrivacyAccountant(order=args.order, steps_total=total_steps, beta=args.beta)

    trained_model, accs = train(train_loader=train_loader, model=model, optimizer=optimizer, 
          criterion=criterion, n_epochs=args.n_epochs, device=device, accountant=wasser_accountant)

    print("Wasserstein DP ({},{}): ".format(chr(949),chr(948)), wasser_accountant.get_privacy(target_delta=1e-5))

elif args.accountant == 'Bayesian':
    bayes_accountant = BayesianPrivacyAccountant(powers=args.order, total_steps=total_steps)
    trained_model, accs = train(train_loader=train_loader, model=model, optimizer=optimizer, 
          criterion=criterion, n_epochs=args.n_epochs, device=device, accountant=bayes_accountant)
    print("Bayesian DP ({},{}): ".format(chr(949),chr(948)), bayes_accountant.get_privacy(target_delta=1e-5))

elif args.accountant == 'all':
    wasser_accountant = WassersteinPrivacyAccountant(order=args.order, steps_total=total_steps, beta=args.beta)
    bayes_accountant = BayesianPrivacyAccountant(powers=args.order, total_steps=total_steps)
    moments_accountant = BayesianPrivacyAccountant(powers=args.order, total_steps=total_steps, bayesianDP=False)

    accountant = list([wasser_accountant, bayes_accountant, moments_accountant])

    trained_model, accs = train(train_loader=train_loader, model=model, optimizer=optimizer, 
          criterion=criterion, n_epochs=args.n_epochs, device=device, accountant=accountant)

    print("Wasserstein DP ({},{}): ".format(chr(949),chr(948)), wasser_accountant.get_privacy(target_delta=1e-5))
    print("Bayesian DP ({},{}): ".format(chr(949),chr(948)), bayes_accountant.get_privacy(target_delta=1e-5))
    print("Classical DP ({},{}): ".format(chr(949),chr(948)), moments_accountant.get_privacy(target_delta=1e-5))







