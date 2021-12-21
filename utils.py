import os
import ot
import torch
import scipy.stats
import numpy as np


def kl_divergence(P,Q):
    #print(type(P)==torch.Tensor)
    #P * np.log(P) - P * np.log(Q)
    return np.sum(P * np.log(P / Q))

def renyi_divergence(alpha, P, Q):
    if alpha == 1:
        out = np.sum(P * np.log(P / Q))
    elif alpha != 1:
        exp = np.sum(Q* ((P/Q) ** alpha))
        out = 1/(alpha - 1) * np.log(exp)
    return out

def wasserstein_distance(u_values, v_values, order):
    out = ot.wasserstein_1d(x_a=_values, x_b=v_values, p=order)
    return out


def RDP_Laplace_epsilon(alpha, scale):
	item_left = alpha/(2*alpha-1) * np.exp((alpha-1)/scale)
	item_right = (alpha-1)/(2*alpha-1) * np.exp(-alpha/scale)
	return 1/(alpha-1) * np.log(item_left + item_right)


def RDP_Gaussian_epsilon(alpha, std):
	return alpha/(2 * std**2)


def DP_Laplace_epsilon(sensitivity, scale):
	return sensitivity/scale


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:

        tmp = np.max(x,axis=1)
        x -= tmp.reshape((x.shape[0],1))
        x = np.exp(x)
        tmp = np.sum(x, axis = 1)
        x /= tmp.reshape((x.shape[0], 1))
    else:

        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


def compute_renyi_wasserstein_privacy(model, mechanism, sensitivity, scale):
    Renyi_privacy_list = []
    Wasser_privacy_list = []
    pure_privacy_list = []

    param_net = list(model.parameters())
    for layer in range(len(param_net)):
        # print(param.size())
        param = param_net[layer].data.reshape(1,-1)
        param = param.squeeze()
        #noise = torch.cuda.FloatTensor(param.size()) if torch.cuda.is_available() else torch.FloatTensor(param.size())
        
        param_size = param.size()[0]
        
        if mechanism == 'Gaussian':
            noise1 = np.random.normal(loc=0, scale=scale, size=param_size)
            noise2 = np.random.normal(loc=sensitivity, scale=scale, size=param_size)
        elif meachanism == 'Laplace':
            noise1 = np.random.laplace(loc=0, scale=scale, size=param_size)
            noise2 = np.random.laplace(loc=sensitivity, scale=scale, size=param_size)
	    
        param = param.cpu()
	#print(param.device)
	#print(noise.device)
	#noise = torch.normal(means=0.0,std=1.0, out=noise)
        param1 = param.numpy() + noise1
        param2 = param.numpy() + noise2
        Wasser_privacy_loss = ot.wasserstein_1d(x_a=param1, x_b=param2, p=1)
        Renyi_privacy_loss = renyi_divergence(P=softmax(param1) , Q=softmax(param2), alpha=1)
        
        pure_privacy_loss = np.max(np.log(softmax(param1)) - np.log(softmax(param2)))
        #print(f'| Number of pixels: {param_size:02} | Privacy Loss: {Wasser_privacy_loss:.3f} | Renyi loss: {Renyi_privacy_loss:.3f}')
        Wasser_privacy_list.append([layer+1, param_size, Wasser_privacy_loss])
        Renyi_privacy_list.append([layer+1, param_size, Renyi_privacy_loss])
        pure_privacy_list.append([layer+1, param_size, pure_privacy_loss])
    return Wasser_privacy_list, Renyi_privacy_list, pure_privacy_list

