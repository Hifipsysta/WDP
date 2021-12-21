



import math
import torch
from torch import Tensor
from typing import List, Optional
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        scale:float,
        C:float,
        batch_size:int,
        device):
    
    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        if i == 0:
            d_p = d_p + torch.normal(0.0, (scale*C)/math.sqrt(batch_size), d_p.shape).to(device)
        param.add_(d_p, alpha=-lr)
        #param.add_(torch.normal(0,1,param.shape).to('cuda'))
        #print('=='*20, i, param.shape)




class SGD(Optimizer):
    def __init__(self, params, grad_clip, C, scale, batch_size, lr=required, device='cuda', 
            momentum=0, dampening=0, weight_decay=0, nesterov=False):
        
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, grad_clip=grad_clip, C=C, scale=scale, batch_size=batch_size, 
                device=device, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(SGD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum'] 
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            grad_clip = group['grad_clip']
            C = group['C']
            scale = group['scale']
            batch_size = group['batch_size']  
            device = group['device']

            
            for p in group['params']:
                if p.grad is not None:
                    p.grad = p.grad / torch.max(torch.tensor([1, p.grad.norm()/C]))
                    #p.grad.data.clamp_(-grad_clip, grad_clip)
                    #print(list(p.grad.size()))
                    
                    #p.grad = p.grad + torch.normal(0.0, scale*C, p.grad.shape).to('cuda')
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else: 
                        momentum_buffer_list.append(state['momentum_buffer'])
	    
            sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    dampening=dampening,
                    nesterov=nesterov,
                    scale=scale,
                    C=C,
                    batch_size=batch_size,
                    device=device)
            
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

