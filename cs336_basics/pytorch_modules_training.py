import torch
import torch.nn as nn
import torch.nn.init as init
import math
import numpy as np
from einops import reduce

def cross_entropy(logits, targets):

    verbose = False

    # logits are B, L, V
    if verbose: print("logits", logits.shape)
    # Targets are B, L
    if verbose: print("targets", targets.shape)
    
    if verbose: 
        ma = logits.max(dim=-1, keepdim=False).values # B, L
        print("ma (no keep dim)", ma.shape)

    ma = logits.max(dim=-1, keepdim=True).values # B, L, 1
    if verbose: print("ma", ma.shape)

    logits = logits-ma # B, L, V (broadcast)

    bot_exp = torch.exp(logits) # B, L, V
    if verbose: print("bot_exp", bot_exp.shape)

    bot_exp_sum = bot_exp.sum(dim=-1, keepdim=False) # B, L
    if verbose: print("bot_exp_sum", bot_exp_sum.shape)

    top_fetch = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1) # B, L, 1
        ).squeeze(-1) # B, L due to squeeze
    if verbose: print("top_fetch", top_fetch.shape)

    return (- top_fetch + torch.log(bot_exp_sum)).mean()

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas= (0.9, 0.95), eps = 1e-8, weight_decay=0.01):
        # assert False
        params, alpha, beta1, beta2, epsilon, lam = params, lr, betas[0], betas[1], eps, weight_decay
        defaults = {
            'a':alpha, 
            'b1': beta1, 
            'b2': beta2, 
            'e': epsilon, 
            'l': lam,
            }
        
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for param_group in self.param_groups:
            b1, b2 = param_group['b1'], param_group['b2']
            a = param_group['a']
            e = param_group['e']
            lam = param_group['l']

            for i, p in enumerate(param_group["params"]):

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                if p.grad is None:
                    continue

                grad = p.grad.data  # Get gradient of loss with respect to p.

                v1 = state.get('v1', torch.zeros_like(p))
                v2 = state.get('v2', torch.zeros_like(p))

                v1 = (v1*b1) + (grad*(1-b1))
                v2 = (v2*b2) + ((grad**2)*(1-b2))

                top = math.sqrt(1-(b2 ** t))
                bot = 1-(b1 ** (t))
                new = a * top / bot

                p.data = p.data - new * (v1 / (torch.sqrt(v2)+e))
                p.data = p.data - a*lam*p.data

                state["v1"] = v1
                state["v2"] = v2
                state["t"] = t + 1 
        
        return loss

def learning_rate_schedule(t, amax, amin, tw, tc):
    if t < tw:
        return (t/tw)*amax
    if t <= tc:
        return amin + (0.5) * (1+math.cos(math.pi * (t-tw) / (tc-tw))) * (amax-amin)
    return amin

def gradient_clipping(params, maxg, epsilon=1e-6):
    params_lst = []
    global_l2 = 0

    for p in params:
        g = p.grad

        if g is not None:
            params_lst.append(p)
            n = g.norm(p=2).item()
            global_l2 += n**2
    global_l2 = math.sqrt(global_l2)

    if global_l2 > maxg:
        for p in params_lst:
            p.grad = p.grad * maxg / (global_l2+epsilon)

    return

def data_loader(x, batch_size, context_length, device):
    befores = []
    afters = []

    for _ in range(batch_size):
        i = np.random.randint(0, len(x)-context_length)
        befores.append(x[i:i+context_length])
        afters.append(x[i+1:i+context_length+1])

    befores = np.array(befores)
    afters = np.array(afters)

    befores = torch.from_numpy(
        np.ascontiguousarray(befores)
    ).to(torch.device(device), dtype=torch.long, non_blocking=True)
    afters = torch.from_numpy(
        np.ascontiguousarray(afters)
    ).to(torch.device(device), dtype=torch.long, non_blocking=True)
    return befores, afters

def save_checkpoint(model, optimizer, iteration, out):
    ret = {}
    ret["model"] = model.state_dict()
    ret["optimizer"] = optimizer.state_dict()
    ret['iter'] = iteration
    torch.save(ret, out)
    return

def load_checkpoint(src, model, optimizer):
    ret = torch.load(src) 
    model.load_state_dict(ret["model"])
    optimizer.load_state_dict(ret["optimizer"])
    return ret['iter']
