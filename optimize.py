from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from confidence import closed_form_confidence, mc_confidence

def optimize_var0(X_val, y_val, model, X_train, y_train, method='FLA', lamda=1.0, it=300, min_logvar0=-10, max_logvar0=20, diag=False, multi_class=False):
    logvar0s = torch.linspace(min_logvar0, max_logvar0, it)
    nlls = []
        
    X_out = torch.rand(50, 2) * 6 - 3 
    y_out = torch.full((50,), 0.5)
    
    pbar = tqdm(logvar0s, position=0, leave=True)

    for logvar0 in pbar:
        var0 = torch.exp(logvar0)
        
        if not multi_class:
            out_in = closed_form_confidence(model, nn.BCEWithLogitsLoss(reduction='sum'), X_train, y_train, X_val, var0=var0, method=method, apply_sigm=False, diag=diag)
            loss_in = nn.BCEWithLogitsLoss()(out_in, y_val).detach().item()
            out_out = closed_form_confidence(model, nn.BCEWithLogitsLoss(reduction='sum'), X_train, y_train, X_out, var0=var0, method=method, apply_sigm=False, diag=diag)
            loss_out = nn.BCEWithLogitsLoss()(out_out, y_out).detach().item()
        else:
            try:
                out_in = mc_confidence(model, nn.CrossEntropyLoss(reduction='sum'), X_train, y_train, X_val, n_classes=4, method=method, var0=var0, diag=True)
                loss_in = nn.CrossEntropyLoss()(out_in, y_val).detach().item()
                out_out = mc_confidence(model, nn.CrossEntropyLoss(reduction='sum'), X_train, y_train, X_out, n_classes=4, method=method, var0=var0, diag=True)
                loss_out = nn.CrossEntropyLoss()(out_out, y_out).detach().item()
            except:
                continue
        
        loss = loss_in + lamda*loss_out
        nlls.append(loss)

        pbar.set_description(f'var0: {var0:.3f}, Loss: {loss:.3f}, loss_in: {loss_in:.3f}, loss_out: {loss_out:.3f}')
            
    best_logvar0 = logvar0s[np.nanargmin(np.array(nlls))]
    best_var0 = torch.exp(best_logvar0)
    print(f'Best var0: {best_var0}')

    return best_var0


def optimize_temp(model, X_test, y_test, many_classes=False):
    logit_val = model(X_test).squeeze().detach()
    
    T = torch.tensor(1).float()
    T.requires_grad = True
    
    optimizer = optim.LBFGS([T], lr=0.1, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        if many_classes:
            loss = nn.CrossEntropyLoss()(logit_val/T, y_test)
        else:
            loss = nn.BCEWithLogitsLoss()(logit_val/T, y_test)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    T = T.detach().item()
    print(f'Temp: {T}')
    
    return T