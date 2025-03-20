import torch
import torch.autograd as autograd

def compute_hessian(loss, params, exact=True, tau=1e-4):
    grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grads_flat = torch.cat([grad_params.view(-1) for grad_params in grads])
    hessian = torch.zeros(grads_flat.shape[0], grads_flat.shape[0])                    
    for i in range(grads_flat.shape[0]):
        hessian_row = autograd.grad(grads_flat[i], params, retain_graph=True)
        hessian_row = torch.cat([hessian_row_params.view(-1) for hessian_row_params in hessian_row])
        hessian[i, :] = hessian_row
    if not exact:
        hessian = hessian + tau*torch.eye(hessian.shape[0])
    return hessian 

def compute_gn_hessian(likelyhood, params, exact=False, tau=1e-4):
    grads = autograd.grad(likelyhood, params, create_graph=True, retain_graph=True)
    grads_flat = torch.cat([grad_params.view(-1) for grad_params in grads])
    diag_F = grads_flat ** 2
    if not exact:
        hessian = tau + diag_F
    return hessian * torch.eye(grads_flat.shape[0])