import torch
import torch.autograd as autograd
import torch.nn as nn
import matplotlib.pyplot as plt
from hessian import compute_hessian


def closed_form_confidence(model, criterion, X_train, y_train, X, method='LLLA', var0=1e5, apply_sigm=True, diag=False):
    y_pred = model(X_train)
    W_list = list(model.parameters()) if method == 'FLA' else [list(model.parameters())[-1]]
    likelihood = criterion(y_pred.view(-1), y_train)
    reg = 0
    for W in W_list: 
        reg += 1/2 * W.flatten() @ (1/var0 * torch.eye(W.numel())) @ W.flatten()
    loss = likelihood + reg
    
    if diag:
        hessian = compute_hessian(loss, W_list, exact=True)
        hessian = hessian.diag() * torch.eye(hessian.shape[0])
    else:
        hessian = compute_hessian(loss, W_list, exact=True)
    sigma = torch.inverse(hessian)
    confidences = torch.zeros(X.shape[0])
    if method == 'FLA':
        d = []
        f_mu = model(X)
        for i in range(X.shape[0]):
            d_ = autograd.grad(f_mu[i], W_list, retain_graph=True)
            d_flat = torch.cat([d_layer.view(-1) for d_layer in d_])
            d.append(d_flat)
        f_mu = f_mu.view(-1)
        d = torch.stack(d)
        denominator = torch.sqrt(1 + (torch.pi / 8) * torch.diag(d @ sigma @ d.T))
        z = f_mu / denominator
        confidences = torch.sigmoid(z) if apply_sigm  else z
    else: # method == 'LLLA'
        phi = model.feature_extr(X)
        w_map = W_list[0].view(-1) # mu
        numerator =  phi @ w_map
        denominator = torch.sqrt(1 + (torch.pi / 8) * (phi @ sigma @ phi.T).diag())
        z = numerator / denominator
        confidences = torch.sigmoid(z) if apply_sigm  else z
    return confidences


def mc_confidence(model_manyclasses, criterion, X_train, y_train, X, n_classes=4, mc_it=10, method='LLLA', var0=10, diag=False, exact=False, tau=None, force_definite_positive=False):
    
    def set_model_params(W_list, theta_sample):
        idx = 0
        for p in W_list:
            n_params = p.numel()
            p.data = theta_sample[idx:idx+n_params].reshape(p.shape)
            idx += n_params
    tau = 1/var0 if not tau else tau
    
    model_manyclasses.zero_grad()
    
    y_pred = model_manyclasses(X_train)
    W_list = list(model_manyclasses.parameters()) if method == 'FLA' else [list(model_manyclasses.parameters())[-1]]
    likelihood = criterion(y_pred, y_train)
    reg = 0
    for W in W_list: 
        reg += 1/2 * W.flatten() @ (1/var0 * torch.eye(W.numel())) @ W.flatten()
    loss = likelihood + reg

    hessian = compute_hessian(loss, W_list, exact=exact, tau=tau)
    if diag:
        hessian = hessian.diag() * torch.eye(hessian.shape[0])  
        
    sigma = torch.inverse(hessian)
    if force_definite_positive:
        epsilon = 1e-4
        eigvals, eigvecs = torch.linalg.eigh(sigma)  # Décomposition en valeurs propres
        eigvals = torch.clamp(eigvals, min=epsilon)  # Remplace les valeurs trop petites
        sigma = eigvecs @ torch.diag(eigvals) @ eigvecs.T  # Reconstitution
        
    theta_map = torch.cat([W.view(-1) for W in W_list])
    model_manyclasses.eval()

    if method == 'FLA':
        mvn = torch.distributions.MultivariateNormal(theta_map, covariance_matrix=sigma)
        
        confidences = torch.zeros((X.shape[0], n_classes))
        for i in range(X.shape[0]):
            x = X[i].unsqueeze(0)
            theta_samples = mvn.rsample((mc_it,))
            y_samples = []
            for it in range(mc_it):
                set_model_params(W_list, theta_samples[it])
                out = model_manyclasses(x).squeeze()
                out = torch.softmax(out, dim=0)
                y_samples.append(out) # (mc_it, classes)
            
            y_samples = torch.stack(y_samples)
            y_mean = y_samples.mean(dim=0) # (classes)
            confidences[i, :] = y_mean # (batch, classes)

    else:
        with torch.no_grad():
            confidences = torch.zeros((X.shape[0], n_classes))
            mvn = torch.distributions.MultivariateNormal(theta_map, covariance_matrix=sigma)
            W = list(model_manyclasses.parameters())[-1]
            for i in range(X.shape[0]):
                x = X[i].unsqueeze(0)
                phi_test = model_manyclasses.feature_extr(x).squeeze()
                py = 0
                for _ in range(mc_it):
                    W_s = mvn.rsample().view(W.shape)
                    py += torch.softmax(phi_test @ W_s.t(), dim=0)
    
                py /= mc_it
                confidences[i, :] = py
    return confidences

def close_form_confidence_multi_class(model_manyclasses, X_train, y_train, criterion=nn.CrossEntropyLoss(reduction='sum'), method='LLLA', var0=10, folder='data', activation_func='ReLU'):
    y_pred = model_manyclasses(X_train)
    W_list = [list(model_manyclasses.parameters())[-1]] if method == 'LLLA' else list(model_manyclasses.parameters())
    likelihood = criterion(y_pred, y_train)
    reg = 0
    for W in W_list: 
        reg += 1/2 * W.flatten() @ (1/var0 * torch.eye(W.numel())) @ W.flatten()
    loss = likelihood + reg

    hessian = compute_hessian(loss, W_list, exact=True, tau=None)
        
    sigma = torch.inverse(hessian)

    size = 50
    test_range = (-15, 15)
    xx, yy = torch.meshgrid(torch.linspace(*test_range, size, dtype=torch.float32),
                        torch.linspace(*test_range, size, dtype=torch.float32),
                        indexing='ij')
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)  # Shape (10000, 2)



    f_mu = model_manyclasses(grid)
    out_4 = torch.zeros(grid.shape[0], 4)
    for out_dim in range(4):
        d = []
        for i in range(grid.shape[0]):
            d_ = autograd.grad(f_mu[i][out_dim], W_list, retain_graph=True)
            d_flat = torch.cat([d_layer.view(-1) for d_layer in d_])
            d.append(d_flat)
        f_mu_ = f_mu[:, out_dim].view(-1)
        d = torch.stack(d)
        denominator = torch.sqrt(1 + (torch.pi / 8) * torch.diag(d @ sigma @ d.T))
        out_4[:, out_dim] = f_mu_/denominator
    confidences = torch.softmax(out_4, dim=1).max(1).values.detach()

    plt.contourf(xx, yy, confidences.view(50, 50), alpha=0.6, cmap='coolwarm')
    plt.colorbar()

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', edgecolors='k')
    plt.title(f'{method}_{activation_func}_cf')
    plt.savefig(f'{folder}/{method}_{activation_func}_cf.png', bbox_inches='tight', pad_inches=0)

    plt.show()