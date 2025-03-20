import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from confidence import closed_form_confidence, mc_confidence

def plot_decision_boundary(X_train, y_train, model, method='MAP', var0=1e5, diag=False, title='MAP Scaling', criterion=nn.BCEWithLogitsLoss(reduction='sum'), T = 1, folder='data', activation_func='ReLU'):
    x_min, x_max = 1*(X_train[:, 0].min() - 1), 1*(X_train[:, 0].max() + 1)
    y_min, y_max = 1*(X_train[:, 1].min() - 1), 1*(X_train[:, 1].max() + 1)
    
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100, dtype=torch.float32),
                        torch.linspace(y_min, y_max, 100, dtype=torch.float32),
                        indexing='ij')
    
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    
    
    if method == 'FLA' or method == 'LLLA': # Laplace
        preds = closed_form_confidence(model, criterion, X_train, y_train, grid, method=method, var0=var0, diag=diag).reshape(xx.shape).detach()
    else: # method == 'MAP'
        with torch.no_grad():
            logits = model(grid).squeeze()
            Temp = T if method == 'TEMP' else 1
            preds = torch.sigmoid(logits/Temp).reshape(xx.shape)

    confidence = 2 * preds - 1
    
    plt.contourf(xx, yy, confidence, alpha=0.6, cmap='coolwarm')
    plt.colorbar()
    plt.contour(xx, yy, preds, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
    
    plt.title(title)
    plt.savefig(f'{folder}/{method}_{activation_func}.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_decision_boundary_many_classes(X_train, y_train, model, method='MAP', var0=1e5, diag=True, title='MAP Scaling',
                           criterion=nn.CrossEntropyLoss(reduction='sum'), T=1, n_classes=4, mc_it=10, exact=False, tau=1e-4, folder='data', activation_func='ReLU'):
    size = 50
    test_range = (-15, 15)
    xx, yy = torch.meshgrid(torch.linspace(*test_range, size, dtype=torch.float32),
                        torch.linspace(*test_range, size, dtype=torch.float32),
                        indexing='ij')
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)  # Shape (10000, 2)

    if method in ['FLA', 'LLLA']:  # Approximation Laplace
        preds = mc_confidence(model, criterion, X_train, y_train, grid, n_classes=n_classes, mc_it=mc_it, method=method, var0=var0, diag=diag, exact=exact, tau=tau)
        preds = preds.max(dim=1).values.reshape(xx.shape).detach()
    else:  # MAP ou TEMP
        with torch.no_grad():
            logits = model(grid)  # (10000, n_classes)
            temp = T if method == 'TEMP' else 1
            preds = torch.softmax(logits / temp, dim=1)  # Probas pour chaque classe
            # preds = -np.sum(preds*np.log(preds + 1e-8), axis=1)
            preds = preds.max(dim=1).values.reshape(xx.shape)  # Prend la confiance max

    plt.contourf(xx, yy, preds, alpha=0.6, cmap='coolwarm')
    plt.colorbar()

    # Points des données
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', edgecolors='k')
    plt.savefig(f'{folder}/{method}_{activation_func}.png', bbox_inches='tight', pad_inches=0)

    plt.title(title)
    plt.show()
