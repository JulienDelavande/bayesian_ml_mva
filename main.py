from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import accuracy_score
from torch import nn, optim
import torch
import copy
from models import BinaryNNReLU, BinaryNNTanh, ManyClassesNNReLU, ManyClassesNNTanh
import tqdm
import argparse
from optimize import optimize_var0, optimize_temp
from plot import plot_decision_boundary, plot_decision_boundary_many_classes
from confidence import close_form_confidence_multi_class

TANH = False
MULTI_CLASS = True
OPTIMIZE = True

# Parameters binary
VAR0_LLLA = 319842
VAR0_FLA = 40378
# Parameters multi-class
# VAR0_LLLA = 1
# VAR0_FLA = 0.1

TEMP = 2
N_EPOCHS = 15000
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network for binary or multi-class classification.")
    parser.add_argument("--tanh", type=bool, default=TANH, help="Use Tanh activation instead of ReLU")
    parser.add_argument("--multi_class", type=bool, default=MULTI_CLASS, help="Use multi-class classification instead of binary classification")
    parser.add_argument("--optimize", type=bool, default=OPTIMIZE, help="Optimize var0 and temperature")
    parser.add_argument("--var0_llla", type=float, default=VAR0_LLLA, help="Initial var0 for LLLA")
    parser.add_argument("--var0_fla", type=float, default=VAR0_FLA, help="Initial var0 for FLA")
    parser.add_argument("--temp", type=float, default=TEMP, help="Initial temperature")
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay for SGD optimizer")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    activation_func = 'ReLU' if not args.tanh else 'Tanh'
    if not args.multi_class:
        print(f'Binary classification with {activation_func} activation function...')
        X_train, y_train = make_moons(n_samples=90, noise=0.1, shuffle=True, random_state=42)
        X_val, y_val = make_moons(n_samples=200, noise=0.1, shuffle=True, random_state=42)
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
        
        model = BinaryNNReLU() if not args.tanh else BinaryNNTanh()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        print('Training model...')
        pbar = tqdm.tqdm(range(args.n_epochs), position=0, leave=True)
        for epoch in pbar:
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs.view(-1), y_train)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'epoch : {epoch}, loss training : {loss.item():.5f}')
        print(f'loss training : {loss.item()}')

        with torch.no_grad():
            logits = model(X_val)
            preds = (logits.squeeze() > 0).long()
            accuracy = accuracy_score(y_val, preds)
        print(f"Accuracy : {accuracy:.2f}")
        
        
        if args.optimize:
            print('Optimizing var0 and temperature...')
            var0_llla = optimize_var0(X_val, y_val, model, X_train, y_train, method='LLLA')
            var0_fla = optimize_var0(X_val, y_val, model, X_train, y_train, method='FLA', it=50)
            temp = optimize_temp(model, X_val, y_val)
        else:
            var0_llla = args.var0_llla
            var0_fla = args.var0_fla
            temp = args.temp
        
        print('Plotting decision boundaries...')
        plot_decision_boundary(X_train, y_train, model, method='MAP', title='MAP Scaling', activation_func=activation_func)
        plot_decision_boundary(X_train, y_train, model, method='TEMP', T=temp, title='TEMP Scaling', activation_func=activation_func)
        plot_decision_boundary(X_train, y_train, model, method='LLLA', var0=var0_llla, diag=False, title='LLLA Scaling', activation_func=activation_func)
        plot_decision_boundary(X_train, y_train, model, method='FLA', var0=var0_fla, diag=False, title='Full Laplace Scaling', activation_func=activation_func)
        
        
    else:
        print(f'Multi-class classification with {activation_func} activation function...')
        X_train, y_train = make_blobs(n_samples=500, centers=4, cluster_std=1.2, center_box=(-10, 10), random_state=42)
        X_val, y_val = make_blobs(n_samples=100, centers=4, cluster_std=1.2, center_box=(-10, 10), random_state=42)
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
        X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
        
        model_manyclasses = ManyClassesNNReLU() if not args.tanh else ManyClassesNNTanh()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_manyclasses.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        print('Training model...')
        pbar = tqdm.tqdm(range(args.n_epochs), position=0, leave=True)
        for epoch in pbar:
            optimizer.zero_grad()
            outputs = model_manyclasses(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'epoch : {epoch}, loss training : {loss.item():.5f}')
        print(f'loss training : {loss.item()}')
        model_manyclasses.eval()
        
        if args.optimize:
            print('Optimizing var0 and temperature...')
            try:
                var0_llla = optimize_var0(X_val, y_val, model_manyclasses, X_train, y_train, method='LLLA', multi_class=True, it=100)
                var0_fla = optimize_var0(X_val, y_val, model_manyclasses, X_train, y_train, method='FLA', multi_class=True, it=50)
                temp = optimize_temp(model_manyclasses, X_val, y_val, many_classes=True)
            except Exception as e:
                print(e)
                print('Optimization failed, using default values')
                var0_llla = args.var0_llla
                var0_fla = args.var0_fla
                temp = args.temp
        else:
            var0_llla = args.var0_llla
            var0_fla = args.var0_fla
            temp = args.temp
    
        print('Plotting decision boundaries...')
        plot_decision_boundary_many_classes(X_train, y_train, model_manyclasses, method='MAP', title='MAP Scaling')
        plot_decision_boundary_many_classes(X_train, y_train, model_manyclasses, method='TEMP', T=temp, title='TEMP Scaling')
        try:
            plot_decision_boundary_many_classes(X_train, y_train, copy.deepcopy(model_manyclasses), method='LLLA', var0=var0_llla, diag=False, title='LLLA Scaling', activation_func=activation_func)
        except Exception as e:
            print(e)
            print('LLLA failed, using enforce diag')
            plot_decision_boundary_many_classes(X_train, y_train, copy.deepcopy(model_manyclasses), method='LLLA', var0=var0_llla, diag=True, title='LLLA Scaling', activation_func=activation_func)
        try:    
            plot_decision_boundary_many_classes(X_train, y_train, copy.deepcopy(model_manyclasses), method='FLA', var0=var0_fla, diag=False, title='Full Laplace Scaling', activation_func=activation_func)
        except Exception as e:
            print(e)
            print('FLA failed, using enforce diag')
            plot_decision_boundary_many_classes(X_train, y_train, copy.deepcopy(model_manyclasses), method='FLA', var0=var0_fla, diag=True, title='Full Laplace Scaling', activation_func=activation_func)
            
        close_form_confidence_multi_class(model_manyclasses, X_train, y_train, criterion=nn.CrossEntropyLoss(reduction='sum'), method='LLLA', var0=var0_llla, activation_func=activation_func)
        close_form_confidence_multi_class(model_manyclasses, X_train, y_train, criterion=nn.CrossEntropyLoss(reduction='sum'), method='FLA', var0=var0_fla, activation_func=activation_func)
        
