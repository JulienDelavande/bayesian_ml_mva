# Bayesian Confidence in ReLU Networks

This project implements the methods described in the paper **"Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks"**. The goal is to improve confidence calibration in neural networks using Bayesian techniques like Local Laplace Approximation (LLLA) and Full Laplace Approximation (FLA). The implementation supports both **binary** and **multi-class classification**. It also explores the impact of different activation functions like **ReLU** and **Tanh** on the confidence calibration of neural networks as well as a closed-form solution for confidence estimation not presented in the paper.

## Authors

 - Anatole Vakili
 - Julien Delavande

## Features
- Binary classification using ReLU or Tanh activation
- Multi-class classification using ReLU or Tanh activation
- Optimization of confidence calibration using LLLA and FLA
- Visualization of decision boundaries
- Confidence estimation with closed-form solutions

## Installation
Ensure you have Python installed along with the required dependencies:
```bash
pip install -r requirements.txt
```

### Main Parameters:
- `--tanh [VALUE]`: Use **Tanh** activation instead of ReLU.
- `--multi_class [VALUE]`: Enable **multi-class classification** (default is binary classification).
- `--optimize [VALUE]`: Optimize **var0** and **temperature** during training.
- `--var0_llla [VALUE]`: Initial variance for **LLLA** approximation (default: `319842`).
- `--var0_fla [VALUE]`: Initial variance for **FLA** approximation (default: `40378`).
- `--temp [VALUE]`: Initial **temperature** scaling (default: `2`).
- `--n_epochs [VALUE]`: Number of training epochs (default: `15000`).
- `--lr [VALUE]`: Learning rate (default: `0.001`).
- `--momentum [VALUE]`: Momentum for SGD optimizer (default: `0.9`).
- `--weight_decay [VALUE]`: Weight decay for SGD optimizer (default: `0.0005`).

### Example Commands:
Train a binary classifier with ReLU activation:
```bash
python train_nn_parser.py --n_epochs 10000 --lr 0.001
```

Train a multi-class classifier with Tanh activation and confidence optimization:
```bash
python train_nn_parser.py --multi_class --tanh --optimize
```

## Visualization
The script provides decision boundary plots for different Bayesian approximations:
- **MAP Scaling** (Maximum a Posteriori)
- **Temperature Scaling**
- **LLLA Scaling** (Local Laplace Approximation)
- **FLA Scaling** (Full Laplace Approximation)

## Reference
Y. Kristiadi, M. Hein, and P. Hennig, "Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks," in NeurIPS 2020. ([Paper Link](https://arxiv.org/abs/2002.10118))

## License
This project is open-source and available under the MIT License.

