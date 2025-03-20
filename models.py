import torch.nn as nn

class BinaryNNReLU(nn.Module):
    def __init__(self, n=2, h=20):
        super(BinaryNNReLU, self).__init__()
        
        self.feature_extr = nn.Sequential(
            nn.Linear(n, h),
            nn.ReLU(), 
            nn.Linear(h, h), 
            nn.ReLU(),
        )
        
        self.clf = nn.Linear(h, 1, bias=False)
        
    def forward(self, x):
        x = self.feature_extr(x)
        return self.clf(x)

class BinaryNNTanh(nn.Module):
    def __init__(self, n=2, h=20):
        super(BinaryNNTanh, self).__init__()
        
        self.feature_extr = nn.Sequential(
            nn.Linear(n, h),
            nn.Tanh(), 
            nn.Linear(h, h), 
            nn.Tanh(),
        )
        
        self.clf = nn.Linear(h, 1, bias=False)
        
    def forward(self, x):
        x = self.feature_extr(x)
        return self.clf(x)
    
class ManyClassesNNReLU(nn.Module):
    def __init__(self, n=2, h=20, out=4):
        super(ManyClassesNNReLU, self).__init__()
        
        self.feature_extr = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.ReLU(), 
            nn.Linear(h, h), 
            nn.BatchNorm1d(h),
            nn.ReLU(),
        )
        
        self.clf = nn.Linear(h, out, bias=False)
        
    def forward(self, x):
        x = self.feature_extr(x)
        return self.clf(x)
    
class ManyClassesNNTanh(nn.Module):
    def __init__(self, n=2, h=20, out=4):
        super(ManyClassesNNTanh, self).__init__()
        
        self.feature_extr = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.Tanh(), 
            nn.Linear(h, h), 
            nn.BatchNorm1d(h),
            nn.Tanh(),
        )
        
        self.clf = nn.Linear(h, out, bias=False)
        
    def forward(self, x):
        x = self.feature_extr(x)
        return self.clf(x)