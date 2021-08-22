import torch
import torch.nn as nn
import torch.nn.functional as F

class GrowNet(nn.Module):

    def __init__(self,input_Shape,Output_Shape,layers=[64,64]):
        super(GrowNet, self).__init__()

        self.hidden = nn.ModuleList()

        self.hidden.append(nn.Linear(input_Shape,layers[0]))

        for k in range(len(layers)-1):
            self.hidden.append(nn.Linear(layers[k],layers[k+1]))

        self.hidden.append(nn.Linear(layers[-1],Output_Shape))

    def forward(self, x):
        for i, l in enumerate(self.hidden):
            x = l(x)

        return x
