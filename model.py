import torch
import torch.nn as nn
import torch.nn.functional as F

class GrowNet(object):

    def __init__(self,inputSize,outputSize,startLayers=[64,64],maxWidth=512,maxDepth=3):
        self._model = GrowNetSubModel(inputSize,outputSize,startLayers)
        self._inputSize = inputSize
        self._outputSize = outputSize
        self._currentLayers = startLayers
        self._maxWidth = maxWidth
        self._maxDepth = maxDepth

    def grow(self):
        for i, layer in enumerate(self._currentLayers):
            if layer!=self._maxWidth:
                newWidth = min(layer*2,self._maxWidth)
                self._widen(newWidth,i)
                
                return ((i+1)==self._maxDepth)&(newWidth==self._maxWidth)
        
        if len(self._currentLayers)<self._maxDepth:
            self._deepen(self._maxWidth//4)
            return False
        else:
            return True
        
    
    def _widen(self,new_width,layer):
        size = self._currentLayers.copy()
        size[layer]=new_width

        newModel = GrowNetSubModel(self._inputSize,self._outputSize,size)

        for name, layer in newModel.state_dict().items():
            alternate = self._model.state_dict()[name]
            if len(layer.size())==1:
                layer[:alternate.size()[0]]=alternate
            else:
                layer[:alternate.size()[0],:alternate.size()[1]]=alternate
        
        self._model = newModel
        self._currentLayers = size

    def _deepen(self, new_layer_width):
        size = self._currentLayers.copy()
        size.append(new_layer_width)

        newModel = GrowNetSubModel(self._inputSize,self._outputSize,size)

        for name, layer in newModel.state_dict().items():
            if name in self._model.state_dict().keys():
                alternate = self._model.state_dict()[name]
                if len(layer.size())==1:
                    layer[:alternate.size()[0]]=alternate
                else:
                    layer[:alternate.size()[0],:alternate.size()[1]]=alternate
            
        self._model = newModel
        self._currentLayers = size
        


class GrowNetSubModel(nn.Module):

    def __init__(self,inputSize,outputSize,layers=[64,64]):
        super(GrowNetSubModel, self).__init__()

        self.hidden = nn.ModuleList()

        self.hidden.append(nn.Linear(inputSize,layers[0]))

        for k in range(len(layers)-1):
            self.hidden.append(nn.Linear(layers[k],layers[k+1]))

        self.hidden.append(nn.Linear(layers[-1],outputSize))

    def forward(self, x):
        for i, l in enumerate(self.hidden):
            x = F.relu(l(x))

        return x 