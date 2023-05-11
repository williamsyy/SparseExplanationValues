import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class SimpleLR(nn.Module):
    def __init__(self, n_features,data_map, data_mean, sev_penalty = 0.01, positive_penalty = 1):
        super(SimpleLR, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        # create the datamap for the dataset
        self.data_map = torch.tensor(np.array(data_map),dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean),dtype=torch.float32).view(-1,1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        # add the sev_penalty to the model
        self.sev_penalty = sev_penalty
        # add the positive penalty to the model
        self.positive_penalty = positive_penalty

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)

    def predict(self,x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = torch.sigmoid(self.linear(x_torch))
        return (logits.squeeze()>0.5).detach().cpu().numpy()

    def predict_proba(self,x):
        x_torch = torch.tensor(np.array(x),dtype=torch.float32)
        logits = torch.sigmoid(self.linear(x_torch))
        res = torch.cat([1-logits.T,logits.T]).T
        return res.detach().cpu().numpy()

class SimpleMLP(nn.Module):
    def __init__(self, n_features, n_hidden, data_map, data_mean, sev_penalty=0.01, negative_penalty=1):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.activation = nn.ReLU()
        
        self.data_map = torch.tensor(np.array(data_map), dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean), dtype=torch.float32).view(-1, 1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        
        self.sev_penalty = sev_penalty
        self.negative_penalty = negative_penalty

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        logits = self.output(x)
        return torch.sigmoid(logits)

    def predict(self, x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = self.forward(x_torch)
        return (logits.squeeze() > 0.5).detach().cpu().numpy()

    def predict_proba(self, x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = self.forward(x_torch)
        res = torch.cat([1-logits.T,logits.T]).T
        return res.detach().cpu().numpy()

class SimpleGBDT(nn.Module):
    def __init__(self, base_model,data_map, data_mean, sev_penalty = 0.01, negative_penalty = 1):
        super(SimpleGBDT, self).__init__()
        # create the datamap for the dataset
        self.data_map = torch.tensor(np.array(data_map),dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean),dtype=torch.float32).view(-1,1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        # add the sev_penalty to the model
        self.sev_penalty = sev_penalty
        # add the negative penalty to the model
        self.negative_penalty = negative_penalty
        # add the information for Gradient
        self.estimators = base_model.estimators_
        self.linear = nn.Linear(len(self.estimators), 1,bias=False)
        self.linear.weights = torch.tensor(np.ones(len(self.estimators))*base_model.learning_rate,dtype=torch.float,requires_grad=True)
        self.bias_predictor = base_model.init_

    def forward(self, x):
        if len(x.shape)==2:
            y_pred = self.linear(torch.tensor([estimator[0].predict(x) for estimator in self.estimators]).transpose(0,1).float()).transpose(0,1)
            bias = torch.tensor(self.bias_predictor.predict_proba(x.detach().numpy())[:,1]).float()
            bias = torch.log(bias/(1-bias))
            out = torch.sigmoid(y_pred+bias)
            return out.view(-1)
        if len(x.shape)==3:
            y_preds = torch.cat([self.linear(torch.tensor([estimator[0].predict(xi) for estimator in self.estimators]).transpose(0,1).float()).transpose(0,1) for xi in x])
            biases = torch.cat([torch.tensor(self.bias_predictor.predict_proba(x.detach().numpy())[:,1]).float()])
            biases = torch.log(biases/(1-biases)).unsqueeze(1)
            out = torch.sigmoid(y_preds+biases)
            return out

    def predict(self,x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        result = self(x_torch)
        return (result>0.5).detach().cpu().numpy()

    def predict_proba(self,x):
        x_torch = torch.tensor(np.array(x),dtype=torch.float32)
        logits = self(x_torch)
        res = torch.cat([1-logits.T,logits.T]).T
        return res.detach().cpu().numpy()

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SEVMeanLoss(nn.Module):
    def __init__(self,model):
        super(SEVMeanLoss, self).__init__()
        self.model = model

    def SEV_Loss(self,x):
        temp_X1 = torch.unsqueeze(self.model.data_map,2) * torch.transpose(x,0,1)
        dataset = torch.stack([torch.tensor(np.where(self.model.data_map==0,self.model.data_mean_map,torch.transpose(i,0,1))) for i in torch.transpose(temp_X1,0,2)],1)
        out = self.model(dataset)
        y_pred = self.model(x)
        loss = -torch.sum(torch.clamp(torch.max(out,dim=0).values-0.5,max= 0.05)*torch.gt(y_pred,0.5))/(torch.sum(torch.gt(y_pred,0.5))+1e-10)
        return self.model.sev_penalty * loss
    
    def Population_Negative_Loss(self):
        # make sure the population mean is negative
        loss = torch.clamp(self.model(self.model.data_mean.view(1,-1)) - 0.5,max=0)
        # if self.model(self.model.data_mean.view(1,-1)) [0] > 0.5:
        #     print("Warning: The baseline is still positive!")
        return self.model.negative_penalty * loss

    def forward(self, output, target, x):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        sev_loss = self.SEV_Loss(x)
        negative_loss = self.Population_Negative_Loss()
        return loss, loss+ sev_loss + negative_loss, sev_loss

class SEVVolumeLoss(nn.Module):
    def __init__(self,model):
        super(SEVVolumeLoss, self).__init__()
        self.model = model

    def SEV_Loss(self,x):
        edge = torch.abs(self.model.linear(self.model.data_mean.view(1,-1))/self.model.linear.weight)
        edge = torch.clamp(edge,min=1e-10)
        loss = torch.log(edge)
        loss = torch.mean(loss)
        return self.model.sev_penalty * loss
    
    def Population_Negative_Loss(self):
        # make sure the population mean is negative
        loss = torch.clamp(self.model(self.model.data_mean.view(1,-1)) - 0.5,max=0)
        # if self.model(self.model.data_mean.view(1,-1)) [0] > 0.5:
        #     print("Warning: The baseline is still positive!")
        return self.model.negative_penalty * loss

    def forward(self, output, target, x):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        sev_loss = self.SEV_Loss(x)
        negative_loss = self.Population_Negative_Loss()
        return loss, loss+ sev_loss + negative_loss, sev_loss

class SEVCounterfactualLoss(nn.Module):
    def __init__(self,model):
        super(SEVCounterfactualLoss, self).__init__()
        self.model = model

    def SEV_Loss(self,x):
        temp_X1 = torch.stack([x for _ in range(self.model.data_map.shape[0])],0)
        dataset = torch.stack([torch.tensor(torch.where(self.model.data_map==0,i,self.model.data_mean_map)) for i in torch.transpose(temp_X1,0,1)],1)
        out = self.model(dataset)
        y_pred = self.model(x)
        loss = torch.sum(torch.clamp(torch.min(out,dim=0).values-0.5,min= -0.05)*torch.gt(y_pred,0.5))/(torch.sum(torch.gt(y_pred,0.5))+1e-10)
        return self.model.sev_penalty * loss
    
    def Population_Negative_Loss(self):
        # make sure the population mean is negative
        loss = torch.clamp(self.model(self.model.data_mean.view(1,-1)) - 0.5,max=0)
        # if self.model(self.model.data_mean.view(1,-1)) [0] > 0.5:
        #     print("Warning: The baseline is still positive!")
        return self.model.negative_penalty * loss

    def forward(self, output, target, x):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        sev_loss = self.SEV_Loss(x)
        negative_loss = self.Population_Negative_Loss()
        return loss, loss+ sev_loss + negative_loss, sev_loss

class SEVActionableCounterfactualLoss(nn.Module):
    def __init__(self,model,choices):
        super(SEVActionableCounterfactualLoss, self).__init__()
        self.model = model
        print(choices)
        self.feature_selection = choices
        
    def SEV_Loss(self,x):
        temp_X1 = torch.stack([x for _ in range(self.model.data_map.shape[0])],0)
        dataset = torch.stack([torch.tensor(torch.where(self.model.data_map==0,i,self.model.data_mean_map)) for i in torch.transpose(temp_X1,0,1)],1)
        out = self.model(dataset)
        out = out[self.feature_selection]
        y_pred = self.model(x)
        loss = torch.sum(torch.clamp(torch.min(out,dim=0).values-0.5,min= -0.05)*torch.gt(y_pred,0.5))/(torch.sum(torch.gt(y_pred,0.5))+1e-10)
        return self.model.sev_penalty * loss
    
    def Population_Negative_Loss(self):
        # make sure the population mean is negative
        loss = torch.clamp(self.model(self.model.data_mean.view(1,-1)) - 0.5,max=0)
        # if self.model(self.model.data_mean.view(1,-1)) [0] > 0.5:
        #     print("Warning: The baseline is still positive!")
        return self.model.negative_penalty * loss

    def forward(self, output, target, x):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        sev_loss = self.SEV_Loss(x)
        negative_loss = self.Population_Negative_Loss()
        return loss, loss+ sev_loss + negative_loss, sev_loss

class OriginalLoss(nn.Module):
    def __init__(self,model):
        super(OriginalLoss, self).__init__()
        self.model = model

    def forward(self, output, target):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        return loss

def calculate_accuracy(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    correct = (y_pred == y_true).float().sum()
    return correct / len(y_true)