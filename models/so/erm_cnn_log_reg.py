from model import Model, Optimizer
import numpy as np
import copy

import torch
from torch.nn.functional import cross_entropy

from utils.torch_utils import numpy_to_torch, torch_to_numpy


class ClientModel(Model):

    def __init__(self, lr, embedding_dim, num_classes, max_batch_size=None, seed=None, optimizer=None):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        model = LogisticRegression(self.embedding_dim, self.num_classes).cuda()
        optimizer = ErmOptimizer(model)
        super(ClientModel, self).__init__(lr, seed, max_batch_size, optimizer=optimizer)

    def create_model(self):
        """Model function for linear model."""
        pass

    def process_x(self, raw_x_batch):
        return raw_x_batch

    def process_y(self, raw_y_batch):
        return raw_y_batch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim=10000, output_dim=500):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        output = self.linear(x)
        return output


class ErmOptimizer(Optimizer):

    def __init__(self, model):
        super(ErmOptimizer, self).__init__(torch_to_numpy(model.trainable_parameters()))
        self.optimizer_model = None
        self.learning_rate = None
        self.global_model = None
        self.personalized = False
        self.model = model
        self.global_w = None
        self.global_w_on_last_update = None
        self.pFedMe = False
        self.dynamic_lambda = False
        

    def initialize_w(self):
        self.w = torch_to_numpy(self.model.trainable_parameters())
        self.w_on_last_update = np.copy(self.w)
        if self.personalized:
            self.global_w = torch_to_numpy(self.global_model.trainable_parameters())
            self.global_w_on_last_update = np.copy(self.global_w)

    def reset_w(self, w):
        self. w = np.copy(w)
        self.w_on_last_update = np.copy(w)
        numpy_to_torch(self.w, self.model)
        
    def reset_global_w(self,global_w):
        self. global_w = np.copy(global_w)
        self.global_w_on_last_update = np.copy(global_w)
        numpy_to_torch(self.global_w, self.global_model)

    def end_local_updates(self):
        self.w = torch_to_numpy(self.model.trainable_parameters())
        if self.personalized:
            self.global_w = torch_to_numpy(self.global_model.trainable_parameters())

    def update_w(self):
        self.w_on_last_update = self.w
        if self.personalized:
            self.global_w_on_last_update = self.global_w

    
    def interpolate(self):
        with torch.no_grad():
            for (p,g_p) in zip(self.model.trainable_parameters(),self.global_model.trainable_parameters()):
                p.mul_(1-self.interpolation_ratio)
                p.add_(g_p,alpha=self.interpolation_ratio)
        return True
    
    def interpolate_copy(self,first_time=True):
        if first_time:
            self.interpolate_copy_model = copy.deepcopy(self.model)
        with torch.no_grad():
            for (i_p,p,g_p) in zip(self.interpolate_copy_model.trainable_parameters(),
                                   self.model.trainable_parameters(),
                                   self.global_model.trainable_parameters()):
                i_p.mul_(0)
                i_p.add_(p,alpha=1-self.interpolation_ratio)
                i_p.add_(g_p,alpha=self.interpolation_ratio)
        return True
    
    
    def run_step_global(self,x,y):
        preds = self.global_model(x)
        loss = cross_entropy(preds, y)
        gradient = torch.autograd.grad(loss, self.global_model.trainable_parameters())
        for p, g in zip(self.global_model.trainable_parameters(), gradient):
            p.data -= self.learning_rate * g.data
        return True

    def run_step_global_pFedMe(self,x,y):
        loss = self.lmbda*self.model_dist_norm_var()
        gradient = torch.autograd.grad(loss, self.global_model.trainable_parameters())
        for p, g in zip(self.global_model.trainable_parameters(), gradient):
            p.data -= self.learning_rate * g.data
        return True

    
    def model_dist_norm_var(self, norm=2):
        size = 0
        for layer in self.model.trainable_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        size = 0
        for (p,g_p) in zip(self.model.trainable_parameters(),
                           self.global_model.trainable_parameters()):
            sum_var[size:size + p.view(-1).shape[0]] = ((p - g_p)).view(-1)
            size += p.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def model_dist_norm_var_for_model(self, model, norm=2):
        size = 0
        for layer in model.trainable_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        size = 0
        for (p,g_p) in zip(model.trainable_parameters(),
                           self.global_model.trainable_parameters()):
            sum_var[size:size + p.view(-1).shape[0]] = ((p - g_p)).view(-1)
            size += p.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def tune_lambda(self,x,y):
        all_lambdas = np.array([0.05,0.1,0.3])
        all_losses = []
        all_models = []
        for target_lambda in all_lambdas:
            attempt_model = copy.deepcopy(self.model)
            preds = attempt_model(x)
            loss = cross_entropy(preds, y) + target_lambda*self.model_dist_norm_var_for_model(attempt_model)
            gradient = torch.autograd.grad(loss, attempt_model.trainable_parameters())
            for p, g in zip(attempt_model.trainable_parameters(), gradient):
                p.data -= self.learning_rate * g.data
            record_loss = cross_entropy(attempt_model(x), y) + target_lambda*self.model_dist_norm_var_for_model(attempt_model)
            all_losses.append(record_loss.item())
            all_models.append(attempt_model)
        best_index = np.argmin(np.array(all_losses))
        self.lmbda = all_lambdas[best_index]



    def loss(self, x, y, mode='local'):
        """Compute batch loss on proceesed batch (x, y)"""
        if mode == 'local':
            preds = self.model(x)
            if self.personalized:
                loss = cross_entropy(preds, y) + self.lmbda*self.model_dist_norm_var()
            else:
                loss = cross_entropy(preds, y)
        elif mode == 'interpolate':
            preds = self.interpolate_copy_model(x)
            loss = cross_entropy(preds, y) + self.lmbda*self.model_dist_norm_var()
        return loss.item()

    def gradient(self, x, y, mode='local'):
        if mode == 'local':
            preds = self.model(x)
            if self.personalized:
                loss = cross_entropy(preds, y) + self.lmbda*self.model_dist_norm_var()
            else:
                loss = cross_entropy(preds, y)
            gradient = torch.autograd.grad(loss, self.model.trainable_parameters())
        elif mode == 'interpolate':
            preds = self.interpolate_copy_model(x)
            loss = cross_entropy(preds, y) + self.lmbda*self.model_dist_norm_var()
            gradient = torch.autograd.grad(loss, self.interpolate_copy_model.trainable_parameters())
        return gradient

    def loss_and_gradient(self, x, y):
        preds = self.model(x)
        loss = cross_entropy(preds, y)
        gradient = torch.autograd.grad(loss, self.model.trainable_parameters())
        return loss, gradient
    

    def run_step(self, batched_x, batched_y, lmbda):
        """Run single gradient step on (batched_x, batched_y) and return loss encountered"""
        if self.personalized:
            if self.pFedMe:
                grad = self.gradient(batched_x,batched_y)
                for p, g in zip(self.model.trainable_parameters(), grad):
                    p.data -= self.learning_rate * g.data
                self.run_step_global_pFedMe(batched_x,batched_y)
                for p, g_p in zip(self.model.trainable_parameters(), self.global_model.trainable_parameters()):
                    p.data = g_p.data
            else:
                self.run_step_global(batched_x,batched_y)
                if self.dynamic_lambda:
                    # self.lmbda = np.random.choice([10,15,20])
                    self.tune_lambda(batched_x,batched_y)
                else:
                    self.lmbda = lmbda
                grad = self.gradient(batched_x,batched_y)
                for p, g in zip(self.model.trainable_parameters(), grad):
                    p.data -= self.learning_rate * g.data          
            return self.loss(batched_x,batched_y)
        
        loss, gradient = self.loss_and_gradient(batched_x, batched_y)
        for p, g in zip(self.model.trainable_parameters(), gradient):
            p.data -= self.learning_rate * g.data

        return loss.item()
        


    def correct(self, x, y, mode='local'):
        if mode == 'local':
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
                pred = outputs.argmax(dim=1, keepdim=True)
                return pred.eq(y.view_as(pred)).sum().item()
        elif mode == 'interpolate':
            self.interpolate_copy_model.eval()
            with torch.no_grad():
                outputs = self.model(x)
                pred = outputs.argmax(dim=1, keepdim=True)
                return pred.eq(y.view_as(pred)).sum().item()

    def size(self):
        return len(self.w)