import numpy as np
import torch
import torch.nn as nn
#import cv2
import os
from PIL import Image as im
from matplotlib import pyplot as plt
import skimage
from skimage.measure import block_reduce as AVG_POOL

class FrameGenModel(nn.Module):
    def __init__(self, in_height, in_width, feature_layers, combine_layers, generation_layers, loss_function, learning_rates: dict):
        super().__init__()
        self.in_h  = in_height
        self.in_w  = in_width
        self.loss_func = loss_function
        
        self.featureModel = torch.nn.Sequential(*feature_layers)
        self.combineModel = torch.nn.Sequential(*combine_layers)
        self.generationModel = torch.nn.Sequential(*generation_layers)
        self.skip_gen_layer  = len(generation_layers) == 1
        
        self.feature_optimizer = torch.optim.Adam(self.featureModel.parameters(), lr=learning_rates['features'])
        self.combine_optimizer = torch.optim.Adam(self.combineModel.parameters(), lr=learning_rates['combine'])
        if not(self.skip_gen_layer): self.generation_optimizer = torch.optim.Adam(self.generationModel.parameters(), lr=learning_rates['generation'])
        
    def make_batch(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        return x
        
    def features_forward(self, in_x):
        #print(in_x)
        in_x = self.make_batch(in_x)
        features_x = self.featureModel(in_x)
        return features_x
    
    def combine_forward(self, features_x):
        features_x = self.make_batch(features_x)
        combine_x = self.combineModel(features_x)
        return combine_x
    
    def generation_forward(self, combine_x):
        combine_x = self.make_batch(combine_x)
        image_out_x = self.generationModel(combine_x)
        return image_out_x
    
    def forward(self, in_x_1, in_x_2):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        features_x_1  = self.features_forward(in_x_1)
        features_x_2  = self.features_forward(in_x_2)
        features_x    = torch.cat((features_x_1, features_x_2), dim = 1)
        combine_x     = self.combine_forward(features_x)
        image_out_x   = self.generation_forward(combine_x)
        return image_out_x
    
    def backprop(self, in_x_1, in_x_2, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        out_x  = self.make_batch(out_x)
            
        self.feature_optimizer.zero_grad()
        self.combine_optimizer.zero_grad()
        if not(self.skip_gen_layer): self.generation_optimizer.zero_grad()
        
        image_out_x = self.forward(in_x_1, in_x_2)
        loss = self.loss_func(image_out_x, out_x)
        loss.backward()
        
        self.feature_optimizer.step()
        self.combine_optimizer.step()
        if not(self.skip_gen_layer): self.generation_optimizer.step()
        
        return loss.item()
    
    def eval_loss_total(self, in_x_1, in_x_2, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        out_x  = self.make_batch(out_x)
        
        image_out_x = self.forward(in_x_1, in_x_2)
        loss = self.loss_func(image_out_x, out_x)
        
        return loss.item()
    
    def eval_loss_per(self, in_x_1, in_x_2, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        out_x  = self.make_batch(out_x)
        
        image_out_x = self.forward(in_x_1, in_x_2)
        loss_per = torch.mean(torch.square(image_out_x - out_x), dim = (1,2,3))
        
        if len(loss_per.size()) > 1: loss_per.squeeze(1)
        assert len(loss_per.size()) == 1
        return loss_per
        
    def eval_acc(self, in_x_1, in_x_2, out_x, cutoff):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        out_x  = self.make_batch(out_x)
        
        loss_per = self.eval_loss_per(in_x_1, in_x_2, out_x)
        acc_per  = torch.minimum(cutoff/loss_per, torch.zeros_like(loss_per) + 1)
        return float(torch.mean(acc_per))
    
class FlowCatFrameGenModel(nn.Module):
    def __init__(self, in_height, in_width, feature_layers, generation_layers, loss_function, learning_rates: dict):
        super().__init__()
        self.in_h  = in_height
        self.in_w  = in_width
        self.loss_func = loss_function
        
        self.featureModel = torch.nn.Sequential(*feature_layers)
        self.generationModel = torch.nn.Sequential(*generation_layers)
        self.skip_gen_layer  = len(generation_layers) == 1
        
        self.feature_optimizer = torch.optim.Adam(self.featureModel.parameters(), lr=learning_rates['features'])
        if not(self.skip_gen_layer): self.generation_optimizer = torch.optim.Adam(self.generationModel.parameters(), lr=learning_rates['generation'])
        
    def make_batch(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        return x
        
    def features_forward(self, in_x):
        in_x = self.make_batch(in_x)
        features_x = self.featureModel(in_x)
        return features_x
    
    def generation_forward(self, combine_x):
        combine_x = self.make_batch(combine_x)
        image_out_x = self.generationModel(combine_x)
        return image_out_x
    
    def forward(self, in_x_1, in_x_2, in_x_3):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        x      = torch.cat((in_x_1, in_x_2, in_x_3), dim = 1)
        features_x     = self.features_forward(x)
        image_out_x   = self.generation_forward(features_x)
        return image_out_x
    
    def backprop(self, in_x_1, in_x_2, in_x_3, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        out_x  = self.make_batch(out_x)
            
        self.feature_optimizer.zero_grad()
        if not(self.skip_gen_layer): self.generation_optimizer.zero_grad()
        
        image_out_x = self.forward(in_x_1, in_x_2, in_x_3)
        loss = self.loss_func(image_out_x, out_x)
        loss.backward()
        
        self.feature_optimizer.step()
        if not(self.skip_gen_layer): self.generation_optimizer.step()
        
        return loss.item()
    
    def eval_loss_total(self, in_x_1, in_x_2, in_x_3, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        out_x  = self.make_batch(out_x)
        
        image_out_x = self.forward(in_x_1, in_x_2, in_x_3)
        loss = self.loss_func(image_out_x, out_x)
        
        return loss.item()
    
    def eval_loss_per(self, in_x_1, in_x_2, in_x_3, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        out_x  = self.make_batch(out_x)
        
        image_out_x = self.forward(in_x_1, in_x_2, in_x_3)
        loss_per = torch.mean(torch.square(image_out_x - out_x), dim = (1,2,3))
        
        if len(loss_per.size()) > 1: loss_per.squeeze(1)
        assert len(loss_per.size()) == 1
        return loss_per
        
    def eval_acc(self, in_x_1, in_x_2, in_x_3, out_x, cutoff):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        out_x  = self.make_batch(out_x)
        
        loss_per = self.eval_loss_per(in_x_1, in_x_2, in_x_3, out_x)
        acc_per  = torch.minimum(cutoff/loss_per, torch.zeros_like(loss_per) + 1)
        return float(torch.mean(acc_per))
    
class AdvancedFlowFrameGenModel(nn.Module):
    def __init__(self, in_height, in_width, feature_layers, generation_layers, mask_layers, loss_function, learning_rates: dict):
        super().__init__()
        self.in_h  = in_height
        self.in_w  = in_width
        self.loss_func = loss_function
        
        self.featureModel = torch.nn.Sequential(*feature_layers)
        self.generationModel = torch.nn.Sequential(*generation_layers)
        self.maskModel = torch.nn.Sequential(*mask_layers)
        self.skip_gen_layer  = len(generation_layers) == 1
        
        self.feature_optimizer = torch.optim.Adam(self.featureModel.parameters(), lr=learning_rates['features'])
        if not(self.skip_gen_layer): self.generation_optimizer = torch.optim.Adam(self.generationModel.parameters(), lr=learning_rates['generation'])
        
    def make_batch(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        return x
        
    def features_forward(self, in_x):
        in_x = self.make_batch(in_x)
        features_x = self.featureModel(in_x)
        return features_x
    
    def generation_forward(self, combine_x):
        combine_x = self.make_batch(combine_x)
        image_out_x = self.generationModel(combine_x)
        return image_out_x
    
    def mask_forward(self, flow_x):
        flow_x = self.make_batch(flow_x)
        mask_x = (self.maskModel(flow_x) <= 0.5).long()
        return mask_x
    
    def forward(self, in_x_1, in_x_2, in_x_3):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        x      = torch.cat((in_x_1, in_x_2, in_x_3), dim = 1)
        features_x     = self.features_forward(x)
        image_out_x   = self.generation_forward(features_x)
        mask_x = self.mask_forward(in_x_3)
        res_image = (-(mask_x - 1)) * image_out_x + mask_x * 0.5 * (in_x_1 + in_x_2)
        #res_image = image_out_x * mask_x
        return res_image
    
    def backprop(self, in_x_1, in_x_2, in_x_3, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        out_x  = self.make_batch(out_x)
            
        self.feature_optimizer.zero_grad()
        if not(self.skip_gen_layer): self.generation_optimizer.zero_grad()
        
        image_out_x = self.forward(in_x_1, in_x_2, in_x_3)
        loss = self.loss_func(image_out_x, out_x)
        loss.backward()
        
        self.feature_optimizer.step()
        if not(self.skip_gen_layer): self.generation_optimizer.step()
        
        return loss.item()
    
    def eval_loss_total(self, in_x_1, in_x_2, in_x_3, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        out_x  = self.make_batch(out_x)
        
        image_out_x = self.forward(in_x_1, in_x_2, in_x_3)
        loss = self.loss_func(image_out_x, out_x)
        
        return loss.item()
    
    def eval_loss_per(self, in_x_1, in_x_2, in_x_3, out_x):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        out_x  = self.make_batch(out_x)
        
        image_out_x = self.forward(in_x_1, in_x_2, in_x_3)
        loss_per = torch.mean(torch.square(image_out_x - out_x), dim = (1,2,3))
        
        if len(loss_per.size()) > 1: loss_per.squeeze(1)
        assert len(loss_per.size()) == 1
        return loss_per
        
    def eval_acc(self, in_x_1, in_x_2, in_x_3, out_x, cutoff):
        in_x_1 = self.make_batch(in_x_1)
        in_x_2 = self.make_batch(in_x_2)
        in_x_3 = self.make_batch(in_x_3)
        out_x  = self.make_batch(out_x)
        
        loss_per = self.eval_loss_per(in_x_1, in_x_2, in_x_3, out_x)
        acc_per  = torch.minimum(cutoff/loss_per, torch.zeros_like(loss_per) + 1)
        return float(torch.mean(acc_per))
    
    