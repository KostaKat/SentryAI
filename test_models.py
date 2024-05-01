
import my_utils as ut
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageFilter
import io
import matplotlib.pyplot as plt
import random
from skimage import data
from scipy.ndimage import rotate
from kernels import *
import os
from torch.utils.data import Subset
from collections import Counter
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import numpy as np
from collections import defaultdict
from torchvision import models

from PIL import Image, ImageFilter
import io
import re
import random
import numpy.random as npr
from skimage import data
from scipy.ndimage import rotate
from kernels import *
import torchvision
import os
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader, Subset
import old_utils_averaged_filters as old_ut_avg
import torchvision.transforms as transforms
from transformers import Swinv2ForImageClassification, SwinConfig
from torch.optim import AdamW
from torchvision import transforms, datasets

class BaseClassifier_MultiImage(nn.Module):
    def __init__(self,kernels):
        super(BaseClassifier_MultiImage, self).__init__()
        self.feature_combiner = ut.CNNBlock(kernels)
        self.feature_combiner2 = ut.CNNBlock(kernels)
        self.classifier = ut.DeepClassifier() 

 
    def forward(self, rich, poor):
       
        x = self.feature_combiner(rich)
        y = self.feature_combiner2(poor)   
        feature_difference = x - y
        outputs = self.classifier(feature_difference)

        return outputs
class ResNetClassifier_MultiImage(nn.Module):
    def __init__(self,kernels):
        super(ResNetClassifier_MultiImage, self).__init__()
        self.feature_combiner = ut.CNNBlock(kernels)
        self.feature_combiner2 = ut.CNNBlock(kernels)
        resnet_weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=resnet_weights)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        # Add a new classifier layer
        self.classifier = nn.Linear(self.resnet.fc.in_features, 2)
        # Adaptive pool to make sure output from feature maps is of fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

 
    def forward(self, rich, poor):
       
        x = self.feature_combiner(rich)
        y = self.feature_combiner2(poor)   
        feature_difference = x - y
        
        # Process feature difference through the ResNet features
        features = self.features(feature_difference)
        pooled_features = self.adaptive_pool(features)
        flat_features = torch.flatten(pooled_features, 1)
        outputs = self.classifier(flat_features)

        return outputs
class SwinClassification_MultiImage(nn.Module):
    def __init__(self,kernels):
        super(SwinClassification_MultiImage, self).__init__()
        self.feature_combiner = ut.CNNBlock(kernels)
        self.feature_combiner2 = ut.CNNBlock(kernels)
        config = SwinConfig.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256',num_classes=2)
        self.transformer = Swinv2ForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256",
            config=config
        )
        
        self.transformer.classifier = nn.Linear(config.hidden_size, 2) 

 
    def forward(self, rich, poor):
       
        x = self.feature_combiner(rich)
        y = self.feature_combiner2(poor)   
        feature_difference = x - y
        outputs = self.transformer(feature_difference)

        return outputs.logits
class SwinClassificationOldest(nn.Module):
    def __init__(self):
        super(SwinClassificationOldest, self).__init__()
        self.feature_combiner = old_ut_avg.CNNBlockOld()
        self.feature_combiner2 = old_ut_avg.CNNBlockOld()
        config = SwinConfig.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256',num_classes=2)
        self.transformer = Swinv2ForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256",
            config=config
        )
        
        self.transformer.classifier = nn.Linear(config.hidden_size, 2) 

 
    def forward(self, rich, poor):
 
        
        x = self.feature_combiner(rich)
        y = self.feature_combiner2(poor) 
        feature_difference = x - y
        outputs = self.transformer(feature_difference)

        return outputs.logits
class BaseClassifierOldest(nn.Module):
    def __init__(self):
        super(BaseClassifierOldest, self).__init__()
        self.feature_combiner = old_ut_avg.CNNBlockOld()
        self.feature_combiner2 = old_ut_avg.CNNBlockOld()
        self.classifier = ut.DeepClassifier() 

 
    def forward(self, rich, poor):
       
  
        
        x = self.feature_combiner(rich)
        y = self.feature_combiner2(poor) 
        feature_difference = x - y
        outputs = self.classifier(feature_difference)

        return outputs
class ResNetClassifierOldest(nn.Module):
    def __init__(self):
        super(ResNetClassifierOldest, self).__init__()
        self.feature_combiner = old_ut_avg.CNNBlockOld()
        self.feature_combiner2 = old_ut_avg.CNNBlockOld()
        resnet_weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=resnet_weights)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        # Add a new classifier layer
        self.classifier = nn.Linear(self.resnet.fc.in_features, 2)
        # Adaptive pool to make sure output from feature maps is of fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

 
    def forward(self, rich, poor):

        
        x = self.feature_combiner(rich)
        y = self.feature_combiner2(poor)    
        feature_difference = x - y
        
        # Process feature difference through the ResNet features
        features = self.features(feature_difference)
        pooled_features = self.adaptive_pool(features)
        flat_features = torch.flatten(pooled_features, 1)
        outputs = self.classifier(flat_features)

        return outputs