VideogradVis Library to Visulaize Pytorch models Gradients to understand model depth 

## <div align="center">Installation</div>
```bash
python setup.py install
```

## <div align="center">Usage</div>
```bash
import torch
from VideogradVis.Vis import GradVis

odel = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
model.blocks[5].proj = nn.Linear(in_features = 2048, out_features = args.classes, bias = True)
print(f"Loading weights: {args.checkpoint}.")
model.load_state_dict(torch.load(args.checkpoint))

vis = GradVis(model)

model_input = torch.rand((1,3,120,250,250)).requires_grad_(True) #(B,C,T,H,W)
vis.compute_grad(input_tensor=input_model, path='/home/osama/pytorch-video/output/')

```
# Model Predictions Visualization

This document provides visualizations explaining the model's predictions on whether a headlight is ON or OFF. Each figure offers insights into the gradients and features that influenced the model's decision-making process.

## Visualizations

### Figure 1: Visualization Explaining Model's Prediction That the Headlight is ON

![Prediction: HeadLight ON](output.gif)  
*Figure 1: Visualization explaining the model's prediction that the headlight is ON.*

### Figure 2: Detailed Grads Highlighting Reasons Behind the ON Headlight Prediction

![Prediction: HeadLight ON](output1.gif)  
*Figure 2: Detailed grads highlighting reasons behind the ON headlight prediction.*

### Figure 3: Insights Into the Model Predicting the Headlight as OFF

![Prediction: HeadLight OFF](output3.gif)  
*Figure 3: Insights into the model predicting the headlight as OFF.*




