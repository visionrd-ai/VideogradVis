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
<div align="left">
  <figure>
      <img width="20%" src="output.gif" alt="Prediction: HeadLight ON">
      <figcaption>Figure 1: Grads showing why model predicted Headlight is ON.</figcaption>
  </figure>
</div>

<div align="center">
  <figure>
      <img width="20%" src="output1.gif" alt="Prediction: HeadLight ON">
      <figcaption>Figure 1: Grads showing why model predicted Headlight is ON.</figcaption>
  </figure>
</div>

<div align="rght">
  <figure>
      <img width="20%" src="output3.gif" alt="Prediction: HeadLight OFF">
      <figcaption>Figure 1: Grads showing why model predicted Headlight is OFF.</figcaption>
  </figure>
</div>



