<p align="center">
  <a href="https://ultralytics.com/">
  <img width="700" src="https://github.com/visionrd-ai/.github/assets/145563962/79a92550-c2e4-49f3-8229-bfe6545e54ea"></a>
</p>






## <div align="center">VIDEOGRAD VIS</div>

VideogradVis is an advanced package that encompasses cutting-edge methods for Explainable AI (XAI) in the realm of computer vision. This versatile, video based tool is designed to facilitate the diagnosis of model predictions, serving a dual purpose in both production environments and during the model development phase. Beyond its practical applications, the project also serves as a comprehensive benchmark, offering a robust foundation for researchers to explore and evaluate novel explainability methods.

1. **Compatibility Assurance**: Rigorously tested on a multitude of Common Convolutional Neural Networks (CNNs).

2. **Versatile Applications**: Unleash the full potential of VideogradVis across a spectrum of advanced use cases. From Classification and Object Detection to Semantic Segmentation, Embedding-similarity, and beyond, this suite empowers your projects with unparalleled flexibility and performance.

## <div align="center">APPROACH</div>
1. ***Imports***:
   - The necessary libraries are imported, including OpenCV (cv2), PyTorch (torch), and its modules.

3. ***GuidedBackpropagation Class***: 
   - This class encapsulates the functionality for guided backpropagation.
   - During initialization (__init__), it takes the neural network model, the selected layer index, and the number of channels in the gradient layer as input.
   - It registers hooks on ReLU layers and the first convolutional layer to capture activations during forward and backward passes.
   - The visualize method takes input frames and an optional target class. It computes the guided backpropagation for the input frames and returns the reconstructed image.

4. ***Main Function* (main)**:
   - It loads the pre-trained video classification model and the input video frames.
   - The frames are converted into a tensor suitable for input to the neural network.
   - An instance of GuidedBackpropagation is created with necessary parameters.
   - Guided backpropagation is applied to each frame of the video using the visualize method.
   - The resulting heatmap is overlaid on the original video frames and saved as an output video.

5. ***Argument Parsing***:
   - Command-line arguments are parsed using the argparse module. These arguments include paths to input video, model weights, and output video, as well as parameters like selected layer, frame height, width, and clip length.

6. ***Execution***:
   - The script is executed, and the main function is called with parsed arguments.
  

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

The model demonstrates which pixels are causing the neurons to fire for the case of headlight/blinker on. In the case of blinkers, the small yellow bulb at the corner has significant contribution. In headlight, the entire headlight region contributes to the model's headlight=ON prediction

![Prediction: HeadLight ON](head_out1.gif)


![Prediction: HeadLight ON](head_out2.gif)
***Figure 1: Localized gradients of a video classifier predicting the state of a car headlight (whether it is on, off, or defective)***


![Prediction: HeadLight ON](test_out.gif)


![Prediction: HeadLight ON](test_out2.gif)  
***Figure 2: Localized gradients of a video classifier predicting the state of a car blinkers (whether it is on, off, or defective).***







