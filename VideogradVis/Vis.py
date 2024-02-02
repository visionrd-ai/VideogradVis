import os
import cv2
import torch
import numpy as np
import torch.nn as nn

class GradVis():
    def __init__(self, model) -> None:
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)
        self.softmax = nn.Softmax(dim=1)
        self.counter = 0

    def overlay_gradients_on_frames(self, frames, gradients):
        overlayed_frames = []
        for i in range(gradients.shape[0]): 
            frame = frames[0][i, :, :, :]
            frame = (frame.detach().cpu())*self.std+self.mean
            # import pdb; pdb.set_trace()
            frame_display = frame.permute(1, 2, 0).numpy() 
            frame_display = (frame_display*255).astype(np.uint8)
            gradient = gradients[i]
            gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            red_gradient = np.zeros_like(frame_display)
            red_gradient[:, :, 2] = gradient 
            overlayed_frame = cv2.addWeighted(frame_display, 0.3, red_gradient, 1.0, 0)
            overlayed_frames.append(overlayed_frame)
        return overlayed_frames
    
    def compute_grad(self, input_tensor, path):
        self.model.eval()
        
        if input_tensor.requires_grad == False:
            print("Input tensor required tensor is not set, update the input tensor to input_tensor.requires_grad_(True)")
            quit
        else:
            print("Input tensor good...")

        output = self.model(input_tensor)
        target = torch.max(output)
        prob = self.softmax(output)

        self.model.zero_grad()
        target.backward()
        gradients = input_tensor.grad.data
        gradients = gradients.abs().sum(dim=1).squeeze(0)
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())
        grads = gradients.detach().cpu().numpy()

        window_size = 3
        rolling_mean = np.empty((grads.shape[0] - window_size + 1, grads.shape[1], grads.shape[2]))
        for i in range(rolling_mean.shape[0]):
            rolling_mean[i] = grads[i:i + window_size].mean(axis=0)

        overlayed_frames = self.overlay_gradients_on_frames(input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[1], input_tensor.shape[3], input_tensor.shape[4]).detach(), rolling_mean)
        video_path = os.path.join(path,f'{self.counter}_{int(torch.argmax(prob).cpu().numpy())}_{torch.max(prob).detach().cpu().numpy()}.mp4')
        self.counter += 1
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (input_tensor.shape[3], input_tensor.shape[3]))

        for frame in overlayed_frames:
            video_writer.write(frame)

        video_writer.release()
        print("GradVis saved!")





