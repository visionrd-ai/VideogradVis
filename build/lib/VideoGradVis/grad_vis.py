import cv2 
import torch
import argparse 
import numpy as np 
from torch import nn
import torch.nn.functional as F 

class GuidedBackpropagation:
    def __init__(self, model, selected_layer, grad_layer_channels):
        self.model = model
        self.image_reconstruction = None  
        self.activation_maps = []  
        self.model.eval()
        self.conv3d = nn.Conv3d(in_channels=grad_layer_channels, out_channels=3, kernel_size=3, padding=1)
        self.grad_scaling_factor = 10e10
        self.selected_layer = selected_layer
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0]*self.grad_scaling_factor
            # print("First layer hook called")

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)
            # print("Forward hook called")

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop()
            grad[grad > 0] = 1
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad
            # print("Backward hook called")
            return (new_grad_in,)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)

        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv3d):
                first_layer = module
                if count == self.selected_layer:
                    break
                count+=1
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, input_frames, target_class=None):
        model_output = self.model(input_frames)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()

        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)
        if target_class is not None:
            grad_target_map[:, target_class] = 1
        else:
            grad_target_map[:, pred_class] = 1

        # Perform backward pass
        model_output.backward(grad_target_map)

        # Visualize reconstructed image
        convd = self.conv3d(self.image_reconstruction.data)
        result =  F.interpolate(F.relu(convd), size=(input_frames.shape[2], input_frames.shape[3], input_frames.shape[4]), mode='trilinear', align_corners=False)
        return result.detach().cpu().numpy()

def main(args):

    from pipeline.infer_vrd import get_video_classifier
    from pipeline.pipeline_utils import get_video_tensor

    grad_channels = 24
    selected_layer = args.selected_layer
    car_cfg = {'cls_w':args.width, 'cls_h':args.height}

    video_model = get_video_classifier()
    weights = torch.load(args.weights_path)
    video_model.load_state_dict(weights)

    video_capture = cv2.VideoCapture(args.input_video)
    frames = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if ret:
            frames.append(frame)
        else:
            break
    video_capture.release()

    video_tensor = get_video_tensor(frames, car_cfg, clip_length=args.clip_length)
    bp = GuidedBackpropagation(video_model, selected_layer, grad_channels) # Our gradient layer has shape (1,24,120,250,250)
    result = bp.visualize(video_tensor, 0)

    result = np.squeeze(result)
    result = np.moveaxis(result, 0, -1)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_video, fourcc, 30.0, (frames[0].shape[1]*2, frames[0].shape[0]))

    for idx, frame_res in enumerate(result):
        hm = np.mean(frame_res, axis=-1)
        hm = cv2.normalize(hm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_HOT)
        hm_resized = cv2.resize(hm_color, (frames[idx].shape[1], frames[idx].shape[0]))
        frame2 = frames[idx]
        weighted = cv2.addWeighted(frame2, 0.7, hm_resized, 0.9, 0)
        stacked = np.hstack((weighted, hm_resized))
        out.write(stacked)

    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process guided backpropagation for video.")
    parser.add_argument("--input_video", type=str, help="Path to input video")
    parser.add_argument("--weights_path", type=str, help="Path to model weights")
    parser.add_argument("--output_video", type=str, help="Path to save output video")
    parser.add_argument("--selected_layer", type=int, help="The layer from which gradients are required")
    parser.add_argument("--height", type=int, help="Frame height")
    parser.add_argument("--width", type=int, help="Frame width")
    parser.add_argument("--clip_length", type=int, help="Number of frames")
    args = parser.parse_args()
    main(args)
