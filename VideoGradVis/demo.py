from VideoGradVis import grad_vis
import cv2 
import torch
import argparse 
import numpy as np 
from torch import nn
import torch.nn.functional as F 

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
    bp = grad_vis.GuidedBackpropagation(video_model, selected_layer, grad_channels) # Our gradient layer has shape (1,24,120,250,250)
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
