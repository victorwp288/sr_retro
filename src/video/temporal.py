import numpy as np
import cv2


def mean3(frame_buffer):
    if not frame_buffer:
        raise RuntimeError("mean3 requires at least one frame")
    stack = np.stack(frame_buffer, axis=0)
    return stack.mean(axis=0)


def flow_blend(prev_lr, curr_lr, prev_sr, curr_sr, flow_strength=0.5):
    # Farneback operates on 8-bit inputs, so we quantize once when building the flow field.
    prev_gray = cv2.cvtColor((prev_lr * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor((curr_lr * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    sr_height, sr_width = prev_sr.shape[:2]
    lr_height, lr_width = flow.shape[:2]

    scale_h = sr_height / lr_height
    scale_w = sr_width / lr_width

    flow_scaled = cv2.resize(flow, (sr_width, sr_height), interpolation=cv2.INTER_LINEAR)
    flow_scaled[..., 0] *= scale_w
    flow_scaled[..., 1] *= scale_h

    grid_x, grid_y = np.meshgrid(np.arange(sr_width), np.arange(sr_height))
    map_x = (grid_x + flow_scaled[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_scaled[..., 1]).astype(np.float32)
    warped_prev = cv2.remap(
        (prev_sr * 255.0).astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    warped_prev = warped_prev.astype(np.float32) / 255.0
    blended = (1.0 - flow_strength) * curr_sr + flow_strength * warped_prev
    return blended
