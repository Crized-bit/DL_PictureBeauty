import numpy as np
import cv2

def tile_resample_processor(img: np.ndarray, thr_a: float = 1.) -> np.ndarray:
    height, width, _  = img.shape

    H, W = int(float(height) / float(thr_a)), int(float(width) / float(thr_a))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    resize_by = 1024 / min(H, W)
    new_width, new_height = int(W * resize_by) // 64 * 64, int(H * resize_by) // 64 * 64
    
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def default_preprocessor(img: np.ndarray) -> np.ndarray:
    height, width, _  = img.shape

    resize_by = 1024 / min(height, width)
    new_width, new_height = int(width * resize_by) // 64 * 64, int(height * resize_by) // 64 * 64

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)