from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

def detect_and_generate_mask(image_folder, mask_scale=1.0, model_path="best.pt"):
    model = YOLO(model_path)
    mask_folder = os.path.join(image_folder, "mask")
    os.makedirs(mask_folder, exist_ok=True)
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".png") and not filename.endswith("_mask.png"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Fail to read: {image_path}")
                continue

            h, w, _ = image.shape

            # YOLO 检测
            results = model(image_path)[0]

            mask = np.zeros((h, w), dtype=np.uint8)

            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)

                box_w = x2 - x1
                box_h = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                new_w = box_w * mask_scale
                new_h = box_h * mask_scale

                new_x1 = int(max(cx - new_w / 2, 0))
                new_y1 = int(max(cy - new_h / 2, 0))
                new_x2 = int(min(cx + new_w / 2, w))
                new_y2 = int(min(cy + new_h / 2, h))

                cv2.rectangle(mask, (new_x1, new_y1), (new_x2, new_y2), color=255, thickness=-1)

            base, _ = os.path.splitext(filename)
            mask_filename = f"{base}_mask.png"
            mask_path = os.path.join(mask_folder, mask_filename)

            cv2.imwrite(mask_path, mask)
            print(f"Mask saved to: {mask_path}")
            
 
detect_and_generate_mask("images", mask_scale=1.3)


