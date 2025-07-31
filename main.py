from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

import torch
import cv2
import numpy as np

if __name__ == "__main__":
    img_name = "hallway"

    detector = YOLO("yolo11x-seg.pt")
    detector.fuse()
    detect_res_list: list[Results] = detector.predict(
        source=f"data/{img_name}.png",
        save=False,
        verbose=False,
        stream=False,
        device="cuda",
        conf=0.5,
        half=True,
    )
    detect_res = detect_res_list[0]
    
    # save annotated image
    annotated_img = detect_res.plot()
    cv2.imwrite(f"data/{img_name}_annotated.png", annotated_img)

    # dinov2 feature extraction
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    vit = AutoModel.from_pretrained("facebook/dinov2-base").eval().to("cuda")

    hallway_img = Image.open(f"data/{img_name}.png").convert("RGB")
    print(hallway_img.size)  # (Width, Height)
    input = proc(hallway_img, return_tensors="pt").to("cuda")
    print(input["pixel_values"].shape)  # (Batch, Channel, Height, Width)
    # print(outputs.last_hidden_state.shape)  # (Batch, TokenSize(CLS + Patch), EmbeddingSize)
    with torch.no_grad():
        outputs = vit(**input)

    patch_tokens = outputs.last_hidden_state[:, 1:]
    pa