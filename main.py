from transformers import AutoImageProcessor, AutoModel
from PIL import Image

import torch
import numpy as np

if __name__ == "__main__":
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    vit = AutoModel.from_pretrained("facebook/dinov2-base").eval().to("cuda")

    hallway_img = Image.open("/home/toyozoshimada/dinov2_test/data/hallway.png").convert("RGB")
    lunchroom_img = Image.open("/home/toyozoshimada/dinov2_test/data/lunchroom.png").convert("RGB")

    input = proc(hallway_img, return_tensors="pt").to("cuda")
    # print(input)
    with torch.no_grad():
        outputs = vit(**input)

    print(outputs.last_hidden_state.shape)  # (1, 197, 768)