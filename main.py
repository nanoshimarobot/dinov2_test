from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path
import time

class ObjectSimilarityVisualizer:
    """Compare the objects detected in two images with DINOv2 features."""

    def __init__(
        self,
        yolo_weights: str = "yolo11x-seg.pt",
        dinov2_name: str = "facebook/dinov2-base",
        device: str = "cuda",
        conf: float = 0.5,
    ) -> None:
        self.device = device
        # --- YOLO detector --------------------------------------------------
        self.detector = YOLO(yolo_weights)
        self.detector.fuse()
        self.conf = conf
        # --- DINOv2 ---------------------------------------------------------
        self.proc = AutoImageProcessor.from_pretrained(dinov2_name)
        self.vit = (
            AutoModel.from_pretrained(dinov2_name)
            .eval()
            .to(self.device)
        )
        self.patch = 14  # DINOv2 uses 14-pixel patches

    # ====================================================================== #
    #  public API                                                            #
    # ====================================================================== #
    def compare_images(
        self,
        img_path_top: str | Path,
        img_path_bottom: str | Path,
        out_path: str | Path = "similarity.png",
    ) -> np.ndarray:
        """
        Run the pipeline and save a visualization image.
        Returns the similarity matrix (top_objects × bottom_objects).
        """
        # t0 = time.perf_counter()
        data_top = self._process_single_image(Path(img_path_top))
        data_bot = self._process_single_image(Path(img_path_bottom))

        # cosine similarity (features are L2-normalised)
        sims = (data_top.vecs @ data_bot.vecs.T).cpu().numpy()  # shape (Nt, Nb)

        vis = self._draw_visualization(data_top, data_bot, sims)
        cv2.imwrite(str(out_path), vis)
        print(f"[✓] saved → {out_path}")
        return sims

    # ====================================================================== #
    #  internal helpers                                                      #
    # ====================================================================== #
    class _ImageData:
        """Bundled outputs for one image."""
        def __init__(
            self,
            orig_img: np.ndarray,        # RGB H×W×3
            annotated: np.ndarray,       # BGR with YOLO boxes
            boxes_xyxy: np.ndarray,      # float Nx4 in orig coords
            centres: np.ndarray,         # int  Nx2  "
            vecs: torch.Tensor,          # (N, 768)  L2-normed
        ):
            self.orig_img = orig_img
            self.annotated = annotated
            self.boxes_xyxy = boxes_xyxy
            self.centres = centres
            self.vecs = vecs  # on CUDA

    # ------------------------------------------------------------------ #
    def _process_single_image(self, path: Path) -> _ImageData:
        whole_time_start = time.perf_counter_ns()
        # ---------- 1. YOLO detection ------------------------------------
        yolo_time_start = time.perf_counter_ns()
        res: Results = self.detector.predict(
            source=str(path),
            stream=False,
            save=False,
            verbose=False,
            device=self.device,
            conf=self.conf,
            half=True,
        )[0]
        yolo_time_end = time.perf_counter_ns()

        annotated = res.plot()  # BGR uint8
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # (N,4) float
        centres = np.column_stack(
            (
                (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2,
                (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2,
            )
        ).astype(int)

        # ---------- 2. DINOv2 region vectors -----------------------------
        dino_time_start = time.perf_counter_ns()
        orig_pil = Image.open(path).convert("RGB")
        dinov_inputs = self.proc(orig_pil, return_tensors="pt").to(self.device)
        dino_proc_end = time.perf_counter_ns()
        with torch.no_grad():
            out = self.vit(**dinov_inputs)
        dino_time_end = time.perf_counter_ns()

        # build patch-grid feature map (B=1, C=768, H_feat, W_feat)
        patch_tokens = out.last_hidden_state[:, 1:]  # drop CLS
        N = patch_tokens.shape[1]
        H_feat = W_feat = int(N ** 0.5)
        feat_map = (
            patch_tokens.reshape(1, H_feat, W_feat, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # scale boxes to ViT input size
        w_orig, h_orig = orig_pil.size
        _, _, h_in, w_in = dinov_inputs["pixel_values"].shape
        scale_x, scale_y = w_in / w_orig, h_in / h_orig
        boxes_scaled = np.column_stack(
            (
                np.zeros(len(boxes_xyxy)),  # batch idx
                boxes_xyxy[:, 0] * scale_x,
                boxes_xyxy[:, 1] * scale_y,
                boxes_xyxy[:, 2] * scale_x,
                boxes_xyxy[:, 3] * scale_y,
            )
        )
        boxes_tensor = torch.tensor(
            boxes_scaled, dtype=feat_map.dtype, device=feat_map.device
        )

        # RoIAlign → (N, 768)
        roi_feats = torchvision.ops.roi_align(
            feat_map,
            boxes_tensor,
            output_size=(1, 1),
            spatial_scale=1.0 / self.patch,
            aligned=True,
        ).squeeze(-1).squeeze(-1)
        vecs = torch.nn.functional.normalize(roi_feats, dim=1)  # L2-norm
        dino_obj_extract_end = time.perf_counter_ns()

        print(f"[✓] processed {path.name} in \n"
              f"{(yolo_time_end - yolo_time_start) / 1e6:.2f} ms (YOLO), \n"
              f"{(dino_proc_end - dino_time_start) / 1e6:.2f} ms (DINO pre-proc), \n"
              f"{(dino_time_end - dino_proc_end) / 1e6:.2f} ms (DINO inference), \n"
              f"{(dino_obj_extract_end - dino_time_end) / 1e6:.2f} ms (DINO obj extract), \n"
              f"total {(dino_obj_extract_end - whole_time_start) / 1e6:.2f} ms")

        return self._ImageData(
            orig_img=np.array(orig_pil),
            annotated=annotated,
            boxes_xyxy=boxes_xyxy,
            centres=centres,
            vecs=vecs,
        )

    # ------------------------------------------------------------------ #
    def _draw_visualization(
        self,
        top: _ImageData,
        bot: _ImageData,
        sims: np.ndarray,
    ) -> np.ndarray:
        """Stack annotated images vertically and draw similarity lines + labels."""
        # ------ ベース画像（上・下を縦連結） ---------------------------------
        pad = 20  # 白い余白（px）
        white = 255 * np.ones(
            (pad, max(top.annotated.shape[1], bot.annotated.shape[1]), 3),
            dtype=np.uint8,
        )
        vis = np.vstack((top.annotated, white, bot.annotated))
        h_top = top.annotated.shape[0]   # 下画像を描く位置調整用

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        offset = 8                       # 文字を線から離す距離(px)

        # --------- 線とスコアを描画 ---------------------------------------
        for i, (cx1, cy1) in enumerate(top.centres):
            for j, (cx2, cy2) in enumerate(bot.centres):
                sim = sims[i, j]
                if sim < 0.8:
                    continue
                pt1 = (int(cx1), int(cy1))
                pt2 = (int(cx2), int(cy2) + h_top + pad)
                # 線
                cv2.line(
                    vis, pt1, pt2,
                    color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA
                )
                # 中点
                mid = np.array([(pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2])

                # ---------- テキスト位置を垂直方向にシフト -----------------
                # 直線 (dx, dy) の垂直方向ベクトル (-dy, dx)
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                norm = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
                # 偶奇で左右交互に振る → さらに重なりにくい
                sign = -1 if (i + j) % 2 == 0 else 1
                shift = sign * offset * np.array([-dy / norm, dx / norm])
                text_pos = tuple((mid + shift).astype(int))

                cv2.putText(
                    vis,
                    f"{sim:.2f}",
                    text_pos,
                    FONT,
                    0.5,               # fontScale
                    (0, 0, 255),       # 赤
                    1,
                    cv2.LINE_AA,
                )

        # ----------- BBox 中心に丸 ---------------------------------------
        for cx, cy in top.centres:
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 0), -1, cv2.LINE_AA)
        for cx, cy in bot.centres:
            cv2.circle(vis, (int(cx), int(cy) + h_top + pad), 4, (0, 255, 0), -1, cv2.LINE_AA)

        return vis



# --------------------------------------------------------------------------- #
# Example usage:
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    vis = ObjectSimilarityVisualizer(device="cuda")
    sim = vis.compare_images("data/complex0.png", "data/complex1.png", "data/compare_complex.png")
    print("similarity matrix:\n", sim)
