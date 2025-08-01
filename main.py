from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

import torch, torchvision
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
    B, N, D = patch_tokens.shape  # N=H_feat*W_feat
    H_feat = W_feat = int(N**0.5)  # 16 × 16
    feat_map = (
        patch_tokens.reshape(B, H_feat, W_feat, D)
        .permute(0, 3, 1, 2)  # (1, 768, 16, 16)
        .contiguous()
    )

    # ----- 2. YOLO の BBox を ViT 入力サイズへスケーリング ------------------------
    w_orig, h_orig = hallway_img.size  # PIL: (W, H)
    _, _, h_in, w_in = input["pixel_values"].shape  # ViT 実入力サイズ

    scale_x = w_in / w_orig
    scale_y = h_in / h_orig

    boxes_xyxy = detect_res.boxes.xyxy.cpu().numpy()  # (num, 4)
    scaled_boxes = [
        [
            0,  # batch_idx = 0 （常に 0）
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ]
        for (x1, y1, x2, y2) in boxes_xyxy
    ]

    boxes_tensor = torch.tensor(
        scaled_boxes, device=feat_map.device, dtype=feat_map.dtype
    )

    # ----- 3. RoIAlign で領域特徴を抽出 -----------------------------------------
    patch_size = 14  # DINOv2 は 14 px 固定
    roi_feats = torchvision.ops.roi_align(
        feat_map,
        boxes_tensor,
        output_size=(1, 1),  # 1×1 平均プール → (num, C, 1, 1)
        spatial_scale=1.0 / patch_size,
        aligned=True,
    )  # (num_bbox, 768, 1, 1)

    region_vecs = roi_feats.squeeze(-1).squeeze(-1)  # (num_bbox, 768)
    region_vecs = torch.nn.functional.normalize(region_vecs, dim=1)

    print("region_vecs:", region_vecs.shape)  # 例: torch.Size([N, 768])

    with torch.no_grad():
        sim_mat = (region_vecs @ region_vecs.t()).cpu().numpy()  # (N, N)

    num_obj = len(boxes_xyxy)
    centers = np.column_stack(
        (
            (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2,  # cx
            (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2,  # cy
        )
    ).astype(
        int
    )  # (N, 2) in original img coord

    # ========= 4. 類似度＋BBox の可視化 ==========================================
    # region_vecs は L2 正規化済みなので内積＝コサイン類似度
    with torch.no_grad():
        sim_mat = (region_vecs @ region_vecs.t()).cpu().numpy()   # (N, N)

    num_obj = len(boxes_xyxy)
    centers = np.column_stack((
        (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2,   # cx
        (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2    # cy
    )).astype(int)                                   # (N, 2)

    # ------------ 可視化キャンバス ------------- #
    # YOLO が BBox を描いた annotated_img (BGR) をコピーして使う
    vis_img = annotated_img.copy()

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(num_obj):
        for j in range(i + 1, num_obj):
            sim = float(sim_mat[i, j])
            # ① 2 点間に線を描く
            pt1 = tuple(centers[i])
            pt2 = tuple(centers[j])
            cv2.line(vis_img, pt1, pt2, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # ② 線の中央にスコアをテキスト表示
            mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            text = f"{sim:.2f}"
            cv2.putText(
                vis_img,
                text,
                mid_point,
                FONT,
                fontScale=0.4,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    # ------------- 既存 for 2重ループで線とスコアを描画 -----------------
    for i in range(num_obj):
        for j in range(i + 1, num_obj):
            sim = float(sim_mat[i, j])
            pt1 = tuple(centers[i])
            pt2 = tuple(centers[j])
            cv2.line(vis_img, pt1, pt2, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(vis_img, f"{sim:.2f}", mid, FONT, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    # ---------- 追加：結び目（中心点）に丸を描く ------------------------
    for cx, cy in centers:
        cv2.circle(
            vis_img,
            (int(cx), int(cy)),   # 中心座標
            radius=4,             # 半径は適宜調整
            color=(0, 255, 0),    # 緑
            thickness=-1,         # 塗りつぶし (-1)
            lineType=cv2.LINE_AA,
        )

    out_path = f"data/{img_name}_similarity.png"
    cv2.imwrite(out_path, vis_img)
    print(f"saved similarity visualization with BBox → {out_path}")