"""
AppByteTrack —— 融合 Re-ID 外观特征的改进 ByteTrack 跟踪器
=============================================================
改进点：
  1. 每帧对高置信度检测框提取 OSNet 外观特征（128 维 L2 归一化）
  2. 每条轨迹用 EMA 维护外观特征，防止特征漂移
  3. 第一阶段匹配代价 = λ_iou × IoU代价 + λ_app × 外观余弦距离代价
  4. 第二阶段（低置信度框）仍用纯 IoU，与原版 ByteTrack 保持一致

依赖：
  pip install torchreid
  或
  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ── Re-ID 模型加载 ────────────────────────────────────────────────────────────
def _build_reid_model(device: str = "cuda"):
    try:
        import torchreid
        import os
        model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,   # ← 改为1000，和预训练权重一致
            pretrained=False,
        )
        weight_path = "/root/autodl-tmp/YOLO11-AnchorPedestrianTrack/models/osnet_x1_0_imagenet.pth"
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location=device, weights_only=False)
            model.load_state_dict(state, strict=True)   # ← 改为strict=True
            print(f"[AppByteTrack] OSNet-x1.0 加载成功（本地权重）")
        else:
            print(f"[AppByteTrack] ⚠️ 未找到权重: {weight_path}")
        model = model.to(device).eval()
        return model
    except Exception as e:
        print(f"[AppByteTrack] 加载失败: {e}")
        print("[AppByteTrack] 降级为纯 IoU 模式")
        return None


# ── 外观特征提取 ──────────────────────────────────────────────────────────────
_REID_MEAN = [0.485, 0.456, 0.406]
_REID_STD  = [0.229, 0.224, 0.225]

def extract_features(
    reid_model,
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    从原图中裁剪检测框区域，经 OSNet 提取 L2 归一化特征。

    Args:
        reid_model : OSNet 模型（None 时返回全零特征）
        frame      : BGR 原图 (H, W, 3)
        boxes_xyxy : shape (N, 4)，每行为 [x1, y1, x2, y2]
        device     : "cuda" or "cpu"

    Returns:
        features: shape (N, 512)，L2 归一化
    """
    N = len(boxes_xyxy)
    feat_dim = 512

    if reid_model is None or N == 0:
        return np.zeros((N, feat_dim), dtype=np.float32)

    H, W = frame.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes_xyxy.astype(int):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            crops.append(np.zeros((256, 128, 3), dtype=np.uint8))
            continue
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 256))          # OSNet 标准输入尺寸
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(crop)

    # 预处理：归一化
    imgs = []
    for crop in crops:
        img = crop.astype(np.float32) / 255.0
        img = (img - _REID_MEAN) / _REID_STD
        img = img.transpose(2, 0, 1)               # HWC -> CHW
        imgs.append(img)

    batch = torch.tensor(np.stack(imgs), dtype=torch.float32).to(device)

    with torch.no_grad():
        feats = reid_model.featuremaps(batch)   # 只取特征，不经过分类头
        feats = feats.mean(dim=[2, 3])          # 全局平均池化
        feats = F.normalize(feats, p=2, dim=1)

    return feats.cpu().numpy()


# ── 代价矩阵计算 ──────────────────────────────────────────────────────────────
def iou_cost(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    计算 IoU 代价矩阵（1 - IoU），shape (M, N)。
    boxes 格式：xyxy
    """
    M, N = len(boxes_a), len(boxes_b)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)

    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter + 1e-7

    iou = inter / union
    return 1.0 - iou


def appearance_cost(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """
    计算外观余弦距离代价矩阵，shape (M, N)。
    输入已经是 L2 归一化特征，余弦距离 = 1 - 点积。
    """
    if feats_a.shape[0] == 0 or feats_b.shape[0] == 0:
        return np.zeros((len(feats_a), len(feats_b)), dtype=np.float32)
    sim = feats_a @ feats_b.T                       # (M, N)，值域 [-1, 1]
    return (1.0 - sim).clip(0, 2) / 2.0             # 归一化到 [0, 1]


def fuse_cost(
    cost_iou: np.ndarray,
    cost_app: np.ndarray,
    lambda_iou: float = 0.5,
    lambda_app: float = 0.5,
) -> np.ndarray:
    """融合 IoU 和外观代价"""
    return lambda_iou * cost_iou + lambda_app * cost_app


# ── 匈牙利匹配 ────────────────────────────────────────────────────────────────
def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """
    用 scipy 线性分配求最优匹配，过滤超过阈值的匹配。
    返回 (matched_pairs, unmatched_rows, unmatched_cols)
    """
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched, unmatched_r, unmatched_c = [], [], []

    valid = cost_matrix[row_ind, col_ind] <= thresh
    matched = np.stack([row_ind[valid], col_ind[valid]], axis=1)

    matched_r = set(row_ind[valid])
    matched_c = set(col_ind[valid])
    unmatched_r = [i for i in range(cost_matrix.shape[0]) if i not in matched_r]
    unmatched_c = [j for j in range(cost_matrix.shape[1]) if j not in matched_c]

    return matched, unmatched_r, unmatched_c


# ── 轨迹类 ────────────────────────────────────────────────────────────────────
class Track:
    """单条轨迹，维护状态和 EMA 外观特征"""

    _id_counter = 0

    def __init__(self, box_xyxy: np.ndarray, score: float, feat: np.ndarray):
        Track._id_counter += 1
        self.track_id   = Track._id_counter
        self.box        = box_xyxy.copy()           # [x1, y1, x2, y2]
        self.score      = score
        self.feat       = feat.copy()               # EMA 特征
        self.hits       = 1
        self.age        = 0                         # 未匹配帧数
        self.state      = "tracked"                 # tracked / lost

    def update(self, box_xyxy: np.ndarray, score: float,
               feat: np.ndarray, alpha: float = 0.9):
        """匹配成功后更新轨迹"""
        self.box   = box_xyxy.copy()
        self.score = score
        self.age   = 0
        self.hits += 1
        self.state = "tracked"
        # EMA 特征更新
        if np.any(feat != 0):
            self.feat = alpha * self.feat + (1 - alpha) * feat
            norm = np.linalg.norm(self.feat)
            if norm > 1e-6:
                self.feat /= norm

    def mark_lost(self):
        self.age  += 1
        self.state = "lost"


# ── AppByteTrack 主类 ─────────────────────────────────────────────────────────
class AppByteTracker:
    """
    改进的 ByteTrack，融合 Re-ID 外观特征。

    参数：
        reid_model    : OSNet 模型（None 时退化为纯 IoU ByteTrack）
        high_thresh   : 高置信度检测框阈值
        low_thresh    : 低置信度检测框阈值
        match_thresh  : 第一阶段匹配阈值
        iou_thresh2   : 第二阶段（纯 IoU）匹配阈值
        max_lost      : 轨迹最大丢失帧数
        lambda_iou    : IoU 代价权重
        lambda_app    : 外观代价权重
        ema_alpha     : EMA 平滑系数
        device        : 推理设备
    """

    def __init__(
        self,
        reid_model=None,
        high_thresh: float = 0.5,
        low_thresh:  float = 0.1,
        match_thresh: float = 0.8,
        iou_thresh2:  float = 0.5,
        max_lost:     int   = 30,
        lambda_iou:   float = 0.5,
        lambda_app:   float = 0.5,
        ema_alpha:    float = 0.9,
        device:       str   = "cuda",
    ):
        self.reid_model   = reid_model
        self.high_thresh  = high_thresh
        self.low_thresh   = low_thresh
        self.match_thresh = match_thresh
        self.iou_thresh2  = iou_thresh2
        self.max_lost     = max_lost
        self.lambda_iou   = lambda_iou
        self.lambda_app   = lambda_app
        self.ema_alpha    = ema_alpha
        self.device       = device

        self.tracked_tracks: List[Track] = []
        self.lost_tracks:    List[Track] = []
        Track._id_counter = 0             # 每次新建 tracker 重置 ID

    def reset(self):
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        Track._id_counter = 0

    def update(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,   # (N, 4)
        scores: np.ndarray,       # (N,)
    ) -> List[Track]:
        """
        处理一帧，返回当前所有 tracked 轨迹。
        """
        # ── 1. 按置信度分高低两组 ──────────────────────────────────────────
        high_mask = scores >= self.high_thresh
        low_mask  = (scores >= self.low_thresh) & (~high_mask)

        high_boxes  = boxes_xyxy[high_mask]
        high_scores = scores[high_mask]
        low_boxes   = boxes_xyxy[low_mask]
        low_scores  = scores[low_mask]

        # ── 2. 提取高置信度框的外观特征 ───────────────────────────────────
        high_feats = extract_features(
            self.reid_model, frame, high_boxes, self.device
        ) if len(high_boxes) > 0 else np.zeros((0, 512), dtype=np.float32)

        # ── 3. 第一阶段匹配：active轨迹 vs 高置信度检测框 ─────────────────
        active = [t for t in self.tracked_tracks if t.state == "tracked"]
        active += self.lost_tracks                   # lost 轨迹也参与第一阶段

        matched1, unmatched_trk1, unmatched_det1 = [], list(range(len(active))), list(range(len(high_boxes)))

        if len(active) > 0 and len(high_boxes) > 0:
            trk_boxes = np.stack([t.box for t in active])
            trk_feats = np.stack([t.feat for t in active])

            c_iou = iou_cost(trk_boxes, high_boxes)
            c_app = appearance_cost(trk_feats, high_feats)
            cost  = fuse_cost(c_iou, c_app, self.lambda_iou, self.lambda_app)

            matched1, unmatched_trk1, unmatched_det1 = linear_assignment(
                cost, self.match_thresh
            )

        # 更新匹配上的轨迹
        for ti, di in matched1:
            active[ti].update(
                high_boxes[di], high_scores[di],
                high_feats[di], self.ema_alpha
            )

        # ── 4. 第二阶段匹配：未匹配轨迹 vs 低置信度检测框（纯 IoU）─────────
        remain_trk = [active[i] for i in unmatched_trk1 if active[i].state == "tracked"]
        matched2, unmatched_trk2, unmatched_det2 = [], list(range(len(remain_trk))), list(range(len(low_boxes)))

        if len(remain_trk) > 0 and len(low_boxes) > 0:
            trk_boxes2 = np.stack([t.box for t in remain_trk])
            c_iou2 = iou_cost(trk_boxes2, low_boxes)
            matched2, unmatched_trk2, unmatched_det2 = linear_assignment(
                c_iou2, 1.0 - self.iou_thresh2
            )

        for ti, di in matched2:
            feat_zero = np.zeros(512, dtype=np.float32)
            remain_trk[ti].update(low_boxes[di], low_scores[di], feat_zero, self.ema_alpha)

        # ── 5. 标记丢失轨迹 ──────────────────────────────────────────────
        still_unmatched = set(unmatched_trk1)
        for t in active:
            if id(t) in {id(active[i]) for i in still_unmatched}:
                t.mark_lost()

        # ── 6. 新建轨迹（第一阶段未匹配的高置信度检测框）────────────────
        for di in unmatched_det1:
            new_t = Track(high_boxes[di], high_scores[di], high_feats[di])
            self.tracked_tracks.append(new_t)

        # ── 7. 移除超时丢失轨迹 ──────────────────────────────────────────
        self.lost_tracks   = [t for t in self.tracked_tracks if t.state == "lost" and t.age <= self.max_lost]
        self.tracked_tracks= [t for t in self.tracked_tracks if t.state == "tracked"]
        # 合并 lost 回主列表（用于下一帧参与匹配）
        self.tracked_tracks += self.lost_tracks

        return [t for t in self.tracked_tracks if t.state == "tracked"]


# ── 集成到 PedestrianTracker ──────────────────────────────────────────────────
class AppPedestrianTracker:
    """
    基于 AppByteTrack 的行人跟踪器，接口与原版 PedestrianTracker 完全兼容。

    使用方法（替换 tracker.py 中的 PedestrianTracker）：
        from app_bytetrack import AppPedestrianTracker as PedestrianTracker
    """

    def __init__(
        self,
        model_path: str,
        use_reid: bool = True,
        lambda_iou: float = 0.5,
        lambda_app: float = 0.5,
        high_thresh: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.det_model = YOLO(model_path)
        self.conf_threshold = 0.25
        self.iou_threshold  = 0.5
        self.device = device

        # Re-ID 模型
        reid_model = _build_reid_model(device) if use_reid else None

        # AppByteTrack 跟踪器
        self.tracker = AppByteTracker(
            reid_model=reid_model,
            high_thresh=high_thresh,
            lambda_iou=lambda_iou,
            lambda_app=lambda_app,
            device=device,
        )

        self.prev_time = time.time()
        self.fps_ema   = 0.0
        self._lock     = Lock()

        print(f"[AppPedestrianTracker] 初始化完成")
        print(f"  检测模型: {model_path}")
        print(f"  Re-ID:   {'启用 (OSNet-x1.0)' if reid_model else '禁用（纯IoU模式）'}")
        print(f"  代价权重: IoU={lambda_iou}, App={lambda_app}")

    def reset_tracking_state(self):
        self.tracker.reset()
        self.prev_time = time.time()
        self.fps_ema   = 0.0

    def _update_fps(self) -> float:
        now = time.time()
        instant = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        if self.fps_ema == 0.0:
            self.fps_ema = instant
        else:
            self.fps_ema = 0.9 * self.fps_ema + 0.1 * instant
        return self.fps_ema

    @staticmethod
    def _draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """在帧上绘制跟踪结果"""
        out = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.box.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{t.track_id}"
            cv2.putText(out, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out

    @staticmethod
    def _overlay_text(frame, count: int, fps: float | None = None):
        cv2.putText(frame, f"Person Count: {count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def _detect(self, frame: np.ndarray):
        """运行 YOLO 检测，返回 boxes_xyxy 和 scores"""
        with self._lock:
            results = self.det_model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                verbose=False,
            )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        xyxy   = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        return xyxy, scores

    def process_tracked_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧，返回带注释的帧（与原版接口一致）"""
        if frame is None:
            return None

        boxes_xyxy, scores = self._detect(frame)
        tracks = self.tracker.update(frame, boxes_xyxy, scores)

        annotated = self._draw_tracks(frame, tracks)
        fps = self._update_fps()
        self._overlay_text(annotated, len(tracks), fps)
        return annotated

    def process_image(self, frame: np.ndarray):
        """处理单张图片（不使用跟踪，只检测）"""
        if frame is None:
            return None, 0
        with self._lock:
            results = self.det_model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                verbose=False,
            )
        result = results[0]
        annotated = result.plot()
        count = len(result.boxes) if result.boxes is not None else 0
        self._overlay_text(annotated, count, fps=None)
        return annotated, count

    def process_video_file(self, input_path: Path, output_path: Path) -> dict:
        """处理视频文件"""
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {input_path}")

        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        in_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = in_fps if in_fps and in_fps > 1 else 25.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (W, H))
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Cannot open writer: {output_path}")

        self.reset_tracking_state()
        start, frames = time.time(), 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                processed = self.process_tracked_frame(frame)
                writer.write(processed)
                frames += 1
        finally:
            cap.release()
            writer.release()

        elapsed = max(time.time() - start, 1e-6)
        return {
            "frames": frames,
            "elapsed_sec": round(elapsed, 2),
            "avg_fps": round(frames / elapsed, 2),
            "output_path": str(output_path),
        }

    # 兼容旧接口
    def process_frame(self, frame):
        return self.process_tracked_frame(frame)