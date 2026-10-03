#!/usr/bin/env python3
"""
optical_flow_scorer.py — 光流物理一致性检测
============================================
用 Farneback 光流检测视频中的物理异常（瞬移、突变、不连续运动）。

三路信号:
  1. flow_magnitude_zscore:  每帧光流幅值偏离视频均值的程度 → 突变检测
  2. flow_direction_change:  相邻帧光流方向突变 → 动作反转/瞬移
  3. flow_divergence:        光流散度异常 → 物体突然出现/消失

集成方式:
  与 V-JEPA + I-JEPA 分数融合，作为第三路物理信号

用法:
  from optical_flow_scorer import OpticalFlowScorer
  scorer = OpticalFlowScorer()
  scores = scorer.score_video(video_path, start_frame=0, end_frame=None)
"""
import cv2
import numpy as np


class OpticalFlowScorer:
    """基于光流的物理异常评分"""

    def __init__(self, scale=0.5):
        """scale: 降采样因子（加速计算）"""
        self.scale = scale
        self.prev_gray = None

    def score_video(self, video_path, start_frame=0, end_frame=None, step=1):
        """
        计算视频的逐帧光流异常分数。

        返回:
          scores: [N_frames] 每帧的综合光流异常分数 (0~1)
          details: dict with per-signal arrays
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total - 1
        end_frame = min(end_frame, total - 1)

        n_frames = (end_frame - start_frame) // step + 1
        if n_frames < 2:
            cap.release()
            return np.zeros(max(1, n_frames)), {}

        # Collect flow statistics
        flow_mags = []
        flow_dirs = []
        flow_divs = []
        frame_indices = []

        prev_gray = None
        for i, fi in enumerate(range(start_frame, end_frame + 1, step)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.scale < 1.0:
                gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                # 1. Flow magnitude (mean + std)
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_mags.append(mag.mean())

                # 2. Flow direction
                dirs = np.arctan2(flow[..., 1], flow[..., 0])
                # Mean direction via circular mean
                cos_mean = np.cos(dirs).mean()
                sin_mean = np.sin(dirs).mean()
                mean_dir = np.arctan2(sin_mean, cos_mean)
                flow_dirs.append(mean_dir)

                # 3. Flow divergence (approximate: sum of spatial gradients)
                fx = flow[..., 0]
                fy = flow[..., 1]
                # Simple divergence: dfx/dx + dfy/dy
                div_x = np.diff(fx, axis=1)
                div_y = np.diff(fy, axis=0)
                # Pad to match sizes
                div = np.abs(div_x[:, :-1]).mean() + np.abs(div_y[:-1, :]).mean()
                flow_divs.append(div)

                frame_indices.append(fi)
            else:
                # First frame: fill with zeros
                flow_mags.append(0.0)
                flow_dirs.append(0.0)
                flow_divs.append(0.0)
                frame_indices.append(fi)

            prev_gray = gray

        cap.release()

        if len(flow_mags) < 2:
            return np.zeros(len(flow_mags)), {}

        mag_arr = np.array(flow_mags)
        dir_arr = np.array(flow_dirs)
        div_arr = np.array(flow_divs)

        # 1. Magnitude z-score: sudden large movements
        mag_mean, mag_std = mag_arr.mean(), mag_arr.std() + 1e-6
        mag_z = np.abs(mag_arr - mag_mean) / mag_std

        # 2. Direction change: angular difference between consecutive frames
        dir_change = np.zeros_like(dir_arr)
        for i in range(1, len(dir_arr)):
            diff = np.abs(dir_arr[i] - dir_arr[i-1])
            # Wrap around ±π
            diff = np.minimum(diff, 2*np.pi - diff)
            dir_change[i] = diff / np.pi  # normalize to [0, 1]

        # 3. Divergence z-score
        div_mean, div_std = div_arr.mean(), div_arr.std() + 1e-6
        div_z = np.abs(div_arr - div_mean) / div_std

        # Combine: 0.5*magnitude + 0.3*direction + 0.2*divergence
        raw_scores = 0.5 * mag_z + 0.3 * dir_change + 0.2 * div_z

        # Sigmoid normalize
        raw_scores = raw_scores / (raw_scores.max() + 1e-8)

        details = {
            "flow_magnitude_z": mag_z.tolist(),
            "flow_direction_change": dir_change.tolist(),
            "flow_divergence_z": div_z.tolist(),
            "frame_indices": frame_indices,
        }

        return raw_scores, details


if __name__ == "__main__":
    scorer = OpticalFlowScorer(scale=0.5)
    scores, details = scorer.score_video(
        "/mnt/c/Users/admin/Desktop/ComfyUI_00013_.webm"
    )
    print(f"Flow scores: {scores.shape}, range=[{scores.min():.3f}, {scores.max():.3f}]")
    # Print first 10 scores
    for i in range(0, len(scores), 10):
        print(f"  Frame {i:3d}: {scores[i]:.3f}")
