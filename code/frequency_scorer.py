#!/usr/bin/env python3
"""
frequency_scorer.py — 频域异常检测
====================================
AI 生成视频常有规律性伪影（重复纹理、网格噪声），
在 DCT 频谱上表现为异常峰值，破坏自然视频的 1/f 分布。

信号:
  1. high_freq_energy:    高频能量占比 → AI 伪影通常在此异常
  2. spectral_peakiness:  频谱峰值尖锐度 → 自然视频平滑，AI 可能有孤立尖峰
  3. temporal_flicker:    相邻帧频谱突变 → 闪烁/跳变

用法:
  from frequency_scorer import FrequencyScorer
  scorer = FrequencyScorer()
  scores = scorer.score_video(video_path)
"""
import cv2
import numpy as np


class FrequencyScorer:
    """基于 DCT 频域分析的异常评分"""

    def __init__(self, scale=0.5, block_size=8):
        self.scale = scale
        self.block_size = block_size

    def _frame_spectral_features(self, gray):
        """计算单帧的频谱特征"""
        h, w = gray.shape
        bs = self.block_size

        # Block-wise DCT
        high_energy = 0.0
        total_energy = 0.0
        peakiness = 0.0

        n_blocks = 0
        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block = gray[y:y+bs, x:x+bs].astype(np.float32)
                block -= block.mean()  # remove DC
                dct = cv2.dct(block)

                # Energy
                energy = (dct ** 2).sum()
                total_energy += energy

                # High frequency energy (右下角 3x3)
                high_energy += (dct[-3:, -3:] ** 2).sum()

                # Peakiness: max / mean ratio
                dct_abs = np.abs(dct)
                peakiness += dct_abs.max() / (dct_abs.mean() + 1e-6)

                n_blocks += 1

        if n_blocks == 0:
            return 0.0, 0.0

        high_ratio = high_energy / (total_energy + 1e-6) if total_energy > 0 else 0
        avg_peakiness = peakiness / n_blocks

        return high_ratio, avg_peakiness

    def score_video(self, video_path, start_frame=0, end_frame=None, step=2):
        """
        返回:
          scores: [N_frames] 逐帧频域异常分数
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total - 1
        end_frame = min(end_frame, total - 1)

        high_ratios = []
        peakinesses = []
        frame_indices = []

        prev_high = None
        flicker = []

        for fi in range(start_frame, end_frame + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.scale < 1.0:
                gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale)

            hr, pk = self._frame_spectral_features(gray)
            high_ratios.append(hr)
            peakinesses.append(pk)
            frame_indices.append(fi)

            if prev_high is not None:
                flicker.append(abs(hr - prev_high))
            else:
                flicker.append(0.0)
            prev_high = hr

        cap.release()

        if len(high_ratios) < 2:
            return np.zeros(len(high_ratios)), {}

        hr_arr = np.array(high_ratios)
        pk_arr = np.array(peakinesses)
        fl_arr = np.array(flicker)

        # 1. High-freq z-score
        hr_z = np.abs(hr_arr - hr_arr.mean()) / (hr_arr.std() + 1e-6)

        # 2. Peakiness z-score
        pk_z = np.abs(pk_arr - pk_arr.mean()) / (pk_arr.std() + 1e-6)

        # 3. Flicker (temporal spectral change)
        fl_z = np.zeros_like(fl_arr)
        if fl_arr[1:].std() > 0:
            fl_z = np.abs(fl_arr - fl_arr.mean()) / (fl_arr.std() + 1e-6)

        # Combine: 0.4*high_freq + 0.3*peakiness + 0.3*flicker
        raw_scores = 0.4 * hr_z + 0.3 * pk_z + 0.3 * fl_z
        raw_scores = raw_scores / (raw_scores.max() + 1e-8)

        return raw_scores, {
            "high_freq_z": hr_z.tolist(),
            "peakiness_z": pk_z.tolist(),
            "flicker_z": fl_z.tolist(),
            "frame_indices": frame_indices,
        }


if __name__ == "__main__":
    scorer = FrequencyScorer(scale=0.5, block_size=8)
    scores, details = scorer.score_video(
        "/mnt/c/Users/admin/Desktop/ComfyUI_00013_.webm",
        step=2
    )
    print(f"Freq scores: {scores.shape}, range=[{scores.min():.3f}, {scores.max():.3f}]")
    for i in range(0, len(scores), 10):
        print(f"  Frame ~{i*2:3d}: {scores[i]:.3f}")
