import cv2
import numpy as np
import csv
import sys

# 미디어파이프 얼굴 랜드마크 인덱스 (468개)
LANDMARK_NUM = 468

# 영상 크기 (원본 영상 크기에 맞게 수정)
FRAME_W = 640
FRAME_H = 480

# 얼굴 랜드마크 시각화 색상
LM_COLOR = (0, 255, 0)

# CSV: [timestamp, wearable, x0, y0, z0, x1, y1, z1, ...]
def visualize_csv(csv_path, out_mp4_path, fps=30):
    frames = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2 + 3 * LANDMARK_NUM:
                continue
            # wearable = int(row[1])  # 필요시 활용
            coords = np.array(row[2:2+3*LANDMARK_NUM], dtype=np.float32).reshape(-1, 3)
            # 0~1 좌표를 프레임 크기로 변환
            pts = np.zeros((LANDMARK_NUM, 2), dtype=np.int32)
            pts[:, 0] = (coords[:, 0] * FRAME_W).astype(np.int32)
            pts[:, 1] = (coords[:, 1] * FRAME_H).astype(np.int32)
            frame = np.ones((FRAME_H, FRAME_W, 3), dtype=np.uint8) * 255
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 2, LM_COLOR, -1)
            frames.append(frame)
    # 영상 저장
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_mp4_path, fourcc, fps, (FRAME_W, FRAME_H))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"시각화 영상 저장 완료: {out_mp4_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python visualize_landmarks_csv.py <csv_path> <out_mp4_path>")
        exit(1)
    visualize_csv(sys.argv[1], sys.argv[2])
