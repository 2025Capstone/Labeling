# 얼굴 랜드마크 라벨링 툴 프로젝트

이 프로젝트는 웹캠 영상에서 얼굴 랜드마크를 추출·저장하고, 졸음 구간 라벨링 및 데이터 시각화/복호화를 지원하는 통합 툴입니다.

---

## 📁 주요 파일 설명

- **record_video.py** : 웹캠 영상 녹화 및 얼굴 랜드마크 추출/저장 (CSV+압축)
- **labeler.py** : 녹화된 영상과 랜드마크 데이터 기반 졸음 구간 라벨링 GUI
- **decode_zstd_csv.py** : 압축된 zstd CSV(.csv.zst)를 원본 CSV로 복호화
- **visualize_landmarks_csv.py** : 저장된 랜드마크 CSV를 30fps mp4로 시각화
- **requirements.txt** : 필요한 파이썬 패키지 목록

---

## ⚡ 커맨드라인 사용법

### 1. 얼굴 데이터 녹화 및 저장

```bash
python record_video.py
```
- GUI 창에서 녹화 버튼 클릭 → 영상(mp4)과 랜드마크(csv) 자동 저장
- 저장 경로: `data/videoN.mp4`, `data/labelN.csv.zst` (압축됨)

### 2. 졸음 구간 라벨링 (라벨링 GUI)

```bash
python labeler.py
```
- GUI 창에서 녹화 버튼 클릭 → 영상(mp4)과 랜드마크(csv) 자동 저장
- 구간별로 졸음 정도(1~5) 라벨 지정 및 저장

### 3. 압축된 CSV 복호화

```bash
python decode_zstd_csv.py data/label1.csv.zst data/label1_wearable.csv data/video1_drowsiness_label.csv data/label1_merged.csv
```
- `data/label1.csv.zst` → `data/label1.csv`로 복원

### 4. 랜드마크 CSV 시각화(mp4 변환)

```bash
python visualize_landmarks_csv.py data/label1.csv data/landmarks_vis.mp4
```
- CSV 기반 랜드마크 점을 30fps mp4로 시각화

---

## Conda 환경에서 실행하기

### 1. Conda 환경 생성 및 활성화

```bash
conda create -n labeling python=3.10
conda activate labeling
```

> **주의:**  
> mediapipe는 Python 3.10 이하에서만 안정적으로 동작합니다.  
> Python 3.11 이상에서는 오류가 발생할 수 있습니다.

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 프로그램 실행

```bash
python record_video.py
```

- 실행 후, GUI 창에서 녹화 버튼을 누르면 웹캠 영상과 얼굴 랜드마크가 실시간으로 기록됩니다.
- 녹화가 끝나면 `data/` 폴더에 영상(mp4)과 랜드마크(csv)가 저장됩니다.

### 4. 저장되는 데이터

- CSV 파일에는 타임스탬프, 웨어러블 상태, 478개 랜드마크의 x, y, z 좌표가 저장됩니다.

---

## 오류가 발생할 때

- Python 및 패키지 버전이 권장 사양과 일치하는지 확인하세요.
- 웹캠이 정상적으로 연결되어 있는지 확인하세요.
