import sys
import os
import cv2
import csv
import datetime
import numpy as np
import mediapipe as mp
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
import zstandard as zstd
import shutil
import firebase_admin
from firebase_admin import credentials, db
import string, random


class DrowsinessApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam & Drowsiness Recorder")
        self.start_time = None
        self.last_logged_second = -0.5
        
        
        # … 기존 UI/타이머 세팅 …
        self.user_id = "123456"
        self.root_ref = db.reference(self.user_id)
        self.pair_code = None
        self.pair_timer = None
        self.recording = False
        self.is_pairing = False     # ← 페어링 중인지를 나타내는 플래그

        # 웹캠 연결 (카메라 인덱스 1)
        self.cap = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # 왼쪽: 영상 표시 영역 (QLabel 사용)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # 녹화 시간 표시용 라벨
        self.time_label = QLabel("00:00.0")
        self.time_label.setStyleSheet("font-size: 18pt; color: red; background-color: rgba(255,255,255,128);")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.time_label.setFixedWidth(200)

        # 오른쪽: 정보 표시 영역
        self.face_label = QLabel("Face Detection: normal")
        self.face_label.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 4px;")
        self.record_time_label = QLabel("")
        self.record_time_label.setStyleSheet("font-size: 18pt; color: #1976d2; background-color: #f0f0f0; border: 1px solid #b0b0b0; border-radius: 8px; padding: 6px 16px; margin-top: 8px; font-weight: bold;")
        self.record_time_label.setVisible(False)
        self.wearable_label = QLabel("Wearable: 0")
        self.wearable_label.setStyleSheet("font-size: 14pt;")
        
        # 페어링 상태 표시
        self.pairing_status_label = QLabel("")
        self.pairing_status_label.setStyleSheet("font-size: 14pt; font-style: italic; color: gray;")


        # 녹화 시작/종료 버튼 (토글)
        self.record_button = QPushButton("▶ Start Recording")
        self.record_button.setStyleSheet("font-size: 20pt; background-color: green; color: white;")
        self.record_button.setFixedSize(300, 80)
        self.record_button.clicked.connect(self.toggle_record)

        # 오른쪽 레이아웃 구성
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.face_label)
        info_layout.addWidget(self.wearable_label)
        info_layout.addWidget(self.pairing_status_label)
        info_layout.addSpacing(20)
        info_layout.addWidget(self.record_button)
        info_layout.addWidget(self.record_time_label)
        info_layout.addStretch(1)

        info_widget = QWidget()
        info_widget.setLayout(info_layout)

        # 메인 레이아웃: 왼쪽 영상, 오른쪽 정보 영역
        # 영상 라벨만 배치 (녹화 시간은 오른쪽 정보 영역에 표시)
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_widget = QWidget()
        video_widget.setLayout(video_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(video_widget)
        main_layout.addWidget(info_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 녹화 관련 변수 초기화
        self.recording = False
        self.video_writer = None
        self.log_data = []  # 각 프레임의 타임스탬프와 상태 정보 저장
        self.partial_saved = False  # partial 저장 여부

        # 기본 프레임 사이즈 (업데이트 시 실제 값으로 변경)
        self.resized_w = 640
        self.resized_h = 480

        # 타이머 설정 (15ms 간격으로 영상 업데이트)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)
    
    def init_pairing(self):
        self.is_pairing = True      # ← 페어링 시작
        # 2) 매번 새로운 6자리 코드 생성
        self.pair_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        # 3) /<UID>/pairing/pair_code 에 쓰기 (덮어쓰기)
        self.root_ref.child("pairing").set({
            "pair_code": self.pair_code,
            "stop": False,
            "paired": False
        })   
        # 4) 대기 메시지 표시
        self.record_button.setText("■ Cancle Pairing")
        self.record_button.setStyleSheet("font-size: 20pt; background-color: red; color: white;")
        
        self.pairing_status_label.setText(f"Waiting Pairing...  (Code: {self.pair_code})")
        # 5) PPG_Data 생길 때까지 1초마다 체크
        self.pair_timer = QTimer(self)
        self.pair_timer.timeout.connect(self.check_paired)
        self.pair_timer.start(1000)
        
    def cancel_pairing(self):
        self.is_pairing = False     # ← 페어링 플래그 리셋
        # 페어링 취소: Firebase 노드 삭제
        self.root_ref.child("pairing").child("pair_code").delete()
        self.root_ref.child("pairing").update({
            "stop": True,
            "paired": False
        })
        # 타이머 중지, 메시지 초기화
        if self.pair_timer:
            self.pair_timer.stop()
        self.pairing_status_label.setText("")
        # 버튼 원래 상태로 복원
        self.record_button.setText("▶ Start Recording")
        self.record_button.setStyleSheet("font-size: 20pt; background-color: green; color: white;")
        
    def check_paired(self):
        data = self.root_ref.child("pairing").get()
        print("Pairing: ", data)
        if data:
            paired = data.get("paired", False)
            stop   = data.get("stop",   True)
            if paired and not stop:
                self.pair_timer.stop()
                self.start_recording()     # 녹화 시작
                self.pairing_status_label.setText("")

    def save_log_data_partial(self, write_header=False):
        """
        log_data를 청크 csv로 분할 저장 (labelN_chunks/chunk_XXXX.csv)
        """
        import uuid
        import glob
        chunk_dir = self.label_filename.replace('.csv', '_chunks')
        os.makedirs(chunk_dir, exist_ok=True)
        # 청크 인덱스 결정
        chunk_files = glob.glob(os.path.join(chunk_dir, 'chunk_*.csv'))
        next_idx = len(chunk_files) + 1
        chunk_path = os.path.join(chunk_dir, f'chunk_{next_idx:04d}.csv')
        with open(chunk_path, 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header or next_idx == 1:
                header = ["Timestamp", "Wearable_Status"] + [f"landmark_{i}_{c}" for i in range(478) for c in ('x', 'y', 'z')]
                writer.writerow(header)
            writer.writerows(self.log_data)
        self.log_data = []
        self.partial_saved = True


    def toggle_record(self):
        if not self.recording:
            if self.is_pairing:
                # 페어링 중이면 즉시 취소
                self.cancel_pairing()
            else:
                # 아니면 페어링 시작
                self.init_pairing()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.record_button.setText("■ Stop Recording")
        self.record_button.setStyleSheet("font-size: 20pt; background-color: red; color: white;")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs("data", exist_ok=True)
        existing_files = [f for f in os.listdir("data") if f.startswith("video") and f.endswith(".mp4")]
        indices = [int(f[5:-4]) for f in existing_files if f[5:-4].isdigit()]
        next_index = max(indices) + 1 if indices else 1
        self.video_filename = f"data/video{next_index}.mp4"
        self.label_filename = f"data/label{next_index}.csv"
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps == 0 or self.actual_fps != self.actual_fps:  # NaN 또는 0 방지
            self.actual_fps = 30.0  # 기본값 지정
        self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, self.actual_fps, (self.resized_w, self.resized_h))
        self.log_data = []
        self.partial_saved = False
        self.start_time = datetime.datetime.now()
        self.last_logged_second = -0.5
        # 프레임 인덱스 초기화
        self.frame_idx = 0
        # 랜드마크 프레임 샘플링 주기 변수 초기화 (프레임 카운터 기반)
        self.frame_idx = 0
        self.save_interval = int(round(self.actual_fps / 30.0)) if self.actual_fps > 0 else 1
        if self.save_interval < 1:
            self.save_interval = 1
        # 기존 csv 파일이 있으면 삭제(덮어쓰기)
        if os.path.exists(self.label_filename):
            os.remove(self.label_filename)


    def stop_recording(self):
        self.recording = False
        self.record_button.setText("▶ Start Recording")
        self.record_button.setStyleSheet("font-size: 20pt; background-color: green; color: white;")
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        # 남은 로그 데이터 저장
        if self.log_data:
            self.save_log_data_partial(write_header=(not self.partial_saved))
            
        # ① 웨어러블 측정 중지 신호 보내기
        self.root_ref.child("pairing").update({
            "stop": True,
            "paired": False
        })
        self.root_ref.child("pairing").child("pair_code").delete()
        
        # 백그라운드에서 압축 시작
        chunk_dir = self.label_filename.replace('.csv', '_chunks')
        zst_path = self.label_filename + '.zst'
        self.compress_thread = ChunkCompressorThread(chunk_dir, zst_path)
        self.compress_thread.finished_signal.connect(self.on_compress_finished)
        self.compress_thread.start()
        # 프레임 인덱스 초기화
        self.frame_idx = 0
        print("녹화 종료 및 로그 저장 및 압축 시작")

    def on_compress_finished(self, success, msg):
        if success:
            print(f"CSV 청크 압축 완료 및 정리: {msg}")
        else:
            print(f"CSV 청크 압축/정리 오류: {msg}")


    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("웹캠으로부터 프레임을 가져오지 못했습니다.")
            return

        # 프레임 크기를 최대 960x720에 맞게 조정 (종횡비 유지)
        max_width = 960
        max_height = 720
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h)
        self.resized_w = int(w * scale)
        self.resized_h = int(h * scale)
        frame = cv2.resize(frame, (self.resized_w, self.resized_h), interpolation=cv2.INTER_AREA)

        face_status = "no_face"
        wearable_status = 0

        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        flat_landmarks = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark[:478]
            flat_landmarks = []
            for lm in landmarks:
                flat_landmarks.extend([lm.x, lm.y, lm.z])
        # UI 업데이트
        self.face_label.setText(f"Face Detection: {'detected' if flat_landmarks else 'no_face'}")
        self.wearable_label.setText(f"Wearable: {wearable_status}")

        # 녹화 시간 표시 (face_label 아래)
        if self.recording:
            elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            self.record_time_label.setText(f"⏺ REC  {minutes:02d}:{seconds:04.1f}")
            self.record_time_label.setVisible(True)
        else:
            self.record_time_label.setText("")
            self.record_time_label.setVisible(False)

        # 프레임을 정상적으로 획득한 경우에만 영상 저장
        if self.recording and ret:
            # 영상 프레임은 모든 프레임마다 저장
            if self.video_writer is not None:
                self.video_writer.write(frame)
            # 랜드마크 row는 1초에 30개씩만 저장 (프레임 카운터 기반)
            if self.frame_idx % self.save_interval == 0:
                timestamp = self.frame_idx / self.actual_fps
                if flat_landmarks is not None:
                    row = [round(timestamp, 4), wearable_status] + flat_landmarks
                else:
                    row = [round(timestamp, 4), wearable_status] + [None]*1434
                self.log_data.append(row)
                # 500개마다 partial 저장
                if len(self.log_data) >= 500:
                    self.save_log_data_partial(write_header=(not self.partial_saved))
            self.frame_idx += 1

        # BGR 이미지를 RGB로 변환 후 QImage로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, self.resized_w, self.resized_h, 3 * self.resized_w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        # 프로그램 종료 시 자원 해제
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        event.accept()

class ChunkCompressorThread(QThread):
    finished_signal = Signal(bool, str)  # (성공여부, 메시지)
    def __init__(self, chunk_dir, zst_path):
        super().__init__()
        self.chunk_dir = chunk_dir
        self.zst_path = zst_path
    def run(self):
        import glob
        import csv
        import shutil
        try:
            chunk_files = sorted(glob.glob(os.path.join(self.chunk_dir, 'chunk_*.csv')))
            if not chunk_files:
                self.finished_signal.emit(False, 'No chunk files found')
                return
            cctx = zstd.ZstdCompressor(level=10)
            with open(self.zst_path, 'wb') as f_out:
                with cctx.stream_writer(f_out) as compressor:
                    for i, chunk_file in enumerate(chunk_files):
                        with open(chunk_file, 'rb') as f_in:
                            shutil.copyfileobj(f_in, compressor)
            # 모든 청크 삭제
            for chunk_file in chunk_files:
                os.remove(chunk_file)
            os.rmdir(self.chunk_dir)
            self.finished_signal.emit(True, self.zst_path)
        except Exception as e:
            self.finished_signal.emit(False, str(e))

if __name__ == "__main__":
    cred = credentials.Certificate("hrvdataset-firebase-adminsdk-oof96-2a96d6ac7f.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://hrvdataset-default-rtdb.firebaseio.com/'
    })
    
    app = QApplication(sys.argv)
    window = DrowsinessApp()
    window.show()
    sys.exit(app.exec())