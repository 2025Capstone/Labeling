from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget, QFileDialog, QSlider
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor
import cv2
import csv
import sys

class LabelBar(QWidget):
    def __init__(self, parent, label_data, total_frames, fps):
        super().__init__(parent)
        self.label_data = label_data
        self.total_frames = total_frames
        self.fps = fps
        self.current_time = 0
        self.start_marker_time = None
        self.setFixedHeight(10)

    def set_labels(self, label_data):
        self.label_data = label_data
        self.update()

    def set_current_time(self, time):
        self.current_time = time
        self.update()

    def set_start_marker(self, time):
        self.start_marker_time = time
        self.update()

    def clear_start_marker(self):
        self.start_marker_time = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()

        color_map = {
            1: QColor("green"),
            2: QColor("yellow"),
            3: QColor("orange"),
            4: QColor("red"),
            5: QColor("purple")
        }

        for start_sec, end_sec, label in self.label_data:
            start_frac = start_sec / (self.total_frames / self.fps)
            end_frac = end_sec / (self.total_frames / self.fps)
            x = int(start_frac * width)
            w = int((end_frac - start_frac) * width)
            painter.fillRect(x, 0, w, self.height(), color_map.get(label, QColor("gray")))

        if self.start_marker_time is not None:
            start_frac = self.start_marker_time / (self.total_frames / self.fps)
            x = int(start_frac * width)
            painter.setPen(QColor("black"))
            painter.drawLine(x, 0, x, self.height())

        pos_frac = self.current_time / (self.total_frames / self.fps)
        x = int(pos_frac * width)
        painter.setPen(QColor("blue"))
        painter.drawLine(x, 0, x, self.height())

class VideoLabeler(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Drowsiness Labeler")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_time = 0
        self.labels = []
        self.start_time = None
        self.ranged_labels = []
        self.ranged_labels.append((0.0, self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps, 1))

        # UI Elements
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.slider.setFixedWidth(640)
        self.slider.sliderReleased.connect(self.seek_video)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(60, 30)
        self.play_button.clicked.connect(self.toggle_playback)

        self.label_bar = LabelBar(self, self.ranged_labels, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.fps)
        self.label_bar.setFixedWidth(640)
        self.label_bar.set_labels(self.ranged_labels)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.play_button)

        self.start_button = QPushButton("Start Section")
        self.start_button.setFixedSize(100, 40)
        self.start_button.clicked.connect(self.set_start_time)

        label_range_layout = QHBoxLayout()
        label_range_layout.addWidget(self.start_button)
        color_map = {
            1: "green",
            2: "yellow",
            3: "orange",
            4: "red",
            5: "purple"
        }
        for i in range(1, 6):
            btn = QPushButton(f"End + Label {i}")
            btn.setFixedSize(100, 40)
            btn.setStyleSheet(f"background-color: {color_map[i]};")
            btn.clicked.connect(lambda checked, v=i: self.set_range_label(v))
            label_range_layout.addWidget(btn)

        self.save_button = QPushButton("Save Labels")
        self.save_button.clicked.connect(self.save_labels)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(control_layout)
        layout.addWidget(self.label_bar)
        layout.addLayout(label_range_layout)
        layout.addWidget(self.save_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)
        self.timer.start(int(1000 / self.fps))

    def play_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.label_bar.set_current_time(self.current_time)
            self.display_frame(frame)
            frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(frame_idx)

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))

    def label_frame(self, value):
        self.labels.append((self.current_time, value))
        print(f"Labeled: {value} at {self.current_time:.2f} sec")

    def save_labels(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if file_path:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["start_time_sec", "end_time_sec", "label"])
                for s, e, l in self.ranged_labels:
                    writer.writerow([s, e, l])
            print("Ranged labels saved.")

    def seek_video(self):
        frame_num = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.label_bar.set_current_time(self.current_time)
            self.display_frame(frame)

    def set_start_time(self):
        self.start_time = self.current_time
        self.label_bar.set_start_marker(self.start_time)
        print(f"Start time set at {self.start_time:.2f} sec")

    def set_range_label(self, label):
        if self.start_time is None:
            print("Start time not set.")
            return
        new_start = self.start_time
        new_end = self.current_time
        new_range = (new_start, new_end, label)
        updated_ranges = []
        for s, e, l in self.ranged_labels:
            # If the existing segment is completely before or after the new range, keep it intact
            if e <= new_start or s >= new_end:
                updated_ranges.append((s, e, l))
            else:
                # If the segment overlaps, split it into non-overlapping parts
                if s < new_start:
                    updated_ranges.append((s, new_start, l))
                if e > new_end:
                    updated_ranges.append((new_end, e, l))
        # Append the new labeled range
        updated_ranges.append(new_range)
        # Sort the segments by their start times
        updated_ranges.sort(key=lambda x: x[0])
        self.ranged_labels = updated_ranges
        print(f"Labeled range: {new_start:.2f} - {new_end:.2f} → {label}")
        self.start_time = None
        self.label_bar.clear_start_marker()
        self.label_bar.set_labels(self.ranged_labels)

    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("▶")
        else:
            self.timer.start(int(1000 / self.fps))
            self.play_button.setText("⏸")

def run_labeler():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
    if not file_path:
        print("No file selected.")
        return
    window = VideoLabeler(file_path)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_labeler()