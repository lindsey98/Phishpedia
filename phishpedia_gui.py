import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from phishpedia import PhishpediaWrapper
import cv2
from PIL import Image
import numpy as np

class PhishpediaGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.phishpedia_cls = PhishpediaWrapper()

    def initUI(self):
        self.setWindowTitle('Phishpedia GUI')
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()

        # URL Input
        url_layout = QHBoxLayout()
        self.url_label = QLabel('Enter URL:')
        url_layout.addWidget(self.url_label)
        self.url_input = QLineEdit()
        url_layout.addWidget(self.url_input)
        main_layout.addLayout(url_layout)

        # Image Upload
        image_layout = QHBoxLayout()
        self.image_label = QLabel('Upload Screenshot:')
        image_layout.addWidget(self.image_label)
        self.image_input = QLineEdit()
        image_layout.addWidget(self.image_input)
        self.image_button = QPushButton('Browse')
        self.image_button.clicked.connect(self.upload_image)
        image_layout.addWidget(self.image_button)
        main_layout.addLayout(image_layout)

        # Detect Button
        self.detect_button = QPushButton('Detect')
        self.detect_button.clicked.connect(self.detect_phishing)
        main_layout.addWidget(self.detect_button)

        # Result Display
        result_layout = QHBoxLayout()
        self.result_label = QLabel('Detection Result:')
        result_layout.addWidget(self.result_label)
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFixedHeight(100)  # 设置固定高度
        result_layout.addWidget(self.result_display)
        main_layout.addLayout(result_layout)

        # Visualization Display
        visualization_layout = QVBoxLayout()
        self.visualization_label = QLabel('Visualization Result:')
        visualization_layout.addWidget(self.visualization_label)
        self.visualization_display = QLabel()
        self.visualization_display.setAlignment(Qt.AlignCenter)
        self.visualization_display.setMinimumSize(600, 400)  # 设置最小大小
        visualization_layout.addWidget(self.visualization_display)
        main_layout.addLayout(visualization_layout)

        self.setLayout(main_layout)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Screenshot", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.image_input.setText(file_name)

    def detect_phishing(self):
        url = self.url_input.text()
        screenshot_path = self.image_input.text()

        if not url or not screenshot_path:
            self.result_display.setText("Please enter URL and upload a screenshot.")
            return

        phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time = self.phishpedia_cls.test_orig_phishpedia(url, screenshot_path, None)

        result_text = f"Phish Category(0 for benign, 1 for phish, default is benign): {phish_category}\n"
        result_text += f"Predicted Target: {pred_target}\n"
        result_text += f"Matched Domain: {matched_domain}\n"
        result_text += f"Siamese Confidence: {siamese_conf}\n"
        result_text += f"Logo Recognition Time: {logo_recog_time} seconds\n"
        result_text += f"Logo Match Time: {logo_match_time} seconds\n"

        self.result_display.setText(result_text)

        if phish_category == 1 and plotvis is not None:
            self.display_image(plotvis)
        if phish_category ==0:
            # 展示已经上传的屏幕截图
            self.display_image(plotvis)
    def display_image(self, plotvis):
        try:
            # Convert plotvis to QImage
            height, width, channel = plotvis.shape
            bytes_per_line = 3 * width
            plotvis_qimage = QImage(plotvis.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert QImage to QPixmap
            plotvis_pixmap = QPixmap.fromImage(plotvis_qimage)
            self.visualization_display.setPixmap(plotvis_pixmap.scaled(self.visualization_display.size(), Qt.KeepAspectRatio))
        except Exception as e:
            print(f"Error converting image: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PhishpediaGUI()
    ex.show()
    sys.exit(app.exec_())