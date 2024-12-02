from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from phishpedia import PhishpediaWrapper
import cv2


class PhishpediaFunction:
    def __init__(self, ui):
        self.ui = ui
        self.phishpedia_cls = PhishpediaWrapper()
        self.current_pixmap = None

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self.ui, "Select Screenshot", "", "Images (*.png *.jpg *.jpeg)",
                                                   options=options)
        if file_name:
            self.ui.image_input.setText(file_name)

    def detect_phishing(self):
        url = self.ui.url_input.text()
        screenshot_path = self.ui.image_input.text()

        if not url or not screenshot_path:
            self.ui.result_display.setText("Please enter URL and upload a screenshot.")
            return

        phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time = self.phishpedia_cls.test_orig_phishpedia(
            url, screenshot_path, None)

        # 根据 phish_category 改变颜色
        phish_category_color = 'green' if phish_category == 0 else 'red'
        result_text = f'<span style="color: {phish_category_color};">Phish Category(0 for benign, 1 for phish, default is benign): {phish_category}</span><br>'
        result_text += f"Predicted Target: {pred_target}<br>"
        result_text += f"Matched Domain: {matched_domain}<br>"
        result_text += f"Siamese Confidence: {siamese_conf}<br>"
        result_text += f"Logo Recognition Time: {logo_recog_time} seconds<br>"
        result_text += f"Logo Match Time: {logo_match_time} seconds<br>"

        self.ui.result_display.setText(result_text)

        if phish_category == 1 and plotvis is not None:
            self.display_image(plotvis)
        if phish_category == 0:
            self.display_image(plotvis)

    def display_image(self, plotvis):
        try:
            # Convert BGR to RGB
            plotvis_rgb = cv2.cvtColor(plotvis, cv2.COLOR_BGR2RGB)
            height, width, channel = plotvis_rgb.shape
            bytes_per_line = 3 * width
            plotvis_qimage = QImage(plotvis_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            self.current_pixmap = QPixmap.fromImage(plotvis_qimage)
            self.update_image_display()
        except Exception as e:
            print(f"Error converting image: {e}")

    def update_image_display(self):
        if self.current_pixmap:
            # Get the actual size of the visualization_display
            display_height = self.ui.visualization_display.height()
            display_width = self.ui.visualization_display.width()
            # Get the original dimensions of the image
            original_width = self.current_pixmap.width()
            original_height = self.current_pixmap.height()
            # Calculate the scaling ratio
            width_ratio = display_width / original_width
            height_ratio = display_height / original_height
            # Use the smaller ratio to ensure the image fits completely within the display area
            scale_ratio = min(width_ratio, height_ratio)
            # Calculate the scaled dimensions
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)
            # Scale the image
            scaled_pixmap = self.current_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.visualization_display.setPixmap(scaled_pixmap)

    def on_resize(self, event):
        self.update_image_display()
