from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QHBoxLayout, QTabWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from .function import PhishpediaFunction


class PhishpediaUI(QWidget):
    def __init__(self):
        super().__init__()
        self.function = PhishpediaFunction(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Phishpedia GUI')
        self.setGeometry(100, 100, 600, 500)

        main_layout = QVBoxLayout()

        # Navigation Bar
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # PhishTest Page
        self.phish_test_page = QWidget()
        self.init_phish_test_page()
        self.tab_widget.addTab(self.phish_test_page, "PhishTest")

        # Import Model Page
        self.import_model_page = QWidget()
        self.init_import_model_page()
        self.tab_widget.addTab(self.import_model_page, "Import Model")

        # Dataset Page
        self.dataset_page = QWidget()
        self.init_dataset_page()
        self.tab_widget.addTab(self.dataset_page, "Dataset")

        self.setLayout(main_layout)

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                color: #424242;
                background-color: #ffffff;
            }
            
            QLabel {
                color: #424242;
                font-weight: 500;
            }
            
            QLineEdit, QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                padding: 8px 12px;
                border-radius: 6px;
                color: #424242;
            }
            
            QLineEdit:focus, QTextEdit:focus {
                border: 2px solid #6c757d;
                background-color: #ffffff;
            }
            
            QPushButton {
                background-color: #495057;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #5a6268;
            }
            
            QPushButton:pressed {
                background-color: #3d4246;
            }
            
            QTabWidget::pane {
                border: 1px solid #e9ecef;
                border-radius: 6px;
                background: white;
            }
            
            QTabBar::tab {
                background: #f8f9fa;
                color: #424242;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            
            QTabBar::tab:selected {
                background: #495057;
                color: white;
            }
            
            QTabBar::tab:hover:!selected {
                background: #e9ecef;
            }
        """)

        # Set dynamic font size based on screen DPI
        self.set_dynamic_font_size()

    def set_dynamic_font_size(self):
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch()
        base_font_size = 16  # Base font size for 150 DPI
        font_size = base_font_size * (dpi / 150)

        font = QFont()
        font.setPointSizeF(font_size)

        for widget in self.findChildren(QLabel) + self.findChildren(QLineEdit) + self.findChildren(QPushButton) + [
                self.result_display]:
            widget.setFont(font)

    def init_phish_test_page(self):
        layout = QVBoxLayout()

        # URL Input
        url_layout = QHBoxLayout()
        self.url_label = QLabel('Enter URL:')
        url_layout.addWidget(self.url_label)
        self.url_input = QLineEdit()
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)

        # Image Upload
        image_layout = QHBoxLayout()
        self.image_label = QLabel('Upload Screenshot:')
        image_layout.addWidget(self.image_label)
        self.image_input = QLineEdit()
        image_layout.addWidget(self.image_input)
        self.image_button = QPushButton('Browse')
        self.image_button.clicked.connect(self.function.upload_image)
        image_layout.addWidget(self.image_button)
        layout.addLayout(image_layout)

        # Detect Button
        self.detect_button = QPushButton('Detect')
        self.detect_button.clicked.connect(self.function.detect_phishing)
        layout.addWidget(self.detect_button)

        # Result Display
        result_layout = QHBoxLayout()
        self.result_label = QLabel('Detection Result:')
        result_layout.addWidget(self.result_label)
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFixedHeight(100)
        result_layout.addWidget(self.result_display)
        layout.addLayout(result_layout)

        # Visualization Display
        visualization_layout = QVBoxLayout()
        self.visualization_label = QLabel('Visualization Result:')
        self.visualization_label.setFixedHeight(self.visualization_label.fontMetrics().height())
        visualization_layout.addWidget(self.visualization_label)
        self.visualization_display = QLabel()
        self.visualization_display.setAlignment(Qt.AlignCenter)
        self.visualization_display.setMinimumSize(300, 200)
        visualization_layout.addWidget(self.visualization_display)
        layout.addLayout(visualization_layout)

        self.phish_test_page.setLayout(layout)

    def init_import_model_page(self):
        layout = QVBoxLayout()
        self.import_model_page.setLayout(layout)

    def init_dataset_page(self):
        layout = QVBoxLayout()
        self.dataset_page.setLayout(layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.function.on_resize(event)
