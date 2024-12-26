from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QTabWidget, QSizePolicy, QTreeWidget
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
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
        base_font_size = 16  # Base font size for 200 DPI
        font_size = base_font_size * (dpi / 175)

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
        self.visualization_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.visualization_display.setMinimumSize(300, 300)
        visualization_layout.addWidget(self.visualization_display, 1)
        
        layout.addLayout(visualization_layout, 1)

        self.phish_test_page.setLayout(layout)

    def init_dataset_page(self):
        layout = QVBoxLayout()

        # Get directory structure
        directory_structure = self.function.get_directory_structure('models/expand_targetlist')

        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create buttons
        self.add_brand_btn = QPushButton("Add Brand")
        self.delete_brand_btn = QPushButton("Delete Brand")
        self.add_logo_btn = QPushButton("Add Logo")
        self.delete_logo_btn = QPushButton("Delete Logo")

        # 设置按钮样式
        button_style = """
            QPushButton {
                background-color: #F0F0F0;
                color: #333;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                border: 1px solid #D0D0D0;
                margin: 0 5px;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
            QPushButton:pressed {
                background-color: #D0D0D0;
            }
        """
        
        self.add_brand_btn.setStyleSheet(button_style)
        self.delete_brand_btn.setStyleSheet(button_style)
        self.add_logo_btn.setStyleSheet(button_style)
        self.delete_logo_btn.setStyleSheet(button_style)

        # 设置按钮大小策略
        for btn in [self.add_brand_btn, self.delete_brand_btn, self.add_logo_btn, self.delete_logo_btn]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(40)

        # Add buttons to layout
        button_layout.addWidget(self.add_brand_btn)
        button_layout.addWidget(self.delete_brand_btn)
        button_layout.addWidget(self.add_logo_btn)
        button_layout.addWidget(self.delete_logo_btn)
        button_layout.setSpacing(10)  # 设置按钮之间的间距

        # Connect button click events
        self.add_brand_btn.clicked.connect(self.function.add_brand)
        self.delete_brand_btn.clicked.connect(self.function.delete_brand)
        self.add_logo_btn.clicked.connect(self.function.add_logo)
        self.delete_logo_btn.clicked.connect(self.function.delete_logo)

        # Create tree view
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Brand Logos")
        self.tree_widget.itemDoubleClicked.connect(self.function.on_item_clicked)

        # 优化树形控件样式
        tree_style = """
            QTreeWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 5px;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 6px;
                margin: 2px 0;
                border-radius: 4px;
            }
            QTreeWidget::item:hover {
                background-color: #F0F0F0;
            }
            QTreeWidget::item:selected {
                background-color: #E0E0E0;
                color: #333;
            }
            QHeaderView::section {
                background-color: #F0F0F0;
                color: #333;
                padding: 5px;
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #D0D0D0;
            }
        """
        self.tree_widget.setStyleSheet(tree_style)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setIndentation(15)  # 调整缩进

        # Populate tree view
        self.function.populate_tree(self.tree_widget, directory_structure)

        # Create reload model button
        reload_layout = QHBoxLayout()
        self.reload_model_btn = QPushButton("Reload Model")
        reload_btn_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3D8B40;
            }
        """
        self.reload_model_btn.setStyleSheet(reload_btn_style)
        self.reload_model_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.reload_model_btn.setMinimumWidth(150)
        self.reload_model_btn.setMinimumHeight(40)
        reload_layout.addStretch()
        reload_layout.addWidget(self.reload_model_btn)
        
        # Connect reload model button event
        self.reload_model_btn.clicked.connect(self.function.reload_models)

        # Add all components to main layout
        layout.addLayout(button_layout)
        layout.addWidget(self.tree_widget)
        layout.addLayout(reload_layout)  # Add reload button layout

        self.dataset_page.setLayout(layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.function.on_resize(event)
