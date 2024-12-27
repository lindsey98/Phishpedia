from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTabWidget, QSizePolicy, QTreeWidget, QDialog,
    QComboBox, QTabBar
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from .function import PhishpediaFunction


class PhishpediaUI(QWidget):
    def __init__(self):
        super().__init__()
        self.function = PhishpediaFunction(self)
        self.default_font_size = 10  # 默认字体大小
        self.current_font_size = self.default_font_size
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Phishpedia GUI')
        self.setGeometry(100, 100, 600, 500)

        main_layout = QVBoxLayout()

        # Top Bar: Tab Labels and Font Size Control
        top_bar = QHBoxLayout()
        
        # Tab Labels (left side)
        tab_labels = QHBoxLayout()
        self.phish_test_btn = QPushButton("PhishTest")
        self.dataset_btn = QPushButton("Dataset")
        self.phish_test_btn.setCheckable(True)
        self.dataset_btn.setCheckable(True)
        self.phish_test_btn.setChecked(True)
        
        # 连接按钮点击事件
        self.phish_test_btn.clicked.connect(lambda: self.switch_page(0))
        self.dataset_btn.clicked.connect(lambda: self.switch_page(1))
        
        tab_labels.addWidget(self.phish_test_btn)
        tab_labels.addWidget(self.dataset_btn)
        tab_labels.addStretch()
        top_bar.addLayout(tab_labels)
        
        # Font Size Control (right side)
        font_control = QHBoxLayout()
        font_size_label = QLabel("Word Size:")
        self.font_size_combo = QComboBox()
        font_sizes = [str(size) for size in range(5, 31)]
        self.font_size_combo.addItems(font_sizes)
        self.font_size_combo.setCurrentText(str(self.default_font_size))
        self.font_size_combo.currentTextChanged.connect(self.update_global_font_size)
        
        # 设置下拉框自适应内容宽度
        self.font_size_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        # 计算最宽的选项需要的宽度
        max_width = 0
        fm = self.font_size_combo.fontMetrics()
        for size in font_sizes:
            width = fm.horizontalAdvance(size) + 30  # 30是额外的padding和箭头的空间
            max_width = max(max_width, width)
        self.font_size_combo.setMinimumWidth(max_width)
        
        font_control.addWidget(font_size_label)
        font_control.addWidget(self.font_size_combo)
        top_bar.addLayout(font_control)
        
        main_layout.addLayout(top_bar)

        # Content Stack
        self.content_stack = QTabWidget()
        self.content_stack.setTabBar(QTabBar())  # 创建一个空的TabBar
        self.content_stack.tabBar().setVisible(False)  # 隐藏TabBar
        main_layout.addWidget(self.content_stack)

        # PhishTest Page
        self.phish_test_page = QWidget()
        self.init_phish_test_page()
        self.content_stack.addTab(self.phish_test_page, "")  # 空标题，因为我们使用自定义按钮

        # Dataset Page
        self.dataset_page = QWidget()
        self.init_dataset_page()
        self.content_stack.addTab(self.dataset_page, "")

        self.setLayout(main_layout)

        # Apply initial font size
        self.update_global_font_size(str(self.default_font_size))

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                color: #424242;
                background-color: #ffffff;
            }
            
            QPushButton {
                background-color: #495057;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                margin: 0px;
            }
            
            QPushButton:checked {
                background-color: #E0E0E0;
                color: #424242;
            }
            
            QPushButton:hover:!checked {
                background-color: #5a6268;
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
            
            QComboBox {
                padding: 5px 25px 5px 5px; /* 使用CSS注释格式 */
                border: 1px solid #e9ecef;
                border-radius: 4px;
                background: white;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #424242;
            }
            
            QComboBox:on {
                border: 2px solid #6c757d;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #e9ecef;
                selection-background-color: #f8f9fa;
                selection-color: #424242;
                background: white;
                padding: 5px;
            }
        """)

    def switch_page(self, index):
        """切换页面并更新按钮状态"""
        self.content_stack.setCurrentIndex(index)
        self.phish_test_btn.setChecked(index == 0)
        self.dataset_btn.setChecked(index == 1)

    def update_global_font_size(self, size):
        """更新所有UI元素的字体大小"""
        try:
            size = int(size)
            font = QFont()
            font.setPointSize(size)
            
            # 更新所有部件的字体
            self.update_widget_fonts(self, font)
            
            # 更新标签页字体
            self.content_stack.setFont(font)
            
            # 保存当前字体大小，用于新创建的对话框
            QApplication.instance().setFont(font)
            
        except ValueError as e:
            print(f"Error updating font size: {e}")
    
    def update_widget_fonts(self, widget, font):
        """递归更新所有子部件的字体"""
        for child in widget.findChildren(QWidget):
            child.setFont(font)
            if isinstance(child, QTreeWidget):
                # 更新树形控件的所有项目字体
                for i in range(child.topLevelItemCount()):
                    item = child.topLevelItem(i)
                    self.update_tree_item_font(item, font)
            self.update_widget_fonts(child, font)

    def update_tree_item_font(self, item, font):
        """更新树形控件项目的字体"""
        item.setFont(0, font)
        for i in range(item.childCount()):
            child_item = item.child(i)
            self.update_tree_item_font(child_item, font)

    def init_phish_test_page(self):
        layout = QVBoxLayout()

        # URL Input
        url_layout = QHBoxLayout()
        self.url_label = QLabel('URL:')
        url_layout.addWidget(self.url_label)
        self.url_input = QLineEdit()
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)

        # Image Upload
        image_layout = QHBoxLayout()
        self.image_label = QLabel('Screenshot:')
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

        # Result Display Section
        result_layout = QVBoxLayout()  # 主结果布局为垂直
        
        # 第一行：检测结果
        detection_layout = QHBoxLayout()
        self.result_label = QLabel('Result:')
        self.category_display = QLineEdit()
        self.category_display.setReadOnly(True)
        detection_layout.addWidget(self.result_label)
        detection_layout.addWidget(self.category_display)
        result_layout.addLayout(detection_layout)

        # 第二行：预测目标和匹配域名（水平排列）
        details_layout = QHBoxLayout()
        
        # 预测目标
        target_layout = QHBoxLayout()
        self.target_label = QLabel('Target:')
        self.target_display = QLineEdit()
        self.target_display.setReadOnly(True)
        target_layout.addWidget(self.target_label)
        target_layout.addWidget(self.target_display)
        details_layout.addLayout(target_layout)
        
        # 匹配域名
        domain_layout = QHBoxLayout()
        self.domain_label = QLabel('Domain:')
        self.domain_display = QLineEdit()
        self.domain_display.setReadOnly(True)
        domain_layout.addWidget(self.domain_label)
        domain_layout.addWidget(self.domain_display)
        details_layout.addLayout(domain_layout)
        
        result_layout.addLayout(details_layout)
        layout.addLayout(result_layout)

        # Visualization Display
        visualization_layout = QVBoxLayout()
        self.visualization_label = QLabel('Visualization Result:')
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
            btn.setMinimumHeight(60)

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

    def create_add_brand_dialog(self, function_instance):
        # Create a custom dialog for adding brand
        dialog = QDialog(self)
        dialog.setWindowTitle('Add Brand')
        dialog.setModal(True)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Brand name input
        brand_label = QLabel('Brand Name:')
        brand_input = QLineEdit()
        brand_input.setPlaceholderText('Enter brand name')
        
        # Domain names input
        domain_label = QLabel('Domain Names:')
        domain_input = QLineEdit()
        domain_input.setPlaceholderText('Example: www.example1.com, www.example2.com')
        
        # Add input fields to layout
        layout.addWidget(brand_label)
        layout.addWidget(brand_input)
        layout.addWidget(domain_label)
        layout.addWidget(domain_input)
        
        # Button layout
        button_layout = QHBoxLayout()
        add_btn = QPushButton('Add')
        cancel_btn = QPushButton('Cancel')
        button_layout.addWidget(add_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Apply StyleSheet to dialog
        dialog.setStyleSheet("""
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
        
        dialog.setFont(QFont('Segoe UI', self.current_font_size))
        
        return dialog, brand_input, domain_input, add_btn, cancel_btn

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.function.on_resize(event)
