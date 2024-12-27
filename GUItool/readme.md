# Phishpedia GUI Tool

Phishpedia GUI is a graphical interface tool for phishing website detection. It provides a user-friendly interface with brand and domain management capabilities, as well as visualization features for phishing detection.

## Installation Requirements

Before using, make sure all necessary dependencies are installed:

```bash
pip install -r requirements.txt
```

## How to Run

Run the following command in the project root directory:

```bash
python phishpedia_gui.py
```

## User Guide

### 1. Phishing Detection Page (PhishTest)

1. **URL Detection**
   - Enter the URL to be tested in the "Enter URL" input box
   - Click the "Browse" button to select the corresponding website screenshot
   - Click the "Detect" button to start detection
   - Detection results will be displayed below, including text results and visual presentation

2. **Result Display**
   - The detection results will be displayed in the "Result" text box
   - The matched logos will be displayed in the "Target" text box
   - The matched domains will be displayed in the "Domain" text box
   - Visual results will be displayed in the "Visualization Result" area
   - You can clearly see the detected brand identifiers and related information

### 2. Dataset Management Page (Dataset)

1. **Brand Management**
   - Click "Add Brand" to add a new brand
   - Enter brand name and corresponding domains in the popup window
   - Click "Delete Brand" to remove the selected brand

2. **Logo Management**
   - After selecting a brand, click "Add Logo" to add brand logos
   - Click "Delete Logo" to remove selected logos
   - All logo files will be displayed in the tree view

3. **Data Update**
   - After making changes, click the "Reload Model" button
   - The system will reload the updated dataset

## Main Features

1. **Phishing Detection**
   - URL input and detection
   - Screenshot upload and analysis
   - Detection result visualization

2. **Brand Management**
   - Add/Delete brands
   - Add/Delete brand logos
   - Domain management
   - Model reloading

## Directory Structure

```
GUItool/
├── ui.py               # UI layout and style definitions
├── function.py         # Core functionality implementation
├── readme.md           # Documentation
└── requirements.txt    # Dependency list
```

### File Description

- **ui.py**: 
  - Defines main window layout
  - Contains all UI component styles
  - Implements dynamic font size adjustment
  - Manages two main tabs: PhishTest and Dataset

- **function.py**: 
  - Implements all core functionalities
  - Handles brand and logo addition/deletion
  - Manages domain mapping
  - Executes phishing detection logic
  - Handles file upload and visualization

- **requirements.txt**: 
  - Lists all required Python packages
  - Contains PyQt5 UI dependencies

---

# Phishpedia GUI 工具

Phishpedia GUI 是一个用于钓鱼网站检测的图形界面工具。它提供了友好的用户界面，支持品牌和域名管理，以及钓鱼网站的可视化检测功能。

## 安装要求

在使用之前，请确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 运行方法

在项目根目录下运行以下命令：

```bash
python phishpedia_gui.py
```

## 使用指南

### 1. 钓鱼检测页面（PhishTest）

1. **URL检测**
   - 在"Enter URL"输入框中输入待检测的网址
   - 点击"Browse"按钮选择对应的网站截图
   - 点击"Detect"按钮开始检测
   - 检测结果将在下方显示，包括文字结果和可视化展示


2. **结果展示**
   - 检测结果会显示在"Result"文本框中
   - 匹配到的logo显示在"Target"文本框中
   - 匹配到的域名显示在"Domain"文本框中
   - 可视化结果会在"Visualization Result"区域展示
   - 可以清晰看到检测到的品牌标识和相关信息

### 2. 数据集管理页面（Dataset）

1. **品牌管理**
   - 点击"Add Brand"添加新的品牌
   - 在弹出窗口中输入品牌名称和对应的域名
   - 点击"Delete Brand"可删除选中的品牌

2. **Logo管理**
   - 选择品牌后，点击"Add Logo"添加品牌Logo
   - 点击"Delete Logo"可删除选中的Logo
   - 所有Logo文件会在树形视图中显示

3. **数据更新**
   - 完成修改后，点击"Reload Model"按钮
   - 系统会重新加载更新后的数据集

## 主要功能

1. **钓鱼检测**
   - URL输入和检测
   - 截图上传和分析
   - 检测结果可视化展示

2. **品牌管理**
   - 添加/删除品牌
   - 添加/删除品牌Logo
   - 域名管理
   - 模型重新加载

## 目录结构

```
GUItool/
├── ui.py               # 界面布局和样式定义
├── function.py         # 功能实现模块
├── readme.md           # 说明文档
└── requirements.txt    # 依赖包列表
```

### 文件说明

- **ui.py**: 
  - 定义了主窗口界面布局
  - 包含所有UI组件的样式设置
  - 实现了动态字体大小调整
  - 管理界面的两个主要标签页：PhishTest和Dataset

- **function.py**: 
  - 实现所有核心功能
  - 处理品牌和Logo的添加/删除
  - 管理域名映射
  - 执行钓鱼检测逻辑
  - 处理文件上传和可视化

- **requirements.txt**: 
  - 列出所有必需的Python包
  - 包含PyQt5 UI相关依赖