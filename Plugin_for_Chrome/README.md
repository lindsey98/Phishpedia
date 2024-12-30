# Plugin_for_Chrome

## Project Overview

`Plugin_for_Chrome` is a Chrome extension project designed to detect phishing websites. The extension automatically retrieves the current webpage's URL and a screenshot when the user presses a predefined hotkey or clicks the extension button, then sends this information to the server for phishing detection. The server utilizes the Flask framework, loads the Phishpedia model for identification, and returns the detection results.

## Directory Structure

```
Plugin_for_Chrome/
├── client/
│   ├── background.js        # Handles the extension's background logic, including hotkeys and button click events.
│   ├── manifest.json        # Configuration file for the Chrome extension.
│   └── popup/
│       ├── popup.html        # HTML file for the extension's popup page.
│       ├── popup.js          # JavaScript file for the extension's popup page.
│       └── popup.css         # CSS file for the extension's popup page.
└── server/
    └── app.py                # Main program for the Flask server, handling client requests and invoking the Phishpedia model for detection.
```

## Installation and Usage

### Frontend

1. Open the Chrome browser and navigate to `chrome://extensions/`.
2. Enable Developer Mode.
3. Click on "Load unpacked" and select the `Plugin_for_Chrome` directory.

### Backend

1. Navigate to the `server` directory:
    ```sh
    cd Plugin_for_Chrome/server
    ```
2. Install the required dependencies:
    ```sh
    pip install flask flask_cors
    ```
3. Run the Flask server:
    ```sh
    python app.py
    ```
## Using the Extension

In the Chrome browser, press the hotkey `Ctrl+Shift+H` or click the extension button.
The extension will automatically capture the current webpage's URL and a screenshot, then send them to the server for analysis.
The server will return the detection results, and the extension will display whether the webpage is a phishing site along with the corresponding legitimate website.

## Notes

Ensure that the server is running locally and listening on the default port 5000.
The extension and the server must operate within the same network environment.

## Contributing

Feel free to submit issues and contribute code!

## 项目简介

`Plugin_for_Chrome` 是一个用于检测钓鱼网站的Chrome插件项目。该插件可以在用户按下设置好的快捷键或者点击插件按钮后，自动获取当前网页的网址以及截图，并将其发送到服务端进行钓鱼网站检测。服务端使用Flask框架，加载Phishpedia模型进行识别，并返回检测结果。

## 目录结构

```
Plugin_for_Chrome/
├── client/
│   ├── background.js        # 处理插件的后台逻辑，包括快捷键和按钮点击事件。
│   ├── manifest.json        # Chrome插件的配置文件。
│   └── popup/
│       ├── popup.html        # 插件弹出页面的HTML文件。
│       ├── popup.js          # 插件弹出页面的JavaScript文件。
│       └── popup.css         # 插件弹出页面的CSS文件。
└── server/
    └── app.py                # Flask服务端的主程序，处理客户端请求并调用Phishpedia模型进行识别。

```

## 安装与使用

### 前端部分

1. 打开Chrome浏览器，进入 `chrome://extensions/`。
2. 启用开发者模式。
3. 点击“加载已解压的扩展程序”，选择 `Plugin_for_Chrome` 目录。

### 后端部分

1. 进入 `server` 目录：
    ```sh
    cd Plugin_for_Chrome/server
    ```
2. 安装所需依赖：
    ```sh
    pip install flask flask_cors
    ```
3. 运行Flask服务：
    ```sh
    python app.py
    ```
**或者**

使用Docker进行部署
```sh
docker build -t phishpedia-app .

docker run -p 5000:5000 phishpedia-app
```

### 使用插件

1. 在Chrome浏览器中，按下快捷键 `Ctrl+Shift+H` 然后点击插件按钮。
2. 插件会自动获取当前网页的网址和截图，并发送到服务端进行检测。
3. 服务端返回检测结果，插件会显示该网页是否为钓鱼网站，以及对应的正版网站。


## 注意事项

- 确保服务端在本地运行，并监听默认的5000端口。
- 插件和服务端需要在同一网络环境下运行。

## 贡献

欢迎提交问题和贡献代码！
