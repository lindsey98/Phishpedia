# Plugin_for_Chrome

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

### 使用插件

1. 在Chrome浏览器中，按下快捷键 `Ctrl+Shift+Y` 或点击插件按钮。
2. 插件会自动获取当前网页的网址和截图，并发送到服务端进行检测。
3. 服务端返回检测结果，插件会显示该网页是否为钓鱼网站，以及对应的正版网站。


## 注意事项

- 确保服务端在本地运行，并监听默认的5000端口。
- 插件和服务端需要在同一网络环境下运行。

## 贡献

欢迎提交问题和贡献代码！
