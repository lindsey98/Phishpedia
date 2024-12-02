from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import time

app = Flask(__name__)
CORS(app)  # 启用跨域支持

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.json
    url = data.get('url')
    screenshot = data.get('screenshot')
    
    # 从Base64字符串中提取图片数据
    image_data = screenshot.split(',')[1]
    image_binary = base64.b64decode(image_data)
    
    # 生成文件名
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # 保存图片
    with open(filepath, 'wb') as f:
        f.write(image_binary)
    
    # 这里可以添加您的图片处理逻辑
    # process_image(filepath)
    
    return jsonify({
        "status": "success",
        "message": "截图已保存并处理完成！",
        "url": url,
        "filename": filename
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
