from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# 这里后续添加模型加载代码
def load_model():
    # TODO: 加载识别模型
    pass

# 在创建应用时初始化模型
with app.app_context():
    load_model()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get('url')
        screenshot_data = data.get('screenshot')
        
        # 解码Base64图片数据
        image_data = base64.b64decode(screenshot_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        
        # TODO: 这里添加识别逻辑
        # 目前返回示例数据
        result = {
            "isPhishing": False,
            "legitUrl": None,
            "confidence": 0.95
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)