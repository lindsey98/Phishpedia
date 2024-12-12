from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
import os
import sys
from phishpedia import PhishpediaWrapper, result_file_write


app = Flask(__name__)
CORS(app)


# 在创建应用时初始化模型
with app.app_context():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir, 'plugin_logs')
    os.makedirs(log_dir, exist_ok=True)
    phishpedia_cls = PhishpediaWrapper()


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print('Request received')
        data = request.get_json()
        url = data.get('url')
        screenshot_data = data.get('screenshot')
        
        # 解码Base64图片数据
        image_data = base64.b64decode(screenshot_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        screenshot_path = 'temp_screenshot.png'
        image.save(screenshot_path, format='PNG')

        # 调用Phishpedia模型进行识别
        phish_category, pred_target, matched_domain, \
            plotvis, siamese_conf, pred_boxes, \
            logo_recog_time, logo_match_time = phishpedia_cls.test_orig_phishpedia(url, screenshot_path, None)
        
        today = datetime.now().strftime('%Y%m%d')
        log_file_path = os.path.join(log_dir, f'{today}_results.txt')

        try:
            with open(log_file_path, "a+", encoding='ISO-8859-1') as f:
                result_file_write(f, current_dir, url, phish_category, pred_target, matched_domain, siamese_conf,
                                  logo_recog_time, logo_match_time)
        except UnicodeError:
            with open(log_file_path, "a+", encoding='utf-8') as f:
                result_file_write(f, current_dir, url, phish_category, pred_target, matched_domain, siamese_conf,
                                  logo_recog_time, logo_match_time)
        # 目前返回示例数据
        result = {
            "isPhishing": bool(phish_category),
            "brand": pred_target,
            "legitUrl": "https://" + matched_domain[0],
            "confidence": float(siamese_conf)
        }
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)
        return jsonify(result)
    
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)
