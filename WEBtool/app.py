from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
import os
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

        # 添加结果处理逻辑
        result = {
            "isPhishing": bool(phish_category),
            "brand": pred_target if pred_target else "unknown",
            "legitUrl": f"https://{matched_domain[0]}" if matched_domain else "unknown",
            "confidence": float(siamese_conf) if siamese_conf is not None else 0.0
        }

        # 记录日志
        today = datetime.now().strftime('%Y%m%d')
        log_file_path = os.path.join(log_dir, f'{today}_results.txt')

        try:
            with open(log_file_path, "a+", encoding='ISO-8859-1') as f:
                result_file_write(f, current_dir, url, phish_category, pred_target,
                                  matched_domain if matched_domain else ["unknown"],
                                  siamese_conf if siamese_conf is not None else 0.0,
                                  logo_recog_time, logo_match_time)
        except UnicodeError:
            with open(log_file_path, "a+", encoding='utf-8') as f:
                result_file_write(f, current_dir, url, phish_category, pred_target,
                                  matched_domain if matched_domain else ["unknown"],
                                  siamese_conf if siamese_conf is not None else 0.0,
                                  logo_recog_time, logo_match_time)

        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)

        return jsonify(result)

    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        log_error_path = os.path.join(log_dir, 'log_error.txt')
        with open(log_error_path, "a+", encoding='utf-8') as f:
            f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {str(e)}\n')
        return jsonify("ERROR"), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
