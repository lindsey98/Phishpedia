import os
import shutil
from flask import request, Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS
from utils_web import allowed_file, convert_to_base64, domain_map_add, domain_map_delete, check_port_inuse, initial_upload_folder
from configs import load_config
from phishpedia import PhishpediaWrapper

phishpedia_cls = None

# flask for API server
app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FILE_TREE_ROOT'] = '../models/expand_targetlist'  # 主目录路径
app.config['DOMAIN_MAP_PATH'] = '../models/domain_map.pkl'


@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传请求"""
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        if filename.count('.') > 1:
            return jsonify({'error': 'Invalid file name'}), 400
        elif any(sep in filename for sep in (os.sep, os.altsep)):
            return jsonify({'error': 'Invalid file name'}), 400
        elif '..' in filename:
            return jsonify({'error': 'Invalid file name'}), 400
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_path = os.path.normpath(file_path)
        if not file_path.startswith(app.config['UPLOAD_FOLDER']):
            return jsonify({'error': 'Invalid file path'}), 400
        file.save(file_path)
        return jsonify({'success': True, 'imageUrl': f'/uploads/{filename}'}), 200

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传文件的访问路径"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/clear_upload', methods=['POST'])
def delete_image():
    data = request.get_json()
    image_url = data.get('imageUrl')

    if not image_url:
        return jsonify({'success': False, 'error': 'No image URL provided'}), 400

    try:
        # 假设 image_url 是相对于静态目录的路径
        filename = image_url.split('/')[-1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_path = os.path.normpath(image_path)
        if not image_path.startswith(app.config['UPLOAD_FOLDER']):
            return jsonify({'success': False, 'error': 'Invalid file path'}), 400
        os.remove(image_path)
        return jsonify({'success': True}), 200
    except Exception:
        return jsonify({'success': False}), 500


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    url = data.get('url', '')
    imageUrl = data.get('imageUrl', '')
    
    filename = imageUrl.split('/')[-1]
    screenshot_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    screenshot_path = os.path.normpath(screenshot_path)
    if not screenshot_path.startswith(app.config['UPLOAD_FOLDER']):
        return jsonify({'success': False, 'error': 'Invalid file path'}), 400

    phish_category, pred_target, matched_domain, plotvis, siamese_conf, _, logo_recog_time, logo_match_time = phishpedia_cls.test_orig_phishpedia(
        url, screenshot_path, None)
    
    # 处理检测结果
    if phish_category == 0:
        if pred_target is None:
            result = 'Unknown'
        else:
            result = 'Benign'
    else:
        result = 'Phishing'

    plot_base64 = convert_to_base64(plotvis)

    # 返回检测结果
    result = {
        'result': result,  # 检测结果
        'matched_brand': pred_target,  # 匹配到的品牌
        'correct_domain': matched_domain,  # 正确的域名
        'confidence': round(float(siamese_conf), 3),  # 置信度，直接返回百分比
        'detection_time': round(float(logo_recog_time) + float(logo_match_time), 3),  # 检测时间
        'logo_extraction': plot_base64  # logo标注结果，直接返回图像
    }
    return jsonify(result)


@app.route('/get-directory', methods=['GET'])
def get_file_tree():
    """
    获取主目录的文件树
    """
    def build_file_tree(path):
        tree = []
        try:
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                entry_path = os.path.normpath(entry_path)
                if not entry_path.startswith(path):
                    continue
                if os.path.isdir(entry_path):
                    tree.append({
                        'name': entry,
                        'type': 'directory',
                        'children': build_file_tree(entry_path)  # 递归子目录
                    })
                elif entry.lower().endswith(('.png', '.jpeg', '.jpg')):
                    tree.append({
                        'name': entry,
                        'type': 'file'
                    })
                else:
                    continue
        except PermissionError:
            pass  # 忽略权限错误
        return sorted(tree, key=lambda x: x['name'].lower())  # 按 name 字段排序，不区分大小写

    root_path = app.config['FILE_TREE_ROOT']
    if not os.path.exists(root_path):
        return jsonify({'error': 'Root directory does not exist'}), 404

    file_tree = build_file_tree(root_path)
    return jsonify({'file_tree': file_tree}), 200


@app.route('/view-file', methods=['GET'])
def view_file():
    file_name = request.args.get('file')
    file_path = os.path.join(app.config['FILE_TREE_ROOT'], file_name)
    file_path = os.path.normpath(file_path)
    if not file_path.startswith(app.config['FILE_TREE_ROOT']):
        return jsonify({'error': 'Invalid file path'}), 400

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    if file_name.lower().endswith(('.png', '.jpeg', '.jpg')):
        return send_from_directory(app.config['FILE_TREE_ROOT'], file_name)
    
    return jsonify({'error': 'Unsupported file type'}), 400


@app.route('/add-logo', methods=['POST'])
def add_logo():
    if 'logo' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    logo = request.files['logo']
    if logo.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    if logo and allowed_file(logo.filename):
        directory = request.form.get('directory')
        if not directory:
            return jsonify({'success': False, 'error': 'No directory specified'}), 400

        directory_path = os.path.join(app.config['FILE_TREE_ROOT'], directory)
        directory_path = os.path.normpath(directory_path)
        if not directory_path.startswith(app.config['FILE_TREE_ROOT']):
            return jsonify({'success': False, 'error': 'Invalid directory path'}), 400
        
        if not os.path.exists(directory_path):
            return jsonify({'success': False, 'error': 'Directory does not exist'}), 400

        file_path = os.path.join(directory_path, logo.filename)
        file_path = os.path.normpath(file_path)
        if not file_path.startswith(directory_path):
            return jsonify({'success': False, 'error': 'Invalid file path'}), 400
        logo.save(file_path)
        return jsonify({'success': True, 'message': 'Logo added successfully'}), 200

    return jsonify({'success': False, 'error': 'Invalid file type'}), 400


@app.route('/del-logo', methods=['POST'])
def del_logo():
    directory = request.form.get('directory')
    filename = request.form.get('filename')

    if not directory or not filename:
        return jsonify({'success': False, 'error': 'Directory and filename must be specified'}), 400

    directory_path = os.path.join(app.config['FILE_TREE_ROOT'], directory)
    directory_path = os.path.normpath(directory_path)
    if not directory_path.startswith(app.config['FILE_TREE_ROOT']):
        return jsonify({'success': False, 'error': 'Invalid directory path'}), 400
    file_path = os.path.join(directory_path, filename)
    file_path = os.path.normpath(file_path)
    if not file_path.startswith(directory_path):
        return jsonify({'success': False, 'error': 'Invalid file path'}), 400

    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File does not exist'}), 400

    try:
        os.remove(file_path)
        return jsonify({'success': True, 'message': 'Logo deleted successfully'}), 200
    except Exception:
        return jsonify({'success': False}), 500


@app.route('/add-brand', methods=['POST'])
def add_brand():
    brand_name = request.form.get('brandName')
    brand_domain = request.form.get('brandDomain')

    if not brand_name or not brand_domain:
        return jsonify({'success': False, 'error': 'Brand name and domain must be specified'}), 400

    # 创建品牌目录
    brand_directory_path = os.path.join(app.config['FILE_TREE_ROOT'], brand_name)
    brand_directory_path = os.path.normpath(brand_directory_path)
    if not brand_directory_path.startswith(app.config['FILE_TREE_ROOT']):
        return jsonify({'success': False, 'error': 'Invalid brand directory path'}), 400
    
    if os.path.exists(brand_directory_path):
        return jsonify({'success': False, 'error': 'Brand already exists'}), 400

    try:
        os.makedirs(brand_directory_path)
        domain_map_add(brand_name, brand_domain, app.config['DOMAIN_MAP_PATH'])
        return jsonify({'success': True, 'message': 'Brand added successfully'}), 200
    except Exception:
        return jsonify({'success': False}), 500


@app.route('/del-brand', methods=['POST'])
def del_brand():
    directory = request.json.get('directory')

    if not directory:
        return jsonify({'success': False, 'error': 'Directory must be specified'}), 400

    directory_path = os.path.join(app.config['FILE_TREE_ROOT'], directory)
    directory_path = os.path.normpath(directory_path)
    if not directory_path.startswith(app.config['FILE_TREE_ROOT']):
        return jsonify({'success': False, 'error': 'Invalid directory path'}), 400

    if not os.path.exists(directory_path):
        return jsonify({'success': False, 'error': 'Directory does not exist'}), 400

    try:
        shutil.rmtree(directory_path)
        domain_map_delete(directory, app.config['DOMAIN_MAP_PATH'])
        return jsonify({'success': True, 'message': 'Brand deleted successfully'}), 200
    except Exception:
        return jsonify({'success': False}), 500


@app.route('/reload-model', methods=['POST'])
def reload_model():
    global phishpedia_cls
    try:
        load_config(reload_targetlist=True)
        # Reinitialize Phishpedia
        phishpedia_cls = PhishpediaWrapper()
        return jsonify({'success': True, 'message': 'Brand deleted successfully'}), 200
    except Exception:
        return jsonify({'success': False}), 500


if __name__ == "__main__":
    ip_address = '0.0.0.0'
    port = 5000
    while check_port_inuse(port, ip_address):
        port = port + 1

    # 加载核心检测逻辑
    phishpedia_cls = PhishpediaWrapper()

    initial_upload_folder(app.config['UPLOAD_FOLDER'])
    
    app.run(host=ip_address, port=port)
