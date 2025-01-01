# help function for phishpedia web app
import os
import pickle
import shutil
import socket
import base64
import io
from PIL import Image
import cv2


def check_port_inuse(port, host):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def initial_upload_folder(upload_folder):
    try:
        shutil.rmtree(upload_folder)
    except FileNotFoundError:
        pass
    os.makedirs(upload_folder, exist_ok=True)
    
    
def convert_to_base64(image_array):
    if image_array is None:
        return None
    
    image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_array_rgb)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    plotvis_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return plotvis_base64


def domain_map_add(brand_name, domains_str, domain_map_path):
    domains = [domain.strip() for domain in domains_str.split(',') if domain.strip()]
    
    # Load existing domain mapping
    with open(domain_map_path, 'rb') as f:
        domain_map = pickle.load(f)
    
    # Add new brand and domains
    if brand_name in domain_map:
        if isinstance(domain_map[brand_name], list):
            # Add new domains, avoid duplicates
            existing_domains = set(domain_map[brand_name])
            for domain in domains:
                if domain not in existing_domains:
                    domain_map[brand_name].append(domain)
        else:
            # If current value is not a list, convert to list
            old_domain = domain_map[brand_name]
            domain_map[brand_name] = [old_domain] + [d for d in domains if d != old_domain]
    else:
        domain_map[brand_name] = domains
    
    # Save updated mapping
    with open(domain_map_path, 'wb') as f:
        pickle.dump(domain_map, f)

        
def domain_map_delete(brand_name, domain_map_path):
    # Load existing domain mapping
    with open(domain_map_path, 'rb') as f:
        domain_map = pickle.load(f)
    
    print("before deleting", len(domain_map))
    
    # Delete brand and its domains
    if brand_name in domain_map:
        del domain_map[brand_name]
    
    print("after deleting", len(domain_map))
    
    # Save updated mapping
    with open(domain_map_path, 'wb') as f:
        pickle.dump(domain_map, f)
