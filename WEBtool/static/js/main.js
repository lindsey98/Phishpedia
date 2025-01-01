new Vue({
    el: '#main-container',
    data() {
        return {
            url: '',
            result: null,
            uploadedImage: null,
            imageUrl: '',
            uploadSuccess: false,
        }
    },
    methods: {
        startDetection() {
            if (!this.url) {
                alert('Please enter a valid URL.');
                return;
            }

            // 发送 POST 请求到 /detect 路由
            fetch('/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: this.url,
                    imageUrl: this.imageUrl
                })
            })
                .then(response => response.json())
                .then(data => {
                    this.result = data;  // Update all data

                    if (data.logo_extraction) { // Logo Extraction Result
                        document.getElementById('original-image').src = `data:image/png;base64,${data.logo_extraction}`;
                    }

                    // Detectoin Result
                    const labelElement = document.getElementById('detection-label');
                    const explanationElement = document.getElementById('detection-explanation');
                    const matched_brand_element = document.getElementById('matched-brand');
                    const siamese_conf_element = document.getElementById('siamese-conf');
                    const correct_domain_element = document.getElementById('correct-domain');
                    const detection_time_element = document.getElementById('detection-time');

                    detection_time_element.textContent = data.detection_time + ' s';
                    if (data.result === 'Benign') {
                        labelElement.className = 'benign';
                        labelElement.textContent = 'Benign';
                        matched_brand_element.textContent = data.matched_brand;
                        siamese_conf_element.textContent = data.confidence;
                        correct_domain_element.textContent = data.correct_domain;
                        explanationElement.innerHTML = `
                            <p>This website has been analyzed and determined to be <strong>${labelElement.textContent.toLowerCase()}</strong>. 
                            Because we have matched a brand <strong>${data.matched_brand}</strong> with confidence <strong>${Math.round(data.confidence * 100, 3)}, </strong>
                            and the domain extracted from url is within the domain list under the brand (which is <strong>[${data.correct_domain}]</strong>). 
                            Enjoy your surfing!</p>
                        `;
                    } else if (data.result === 'Phishing') {
                        labelElement.className = 'phishing';
                        labelElement.textContent = 'Phishing';
                        matched_brand_element.textContent = data.matched_brand;
                        siamese_conf_element.textContent = data.confidence;
                        correct_domain_element.textContent = data.correct_domain;
                        explanationElement.innerHTML = `
                            <p>This website has been analyzed and determined to be <strong>${labelElement.textContent.toLowerCase()}</strong>. 
                            Because we have matched a brand <strong>${data.matched_brand}</strong> with confidence <strong>${Math.round(data.confidence * 100, 3)}%</strong>, 
                            but the domain extracted from url is NOT within the domain list under the brand (which is <strong>[${data.correct_domain}]</strong>). 
                            Please proceed with caution!</p>
                        `;
                    } else {
                        labelElement.className = 'unknown';
                        labelElement.textContent = 'Unknown';
                        matched_brand_element.textContent = "unknown";
                        siamese_conf_element.textContent = "0.00";
                        correct_domain_element.textContent = "unknown";
                        explanationElement.innerHTML = `
                            <p>Sorry, we don't find any matched brand in database so this website is determined to be <strong>${labelElement.textContent.toLowerCase()}</strong>.</p>
                            <p>It is still possible that this is a <strong>phishing</strong> site. Please proceed with caution!</p>
                        `;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('检测失败，请稍后重试。');
                });
        },
        handleImageUpload(event) {  // 处理图片上传事件
            const file = event.target.files[0];
            if (file) {
                this.uploadedImage = file;
                this.uploadImage();
            }
        },
        uploadImage() {  // 上传图片到服务器
            const formData = new FormData();
            formData.append('image', this.uploadedImage);

            fetch('/upload', {  // 假设上传图片的路由是 /upload
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        this.imageUrl = data.imageUrl;  // 更新图片URL
                        this.uploadSuccess = true;  // 标记上传成功
                    } else {
                        alert('上传图片失败: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('上传图片失败，请稍后重试。');
                });
        },
        clearUpload() {  // 清除上传的图像
            fetch('/clear_upload', {  // 假设删除图片的路由是 /delete-image
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ imageUrl: this.imageUrl })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        this.imageUrl = '';
                        this.uploadSuccess = false;  // 重置上传状态
                    } else {
                        alert('删除图片失败: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('删除图片失败，请稍后重试。');
                });
        }
    }
});
