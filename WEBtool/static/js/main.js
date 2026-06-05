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

            // Send a POST request to the /detect route
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
                    alert('Detection failed, please try again later.');
                });
        },
        handleImageUpload(event) {  // Handle the image upload event
            const file = event.target.files[0];
            if (file) {
                this.uploadedImage = file;
                this.uploadImage();
            }
        },
        uploadImage() {  // Upload the image to the server
            const formData = new FormData();
            formData.append('image', this.uploadedImage);

            fetch('/upload', {  // upload route
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        this.imageUrl = data.imageUrl;  // update the image URL
                        this.uploadSuccess = true;  // mark upload as successful
                    } else {
                        alert('Failed to upload image: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to upload image, please try again later.');
                });
        },
        clearUpload() {  // Clear the uploaded image
            fetch('/clear_upload', {  // clear-upload route
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
                        this.uploadSuccess = false;  // reset upload state
                    } else {
                        alert('Failed to delete image: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to delete image, please try again later.');
                });
        }
    }
});
