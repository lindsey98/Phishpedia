document.addEventListener('DOMContentLoaded', () => {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const loading = document.getElementById('loading');
  const result = document.getElementById('result');
  const status = document.getElementById('status');
  const legitUrl = document.getElementById('legitUrl');
  const legitUrlLink = document.getElementById('legitUrlLink');
  const error = document.getElementById('error');
  
  // Handle the analyze button click
  analyzeBtn.addEventListener('click', () => {
    // Show the loading state
    loading.classList.remove('hidden');
    result.classList.add('hidden');
    error.classList.add('hidden');

    // Notify the background script
    chrome.runtime.sendMessage({
      type: 'analyze'
    });
  });

  // Listen for messages from the background script
  chrome.runtime.onMessage.addListener((message) => {
    loading.classList.add('hidden');
    
    if (message.type === 'analysisResult') {
      result.classList.remove('hidden');
      
      if (message.data.isPhishing) {
        status.innerHTML = '<span class="dangerous">⚠️ Warning: this may be a phishing website!</span>';
        if (message.data.legitUrl) {
          legitUrl.classList.remove('hidden');
          legitUrlLink.href = message.data.legitUrl;
          legitUrlLink.textContent = message.data.brand;
        }
      } else {
        status.innerHTML = '<span class="safe">✓ This appears to be a safe website</span>';
        legitUrl.classList.add('hidden');
      }
    } else if (message.type === 'error') {
      error.classList.remove('hidden');
      error.querySelector('.error-message').textContent = message.data;
    }
  });
});