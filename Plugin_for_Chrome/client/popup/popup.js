document.addEventListener('DOMContentLoaded', () => {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const loading = document.getElementById('loading');
  const result = document.getElementById('result');
  const status = document.getElementById('status');
  const legitUrl = document.getElementById('legitUrl');
  const legitUrlLink = document.getElementById('legitUrlLink');
  const error = document.getElementById('error');
  
  // 点击分析按钮
  analyzeBtn.addEventListener('click', () => {
    // 显示加载状态
    loading.classList.remove('hidden');
    result.classList.add('hidden');
    error.classList.add('hidden');
    
    // 发送消息给background script
    chrome.runtime.sendMessage({
      type: 'analyze'
    });
  });
  
  // 监听来自background的消息
  chrome.runtime.onMessage.addListener((message) => {
    loading.classList.add('hidden');
    
    if (message.type === 'analysisResult') {
      result.classList.remove('hidden');
      
      if (message.data.isPhishing) {
        status.innerHTML = '<span class="dangerous">⚠️ 警告：这可能是一个钓鱼网站！</span>';
        if (message.data.legitUrl) {
          legitUrl.classList.remove('hidden');
          legitUrlLink.href = message.data.legitUrl;
          legitUrlLink.textContent = message.data.brand;
        }
      } else {
        status.innerHTML = '<span class="safe">✓ 这是一个安全的网站</span>';
        legitUrl.classList.add('hidden');
      }
    } else if (message.type === 'error') {
      error.classList.remove('hidden');
      error.querySelector('.error-message').textContent = message.data;
    }
  });
});