chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'show_result') {
    // 创建一个悬浮提示框
    const div = document.createElement('div');
    div.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 15px;
      background: #333;
      color: white;
      border-radius: 5px;
      z-index: 10000;
    `;
    div.textContent = message.result.message;
    document.body.appendChild(div);
    
    // 3秒后自动消失
    setTimeout(() => {
      div.remove();
    }, 3000);
  }
});
