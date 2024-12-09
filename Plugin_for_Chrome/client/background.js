// 处理截图和URL获取
async function captureTabInfo(tab) {
  try {
    // 获取截图
    const screenshot = await chrome.tabs.captureVisibleTab(null, {
      format: 'png'
    });
    
    // 获取当前URL
    const url = tab.url;
    
    // 发送到服务器进行分析
    const response = await fetch('http://localhost:5000/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: url,
        screenshot: screenshot
      })
    });
    
    const result = await response.json();
    
    // 将结果发送到popup
    chrome.runtime.sendMessage({
      type: 'analysisResult',
      data: result
    });
    
  } catch (error) {
    console.error('Error capturing tab info:', error);
    chrome.runtime.sendMessage({
      type: 'error',
      data: error.message
    });
  }
}

// 监听快捷键命令
chrome.commands.onCommand.addListener(async (command) => {
  if (command === '_execute_action') {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab) {
      await captureTabInfo(tab);
    }
  }
});

// 监听来自popup的消息
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'analyze') {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (tabs[0]) {
        await captureTabInfo(tabs[0]);
      }
    });
  }
  return true;
});