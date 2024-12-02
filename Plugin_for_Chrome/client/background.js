chrome.commands.onCommand.addListener((command) => {
  if (command === "capture-page") {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const currentTab = tabs[0];
      
      // 获取页面截图
      chrome.tabs.captureVisibleTab(null, {format: 'png'}, function(dataUrl) {
        // 将截图和URL发送到服务器
        const data = {
          url: currentTab.url,
          screenshot: dataUrl
        };
        
        fetch('http://localhost:5000/upload', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
          // 显示结果通知
          chrome.tabs.sendMessage(currentTab.id, {
            type: 'show_result',
            result: result
          });
        })
        .catch(error => console.error('Error:', error));
      });
    });
  }
});
