document.addEventListener('DOMContentLoaded', function() {
    // 获取当前快捷键设置并显示
    chrome.commands.getAll((commands) => {
        const captureCommand = commands.find(command => command.name === "capture-page");
        if (captureCommand && captureCommand.shortcut) {
            const shortcutText = document.querySelector('p');
            shortcutText.textContent = `使用快捷键 ${captureCommand.shortcut} 捕获当前页面`;
        }
    });

    // 添加手动捕获按钮（可选）
    const captureButton = document.createElement('button');
    captureButton.textContent = '手动捕获';
    captureButton.style.cssText = `
        margin-top: 10px;
        padding: 8px 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    `;
    
    captureButton.addEventListener('click', function() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            const currentTab = tabs[0];
            
            chrome.tabs.captureVisibleTab(null, {format: 'png'}, function(dataUrl) {
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
                    // 显示结果
                    const resultDiv = document.createElement('div');
                    resultDiv.textContent = result.message;
                    resultDiv.style.cssText = `
                        margin-top: 10px;
                        padding: 8px;
                        background-color: #f0f0f0;
                        border-radius: 4px;
                    `;
                    document.body.appendChild(resultDiv);
                    
                    // 3秒后移除结果提示
                    setTimeout(() => {
                        resultDiv.remove();
                    }, 3000);
                })
                .catch(error => {
                    console.error('Error:', error);
                    const errorDiv = document.createElement('div');
                    errorDiv.textContent = '发生错误，请检查服务器是否正常运行';
                    errorDiv.style.color = 'red';
                    document.body.appendChild(errorDiv);
                });
            });
        });
    });
    
    document.body.appendChild(captureButton);
});
