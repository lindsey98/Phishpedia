// Capture the screenshot and URL of the active tab, then request analysis
async function captureTabInfo(tab) {
  try {
    // Capture the visible tab as a screenshot
    const screenshot = await chrome.tabs.captureVisibleTab(null, {
      format: 'png'
    });

    // Get the current URL
    const url = tab.url;

    // Send to the server for analysis
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

    // Forward the result to the popup
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

// Listen for the keyboard shortcut command
chrome.commands.onCommand.addListener(async (command) => {
  if (command === '_execute_action') {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab) {
      await captureTabInfo(tab);
    }
  }
});

// Listen for messages from the popup
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