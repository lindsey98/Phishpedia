# Plugin_for_Chrome

## Project Overview

`Plugin_for_Chrome` is a Chrome extension project designed to detect phishing websites. 
The extension automatically retrieves the current webpage's URL and a screenshot when the user presses a predefined hotkey or clicks the extension button, then sends this information to the server for phishing detection. The server utilizes the Flask framework, loads the Phishpedia model for identification, and returns the detection results.

## Directory Structure

```
Plugin_for_Chrome/
├── client/
│   ├── background.js        # Handles the extension's background logic, including hotkeys and button click events.
│   ├── manifest.json        # Configuration file for the Chrome extension.
│   └── popup/
│       ├── popup.html        # HTML file for the extension's popup page.
│       ├── popup.js          # JavaScript file for the extension's popup page.
│       └── popup.css         # CSS file for the extension's popup page.
└── server/
    └── app.py                # Main program for the Flask server, handling client requests and invoking the Phishpedia model for detection.
```

## Installation and Usage

### Frontend

1. Open the Chrome browser and navigate to `chrome://extensions/`.
2. Enable Developer Mode.
3. Click on "Load unpacked" and select the `Plugin_for_Chrome` directory.

### Backend

1. Run the Flask server:
    ```bash
    pixi run python -m Plugin_for_Chrome.server.app
    ```
## Using the Extension

In the Chrome browser, press the hotkey `Ctrl+Shift+H` or click the extension button.
The extension will automatically capture the current webpage's URL and a screenshot, then send them to the server for analysis.
The server will return the detection results, and the extension will display whether the webpage is a phishing site along with the corresponding legitimate website.

## Notes

Ensure that the server is running locally and listening on the default port 5000.
The extension and the server must operate within the same network environment.

## Contributing

Feel free to submit issues and contribute code!

