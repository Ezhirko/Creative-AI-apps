(function() {
  if (window.hasRun) {
    return;
  }
  window.hasRun = true;

  console.log('Content script loaded and initialized');

  let latestSelectedText = '';

  function updateSelectedText() {
    const selection = window.getSelection();
    latestSelectedText = selection.toString().trim();
    if (latestSelectedText) {
      console.log('Text selected:', latestSelectedText.substring(0, 100) + '...');
    }
  }

  document.addEventListener('mouseup', updateSelectedText);
  document.addEventListener('keyup', updateSelectedText);

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Message received in content script:', request.action);
    if (request.action === "getSelectedText") {
      console.log('Sending selected text:', latestSelectedText.substring(0, 100) + '...');
      sendResponse({text: latestSelectedText});
    } else if (request.action === "checkContentScript") {
      sendResponse({loaded: true});
    }
    return true; // Keep the message channel open for asynchronous response
  });

  // Notify that the content script is loaded
  chrome.runtime.sendMessage({action: "contentScriptLoaded"});
})();

// Log a message every 5 seconds to ensure the script is running
setInterval(() => {
    console.log('Content script is still running');
}, 5000);
