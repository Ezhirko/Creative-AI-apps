// Set up PDF.js
const pdfjsLib = window['pdfjs-dist/build/pdf'];
pdfjsLib.GlobalWorkerOptions.workerSrc = 'pdf.worker.min.js';

document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM fully loaded and parsed');
  
  const resumeUpload = document.getElementById('resumeUpload');
  const analyzeButton = document.getElementById('analyzeButton');
  const resultDiv = document.getElementById('result');
  const statusDiv = document.getElementById('status');

  console.log('Popup script loaded');

  resumeUpload.addEventListener('change', function(event) {
    console.log('File selected');
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      console.log('PDF file detected');
      const reader = new FileReader();
      reader.onload = function(e) {
        console.log('File read successfully');
        const typedArray = new Uint8Array(e.target.result);
        pdfjsLib.getDocument(typedArray).promise.then(function(pdf) {
          console.log('PDF loaded successfully');
          const numPages = pdf.numPages;
          let pagesPromises = [];

          for (let i = 1; i <= numPages; i++) {
            pagesPromises.push(getPageText(pdf, i));
          }

          Promise.all(pagesPromises).then(function(pagesText) {
            const resumeText = pagesText.join(' ');
            console.log('Resume text extracted:', resumeText.substring(0, 100) + '...');
            chrome.storage.local.set({ resumeText: resumeText }, function() {
              console.log('Resume text saved to storage');
              statusDiv.textContent = 'Resume uploaded successfully!';
            });
          }).catch(function(error) {
            console.error('Error extracting text from PDF:', error);
            statusDiv.textContent = 'Error extracting text from PDF. Please try again.';
          });
        }).catch(function(error) {
          console.error('Error loading PDF:', error);
          statusDiv.textContent = 'Error loading PDF. Please try again.';
        });
      };
      reader.readAsArrayBuffer(file);
    } else {
      console.log('Invalid file type');
      statusDiv.textContent = 'Please upload a PDF file.';
    }
  });

  analyzeButton.addEventListener('click', function() {
    console.log('Analyze button clicked');
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      if (tabs[0]) {
        checkContentScript(tabs[0].id);
      } else {
        console.error("No active tab found");
        statusDiv.textContent = "Error: No active tab found";
      }
    });
  });

  function checkContentScript(tabId) {
    chrome.tabs.sendMessage(tabId, {action: "checkContentScript"}, function(response) {
      if (chrome.runtime.lastError) {
        console.log("Content script not loaded, injecting now");
        chrome.scripting.executeScript({
          target: { tabId: tabId },
          files: ['content.js']
        }, () => {
          if (chrome.runtime.lastError) {
            console.error('Error injecting content script:', chrome.runtime.lastError);
            statusDiv.textContent = "Error: Unable to inject content script";
          } else {
            console.log('Content script injected successfully');
            setTimeout(() => getSelectedText(tabId), 100);
          }
        });
      } else {
        console.log("Content script already loaded");
        getSelectedText(tabId);
      }
    });
  }

  function getSelectedText(tabId) {
    chrome.tabs.sendMessage(tabId, {action: "getSelectedText"}, function(response) {
      if (chrome.runtime.lastError) {
        console.error("Error getting selected text:", chrome.runtime.lastError);
        statusDiv.textContent = "Error: Unable to get selected text. Make sure you're on a web page.";
        return;
      }
      if (response && response.text) {
        console.log('Job description received:', response.text.substring(0, 100) + '...');
        chrome.runtime.sendMessage({
          action: "analyzeJobDescription",
          jobDescription: response.text
        }, function(response) {
          if (chrome.runtime.lastError) {
            console.error("Error analyzing job description:", chrome.runtime.lastError);
            statusDiv.textContent = "Error: Unable to analyze job description";
          } else {
            console.log('Analysis result received');
            resultDiv.innerHTML = response.result; // Changed from textContent to innerHTML
          }
        });
      } else {
        console.log('No text selected');
        statusDiv.textContent = "No text selected. Please select the job description.";
      }
    });
  }

  function getPageText(pdf, pageNum) {
    return pdf.getPage(pageNum).then(function(page) {
      return page.getTextContent();
    }).then(function(textContent) {
      return textContent.items.map(item => item.str).join(' ');
    });
  }
});
