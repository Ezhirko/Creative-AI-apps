console.log('Background script loaded');

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log('Message received in background script:', request.action);
  if (request.action === "analyzeJobDescription") {
    console.log('Analyze job description request received');
    analyzeJobDescription(request.jobDescription, sendResponse);
    return true; // Keeps the message channel open for asynchronous response
  }
});

function getAuthToken(callback) {
  chrome.identity.getAuthToken({ interactive: true }, function (token) {
    if (chrome.runtime.lastError) {
      console.error(chrome.runtime.lastError);
      callback(null);
    } else {
      callback(token);
    }
  });
}

function analyzeJobDescription(jobDescription, sendResponse) {
  chrome.storage.local.get('resumeText', function (data) {
    if (chrome.runtime.lastError) {
      console.error('Error retrieving resume text:', chrome.runtime.lastError);
      sendResponse({ result: "Error: Unable to retrieve resume text" });
      return;
    }

    const resumeText = data.resumeText;
    if (!resumeText) {
      console.log('No resume text found in storage');
      sendResponse({ result: "Error: No resume uploaded" });
      return;
    }

    console.log('Resume text retrieved from storage');

    console.log('Sending request to Google API');
    fetch('http://localhost:5000/concatenate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        JobDescription: jobDescription,
        ResumeText: resumeText
      }),
    })
      .then(response => response.json())
      .then(data => {
        console.log('API response received:', data);
        const parsedResult = JSON.parse(data.full_name);
        const formattedResult = formatResultAsTable(parsedResult);
        sendResponse({ result: formattedResult });
      })
      .catch(error => {
        console.error('Error calling Google API:', error);
        sendResponse({ result: "Error: Unable to analyze job description. " + error.message });
      });
  });
}

function formatResultAsTable(result) {
  let tableHtml = '<table style="width:100%; border-collapse:collapse; margin-top:10px;">';
  
  // JD Match row
  tableHtml += `
    <tr>
      <td style="border:1px solid #ddd; padding:8px; font-weight:bold; background-color:#f2f2f2;">JD Match</td>
      <td style="border:1px solid #ddd; padding:8px;">${result['JD Match']}</td>
    </tr>
  `;
  
  // Missing Keywords row
  tableHtml += `
    <tr>
      <td style="border:1px solid #ddd; padding:8px; font-weight:bold; background-color:#f2f2f2;">Missing Keywords</td>
      <td style="border:1px solid #ddd; padding:8px;">${result['MissingKeywords'].join(', ')}</td>
    </tr>
  `;
  
  tableHtml += '</table>';
  return tableHtml;
}
