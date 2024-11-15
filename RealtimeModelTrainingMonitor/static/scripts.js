// Periodically update the accuracy and loss plots
function refreshPlots() {
    document.getElementById('accuracy_plot').src = '/plot/accuracy_plot.png?' + new Date().getTime();
    document.getElementById('loss_plot').src = '/plot/loss_plot.png?' + new Date().getTime();

    fetch('/get_results')
    .then(response => response.json())
    .then(data => {
        // Update Model 1 metrics
        document.getElementById('model1_accuracy').textContent = data.model1.accuracy.slice(-1)[0] || 0;
        document.getElementById('model1_loss').textContent = data.model1.loss.slice(-1)[0] || 0;
        document.getElementById('model1_epoch').textContent = data.model1.epoch.slice(-1)[0] || 0;

        // Update Model 2 metrics
        document.getElementById('model2_accuracy').textContent = data.model2.accuracy.slice(-1)[0] || 0;
        document.getElementById('model2_loss').textContent = data.model2.loss.slice(-1)[0] || 0;
        document.getElementById('model2_epoch').textContent = data.model2.epoch.slice(-1)[0] || 0;
    })
    .catch(error => console.error('Error fetching results:', error));
}

setInterval(refreshPlots, 3000);

function get_model_summary() {
    fetch('/get_Model_summary')
    .then(response => response.json())
    .then(data => {
        // Update Model 1 metrics
        document.getElementById('model1_layers').innerHTML = data.model1.layers;
        document.getElementById('model1_params').innerHTML = data.model1.tlt_params;

        // Update Model 2 metrics
        document.getElementById('model2_layers').innerHTML = data.model2.layers;
        document.getElementById('model2_params').innerHTML = data.model2.tlt_params;
    })
    .catch(error => console.error('Error fetching results:', error));
}

setInterval(get_model_summary, 15000);

document.getElementById("modelForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent form from submitting traditionally

    // Collect kernel sizes
    const kernel1 = document.getElementById("kernel1").value;
    const kernel2 = document.getElementById("kernel2").value;

    // Collect training parameters
    const optimizer = document.querySelector('input[name="optimizer"]:checked').value;
    const batchSize = document.getElementById("batchSize").value;
    const epochs = document.getElementById("epochs").value;
    const learningRate = document.getElementById("learningRate").value;

    // Prepare data to send
    const data = {
        kernel1: kernel1,
        kernel2: kernel2,
        optimizer: optimizer,
        batchSize: batchSize,
        epochs: epochs,
        learningRate: learningRate
    };

    // Send data to the server
    fetch('/train_models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(responseData => {
        console.log('Success:', responseData);
        alert("Training has been started!")
        // Handle server response here (e.g., update UI with results)
    })
    .catch(error => console.error('Error:', error));
});

// Handle form submission
// document.getElementById("kernelForm").onsubmit = function(event) {
//     event.preventDefault(); // Prevent form from submitting the traditional way

//     const kernel1 = document.getElementById("kernel1").value;
//     const kernel2 = document.getElementById("kernel2").value;

//     fetch('/train_models', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ kernel1, kernel2 })
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.message) {
//             alert(data.message);
//         }
//     })
//     .catch(error => console.error('Error:', error));
// };

function updateTestSamples() {
    fetch('/test_samples')
        .then(response => response.json())
        .then(data => {
            if (!data.error && data.model1 && data.model2) {
                // Update Model 1 samples
                const grid1 = document.getElementById('samples-grid-1');
                grid1.innerHTML = '';
                data.model1.forEach(sample => {
                    const div = document.createElement('div');
                    div.className = 'sample-box';
                    const correct = sample.predicted === sample.actual;
                    div.innerHTML = `
                        <img src="data:image/png;base64,${sample.image}" />
                        <p class="${correct ? 'correct' : 'incorrect'}">
                            Pred: ${sample.predicted}<br>
                            Act: ${sample.actual}
                        </p>
                    `;
                    grid1.appendChild(div);
                });

                // Update Model 2 samples
                const grid2 = document.getElementById('samples-grid-2');
                grid2.innerHTML = '';
                data.model2.forEach(sample => {
                    const div = document.createElement('div');
                    div.className = 'sample-box';
                    const correct = sample.predicted === sample.actual;
                    div.innerHTML = `
                        <img src="data:image/png;base64,${sample.image}" />
                        <p class="${correct ? 'correct' : 'incorrect'}">
                            Pred: ${sample.predicted}<br>
                            Act: ${sample.actual}
                        </p>
                    `;
                    grid2.appendChild(div);
                });
            }
        });
}
setInterval(updateTestSamples, 12000);
