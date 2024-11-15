from flask import Flask, render_template, request, jsonify,send_from_directory
import subprocess
import json
import threading

app = Flask(__name__)

# Initialize an empty dictionary to store the training results
training_results = {
    "model1": {"loss": [], "accuracy": [], "epoch":[]},
    "model2": {"loss": [], "accuracy": [], "epoch":[]}
}

model_summary = {
    "model1": {"layers":[], "tlt_params":[]},
    "model2": {"layers":[], "tlt_params":[]}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plot/<filename>')
def plot(filename):
    return send_from_directory('static', filename)

def parse_loss_epoch_accuracy(output_line):
    # Extract loss and accuracy from the output string
    # Implement your own logic to extract these values
    # For example, using regex or string splitting:
    parts = output_line.split(", ")
    loss = float(parts[0].split("=")[1].strip())
    accuracy = float(parts[1].split("=")[1].strip())
    epoch = float(parts[2].split("=")[1].strip())
    return loss, accuracy, epoch

def parse_model_summary(output_line):
    parts = output_line.split(",  ")
    layers = parts[0].split("=")[1].strip()
    total_parms = parts[1].split("=")[1].strip()
    return layers, total_parms

def train_model_in_background(kernel1, kernel2,
                              optimizer, batchSize,
                              epochs, learningRate):
    try:
        print('train_model_in_background')
        # Run the training script with subprocess and capture the results in real-time
        process1 = subprocess.Popen(
            ["python", "scripts/train.py", kernel1, kernel2, optimizer, batchSize, epochs, learningRate],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Read stdout and stderr
        for line in iter(process1.stdout.readline, b''):
            decoded_line = line.decode("utf-8")
            print(f"Printing the decoded line: {decoded_line}")
            # Assuming that the loss and accuracy are printed in a specific format
            if "loss" in decoded_line and "accuracy" and "epoch" in decoded_line:
                # Parse the loss and accuracy values from the output
                # You need to define a proper regex or string manipulation to extract these values
                if "Model 1" in decoded_line:
                    # Example: Model 1: loss = 0.2, accuracy = 94.5
                    loss, accuracy, epoch = parse_loss_epoch_accuracy(decoded_line)
                    training_results["model1"]["loss"].append(loss)
                    training_results["model1"]["accuracy"].append(accuracy)
                    training_results["model1"]["epoch"].append(epoch)
                elif "Model 2" in decoded_line:
                    loss, accuracy, epoch = parse_loss_epoch_accuracy(decoded_line)
                    training_results["model2"]["loss"].append(loss)
                    training_results["model2"]["accuracy"].append(accuracy)
                    training_results["model2"]["epoch"].append(epoch)

            if "layers" in decoded_line and "tlt_params" in decoded_line:
                if "Model 1" in decoded_line:
                    layer, total_parms = parse_model_summary(decoded_line)
                    model_summary["model1"]["layers"].append(layer)
                    model_summary["model1"]["tlt_params"].append(total_parms)
                elif "Model 2" in decoded_line:
                    layer, total_parms = parse_model_summary(decoded_line)
                    model_summary["model2"]["layers"].append(layer)
                    model_summary["model2"]["tlt_params"].append(total_parms)

            # You can also log errors if necessary
            if process1.stderr:
                error_line = process1.stderr.readline().decode("utf-8")
                if error_line:
                    print(f"Error: {error_line}")
    except Exception as ex:
        print(f"Some Exception:{ex}")

@app.route('/train_models', methods=['POST'])
def train_models():
    data = request.json
    kernel1 = data['kernel1']
    kernel2 = data['kernel2']
    optimizer = data['optimizer']
    batchSize = data['batchSize']
    epochs = data['epochs']
    learningRate = data['learningRate']

    # Start training in a separate thread so it doesn't block the main Flask server
    threading.Thread(target=train_model_in_background, args=(kernel1,kernel2,optimizer,batchSize,epochs,learningRate)).start()

    return jsonify({"message": "Training started for both models."}), 200

@app.route('/get_results')
def get_results():
    # Return the training results in JSON format
    return jsonify(training_results)

@app.route('/get_Model_summary')
def get_Model_summary():
    # Return the training results in JSON format
    return jsonify(model_summary)

@app.route('/test_samples')
def get_test_samples():
    try:
        with open('static/test_samples.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'error': 'No test samples available yet'}

if __name__ == "__main__":
    app.run(debug=True)
