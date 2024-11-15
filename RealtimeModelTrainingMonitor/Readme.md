# MNIST CNN Model Comparison Application 🚀

This application allows users to compare two different CNN models for MNIST classification. Users can specify kernel sizes, training parameters, and view training results in real time. Model summaries, accuracies, losses, and training plots are displayed on an interactive HTML interface.

## Features
- Compare two different CNN architectures on the MNIST dataset.
- Set kernel sizes, optimizer, batch size, learning rate, and number of epochs.
- Display model summaries, layer parameters, accuracy, and loss in real-time.
- Visualize accuracy and loss curves after training.

## Requirements
- Python 3.x
- Flask
- PyTorch
- Bootstrap (for UI styling)
- JavaScript (for real-time updates)

## Additional Python Libraries

Install these using pip:
- flask
- torch
- torchvision

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-cnn-compare.git
   cd mnist-cnn-compare
2. **Install Dependencies** Make sure you have all required packages installed. You can install dependencies by running:
   ```bash
   pip install -r requirements.txt
3. **Set Up Static and Template Files** Ensure you have the following structure:
   ```graphql
   ├── static/
    │   ├── styles.css           # Custom CSS styles
    │   ├── scripts.js           # JavaScript file for real-time updates
    ├── templates/
    │   ├── index.html           # Main HTML page for input and results display

4. **Run the Flask server** Start the server by running:
   ```bash
   python server.py

   The application will run on http://127.0.0.1:5000.
   
5. **Access the Application** Open your web browser and go to:
   ```bash
   http://127.0.0.1:5000

---

## How to Use the Application
1. **Enter Model Parameters**
In Section 1, input kernel sizes for each model as comma-separated values. For example, 16,32,64 for Model 1 and 8,8,8 for Model 2.

2. **Set Training Parameters**
In Section 2, choose an optimizer (Adam or SGD), specify batch size, learning rate, and the number of epochs.

3. **Start Training**
Click "Train and Compare Models." The application will start training both models in alternating epochs, and results will be updated in real-time on the HTML page.

4. **View Results**
Model summaries, accuracy, and loss for each model will be displayed, along with accuracy and loss plots after training.

## Example Output
- **Model Summaries:** Detailed layer-wise information.
- **Training Plots:** Visual representations of accuracy and loss over time.
_ **Performance Metrics:** Final accuracy and loss values for each model.

## Troubleshooting
If you encounter any issues, try the following:
- Ensure all dependencies are installed.
- Verify that the server.py file is running.
- Check the console for any error messages in the terminal or web browser.
  
## License
This project is licensed under the MIT License.

[Check out the demo on YouTube](https://youtu.be/NmfjHihXJ-k)
