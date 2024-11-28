[![Run CNN Model Tests](https://github.com/Ezhirko/Creative-AI-apps/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/Ezhirko/Creative-AI-apps/actions/workflows/ci_cd.yml)

# MNIST Digit Classification - Fine-Tuned CNN Models

This repository contains the results of experiments conducted to build a lightweight Convolutional Neural Network (CNN) model for the MNIST digit classification dataset. The goal was to create a model with less than **20,000 parameters**, achieving a **test accuracy of at least 99.4%**, while ensuring minimal overfitting.

---

## **Dataset**
- **Train Set**: 60,000 images
- **Test Set**: 10,000 images
- Images: 28x28 grayscale digits (0–9)

---

## **Experiments**
The models were iteratively fine-tuned across three versions to meet the requirements.

### **Version 1**
- **Parameters**: 20,930 (slightly above the requirement)
- **Features**:
  - Basic CNN architecture
  - No Batch Normalization or Dropout layers
- **Performance**:
  - Did not achieve the required test accuracy of 99.4% after 10 epochs
  - Overfitting observed: training accuracy > test accuracy
- **Model Summary**:  
  *Insert Version 1 Model Summary Image Here*

---

### **Version 2**
- **Parameters**: 17,994 (meets the requirement)
- **Features**:
  - Added **Batch Normalization** between CNN layers and activation functions
- **Performance**:
  - Achieved the required test accuracy of 99.4% within 3–4 epochs
  - Occasional overfitting observed: training accuracy > test accuracy
- **Model Summary**:  
  *Insert Version 2 Model Summary Image Here*

---

### **Version 3 (Final Model)**
- **Parameters**: 17,994 (meets the requirement)
- **Features**:
  - Added **Batch Normalization** and **Dropout** layers between CNN layers and activation functions
- **Performance**:
  - Achieved the required test accuracy of 99.4% within 3–4 epochs
  - More generalized outcomes with minimal overfitting
  - Faster convergence
- **Model Summary**:  
  *Insert Version 3 Model Summary Image Here*

---

## **Conclusion**
- **Version 3** is the optimal model, offering:
  - Less than 20,000 parameters
  - Balanced use of **Batch Normalization** and **Dropout**
  - Faster convergence
  - Generalized performance with minimal overfitting

---

## **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/mnist-cnn-finetune.git](https://github.com/Ezhirko/Creative-AI-apps.git

2. Navigate to the project directory:
   ```bash
   cd TunningCNN
   
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt

4. Train the models:
   ```bash
   python train.py

5. Run the Unit test:
   ```bash
   python test_pipeline.py
