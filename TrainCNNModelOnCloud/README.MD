# CNN Model Training Experiments on MNIST Dataset

## Goal
The goal of this project is to achieve **99.4% test accuracy in 15 epochs** using a **CNN network with less than 8000 parameters**. 

This repository contains multiple experiments conducted on the MNIST dataset to achieve the desired results. A CNN model based on the **Squeeze and Excite (SE) model structure** was implemented and iteratively improved across 5 versions.

---

## Experiments

### **Version 1: Basic Skeleton**
#### Target
- Develop the basic skeleton of the Squeeze and Excite model structure.

#### Results
- **Parameters**: 13,832  
- **Best Train Accuracy**: 99.16%  
- **Best Test Accuracy**: 98.63%  

#### Analysis
- Model had far more parameters than the required threshold of 8000.  
- Overfitting observed.  
- Need to create a lighter model in the next version.

---

### **Version 2: Lighter Model**
#### Target
- Make the model lighter by replacing the last convolution layer with a **Global Average Pooling (GAP)** layer.

#### Results
- **Parameters**: 7,432  
- **Best Train Accuracy**: 96.04%  
- **Best Test Accuracy**: 96.30%  

#### Analysis
- GAP layer reduced the parameters from 13,832 to 7,432.  
- Overfitting reduced with fewer parameters.  
- However, the reduction in network capacity led to lower accuracies.  

---

### **Version 3: Batch Normalization**
#### Target
- Add **Batch Normalization** in each convolution block (except the last layer) to normalize kernel outputs and improve convergence.

#### Results
- **Parameters**: 7,612  
- **Best Train Accuracy**: 99.12%  
- **Best Test Accuracy**: 99.05%  

#### Analysis
- Batch Normalization improved convergence and pushed accuracies higher.  
- Slight overfitting was observed.  
- Further tuning with Learning Rate Schedulers in the next version.

---

### **Version 4: Learning Rate Scheduler**
#### Target
1. Increase the learning rate to **0.1** for faster convergence.  
2. Add **StepLR** to reduce the learning rate dynamically (Step size = 7, Gamma = 0.1).

#### Results
- **Parameters**: 7,612  
- **Best Train Accuracy**: 99.63%  
- **Best Test Accuracy**: 99.19%  

#### Analysis
- Increased learning rate significantly improved accuracy.  
- Overfitting persisted as train accuracies were higher than test accuracies in most epochs.  
- Apply Image Augmentation in the next version to regularize the model.

---

### **Version 5: Image Augmentation and StepLR Scheduler**
#### Target
1. Add **Image Augmentation** (random rotation in the range of -15° to 15° and fill with 1).  
2. Increase the learning rate to **0.1** for faster convergence.  
3. Add **StepLR** with Step size = 11, Gamma = 0.2.  

#### Results
- **Parameters**: 7,612  
- **Best Train Accuracy**: 99.16%  
- **Best Test Accuracy**: 99.44%  

#### Analysis
- Image augmentation regularized the model and resolved overfitting issues.  
- LR scheduling tuned with StepSize = 11 and Gamma = 0.2 gave the best consistency, achieving ≥99.4% accuracy in the last three epochs.  
- Experiment achieved a maximum of 99.44% test accuracy with less than 8000 parameters.

#### Logs

---

## Repository Structure
- **Version 1**: Basic skeleton implementation.
- **Version 2**: Lighter model with GAP layer.
- **Version 3**: Model with Batch Normalization.
- **Version 4**: Learning Rate Scheduler implementation.
- **Version 5**: Final model with Image Augmentation and StepLR Scheduler.

---

## Dataset
The experiments were conducted on the MNIST dataset, which consists of grayscale images of handwritten digits (0-9).

---

## Conclusion
This project demonstrates how iterative improvements, including parameter reduction, Batch Normalization, Learning Rate Schedulers, and Image Augmentation, can help achieve high accuracy while meeting specific constraints like parameter count. The final model successfully met the target with **99.44% test accuracy** in 15 epochs using a CNN network with less than **8000 parameters**.

---

## Future Work
- Explore further optimizations to reduce training time.  
- Extend experiments to other datasets like FashionMNIST or CIFAR-10.  
- Investigate deployment of the model on edge devices.

---
