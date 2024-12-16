# CNN-Based Skin Cancer Detection

This project involves building a Convolutional Neural Network (CNN) model to accurately detect melanoma, a severe form of skin cancer that accounts for 75% of skin cancer deaths. Early detection of melanoma can significantly improve treatment outcomes, and this project aims to assist dermatologists by automating melanoma detection through image analysis.



## Table of Contents
* [Problem Statement]
* [Dataset]
* [Features of the Project]
* [Prerequisites]
* [How to Run the Project]
* [Results]
* [Future Improvements]
* [References]
* [Analysis of the Project]
* [Contact]

<!-- You can include any other section that is pertinent to your problem -->

## Problem Statement
To build a CNN-based model capable of detecting melanoma with high accuracy, using a dataset of skin lesion images classified into different skin cancer types.

## Dataset
The dataset used in this project contains **2,357 images** of skin lesions across **9 categories of skin cancer types**. It is organized into:
- **Train Subdirectory:** Images used for training the model.
- **Test Subdirectory:** Images used for evaluating the model's performance.

The dataset structure includes the following categories:
- Melanoma
- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanocytic Nevus
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesion
- Benign Keratosis

## Features of the Project
1. **Data Understanding and Preprocessing:**
   - Data loading and visualization.
   - Addressing class imbalances.
   - Implementing data augmentation to improve model generalization.

2. **Model Architecture:**
   - Use of TensorFlow and Keras for building the CNN.
   - Layers include convolutional, pooling, normalization, and dense layers.

3. **Training and Evaluation:**
   - Evaluation of the model's performance on training and validation datasets.
   - Achieving a maximum accuracy of approximately 75%.
   - Overcoming overfitting through regularization techniques.

4. **Libraries Used:**
   - TensorFlow
   - NumPy
   - Matplotlib
   - Pandas
   - PIL (Python Imaging Library)

## Prerequisites
To run this project, ensure you have the following installed:
- Python 3.7+
- TensorFlow 2.0+
- Required Python libraries (`pip install tensorflow matplotlib numpy pandas pillow`)

## How to Run the Project
1. Clone the repository:
   ```bash 
   git clone https://github.com/MangaiMaharajan/CNNAssignment
   cd skin-cancer-detection
   ```
2. Ensure the dataset is properly structured in `Train` and `Test` subdirectories.
3. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Starter_code_Assignment_CNN_Skin_Cancer.ipynb
   ```
5. Run the cells sequentially to preprocess the data, train the model, and evaluate performance.

## Results
- The model achieved a maximum accuracy of **~75%** on both training and validation datasets.
- Implementing data augmentation and addressing class imbalance helped improve accuracy and reduce overfitting.

## Future Improvements
- Fine-tune the model for higher accuracy.
- Use larger and more diverse datasets for better generalization.
- Explore advanced architectures like ResNet or EfficientNet.

## References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [ISIC Dataset Source](https://www.isic-archive.com/)

---
## Analysis of the Project
1. Problem and Objectives

    The primary objective of this project is to develop a CNN model for accurate detection of melanoma, a severe and potentially fatal type of skin cancer.
    Early detection through automated systems has the potential to significantly reduce manual diagnostic efforts for dermatologists, improving patient outcomes.

2. Dataset and its Challenges

    Dataset Overview: The dataset contains 2,357 images spanning 9 categories of skin cancer types, including both benign and malignant cases.
    Challenges Identified:
        Class imbalance: Some cancer types might dominate others, leading to biased predictions.
        Limited data: Data augmentation is required to simulate more diverse training samples and avoid overfitting.

3. Technical Implementation

    Preprocessing:
        Data augmentation (e.g., rotation, flipping) is employed to enhance generalization and balance the dataset.
        Efficient data loading and visualization were achieved using TensorFlow utilities.
    Model Architecture:
        A custom CNN is designed with convolutional, pooling, normalization, and dense layers.
        The use of TensorFlow and Keras ensures flexibility in experimentation and scalability.
    Training and Evaluation:
        Achieved a maximum accuracy of ~75%, a strong baseline for melanoma detection.
        Overfitting was mitigated through regularization and data augmentation.

4. Strengths of the Project

    Practical Relevance: Targets a critical healthcare issue with significant societal impact.
    Comprehensive Workflow: Includes data preprocessing, augmentation, training, and evaluation steps.
    Flexibility: The modular architecture and choice of TensorFlow allow for easy scalability and upgrades.

5. Areas for Improvement

    Accuracy and Performance: The current accuracy (~75%) is promising but leaves room for improvement, particularly for clinical applications where higher precision is critical.
    Advanced Architectures: Experimenting with pre-trained models like ResNet, EfficientNet, or MobileNet could lead to better results.
    Dataset Diversity: Incorporating larger and more diverse datasets could improve the model's robustness and reduce bias.
    Metrics: Incorporating precision, recall, and F1-score metrics can provide a more detailed understanding of the model's performance, especially in imbalanced datasets.

6. Impact and Future Scope

    Impact: This project lays the foundation for an AI-driven diagnostic tool for melanoma detection, with the potential to support dermatologists and improve diagnostic accuracy.
    Future Scope:
        Deploy the model in real-world settings, such as mobile or web applications.
        Validate the model on unseen datasets to assess its generalizability.
        Collaborate with healthcare professionals to refine the system for clinical adoption.
*This project aims to assist dermatologists by providing an automated solution for melanoma detection, potentially reducing manual diagnosis effort.*


## Contact
Created by [@MangaiMaharajan] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
