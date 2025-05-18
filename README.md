# Acne Analysis Model

![Acne Analysis](https://img.shields.io/badge/Status-Progress-orange) ![Python](https://img.shields.io/badge/Python-3.9-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![License](https://img.shields.io/badge/License-MIT-green)

This project aims to develop a **deep learning model** capable of analyzing facial images to detect common skin conditions such as **Blackheads**, **Cyst**, **Papules**, **Pustules**, and **Whiteheads**. The system not only identifies these conditions but also provides personalized recommendations for each detected issue, making it a helpful tool for users seeking insights into their skin health.

---

## Overview

The **Acne Analysis Model** leverages **EfficientNetB0**, a state-of-the-art deep learning architecture, to classify facial images into five categories of acne conditions. The model is trained on a custom dataset and incorporates advanced techniques like **data augmentation**, **class weights**, and **dropout regularization** to ensure robust performance. Additionally, the project includes a **confusion matrix** and **F1-score analysis** to evaluate model performance.

check out the live demo 🚀 https://acne-analysis-model-adtzbokgpsv8gcnv6zhark.streamlit.app/

---

## Features

- **Multi-Class Classification**: Detects five types of acne conditions: Blackheads, Cyst, Papules, Pustules, and Whiteheads.
- **Personalized Recommendations**: Provides tailored skincare advice based on detected conditions.
- **Data Augmentation**: Enhances the dataset with techniques like flipping, rotation, brightness adjustment, and Gaussian noise.
- **Class Weights**: Addresses class imbalance during training.
- **Confusion Matrix & Metrics**: Evaluates model performance using metrics like accuracy, precision, recall, and F1-score.
- **Custom Image Prediction**: Allows users to upload their own images for real-time predictions.

---

## Dataset

### Dataset Details
- **Source**: roboflow (Acne Scanner) | Kaggle ([Acne_dataset](https://www.kaggle.com/datasets/anshchauhan248/acne-dataset))
- **Total Images**: 4,607
  - Train: 2,768
  - Validation: 921
  - Test: 918
- **Classes**: 5 balanced categories:
  - Blackheads
  - Cyst
  - Papules
  - Pustules
  - Whiteheads
- **Image Size**: Resized to 150x150 pixels for consistency.

---

## Results

### Performance Metrics
- **Test Accuracy**: 96.84%
- **Test Precision**: 97.05%
- **Test Recall**: 96.62%
- **F1-Score (Macro Avg)**: 96.74%

### Confusion Matrix
The confusion matrix shows minimal misclassifications across all classes, with **Pustules** and **Whiteheads** achieving **100% confidence** in test cases.

---

## Credits

This project was developed collaboratively by:

- [**Ansh Chauhan**](https://github.com/Anshchauhanhub): Model creation and optimization.
- [**Tushar Rajput**](https://github.com/iam-tsr): Testing and deployment.
- [**Harshit Saini**](https://github.com/sainiharshit322): Documentation and testing.

Special thanks to the open-source community for their tools and libraries that made this project possible.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

Feel free to contribute to this project or reach out with any questions! 🚀
