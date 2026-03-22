# Automated Detection of Thoracic Pathologies from Chest X-rays

An automated deep learning system designed to classify 20 different thoracic pathologies from chest X-ray images. This project implements **EfficientNetB0** via transfer learning to address challenges in medical imaging, specifically focusing on extreme class imbalance and clinical misclassification costs.

##  Project Overview
Chest X-ray imaging is a vital diagnostic tool, but manual analysis is time-consuming and subject to human variability. This project automates the detection of conditions like Pneumonia, Atelectasis, and Effusion.

### Key Challenges:
*   **Multi-class Classification:** 20 distinct pathology classes.
*   **Extreme Class Imbalance:** The most frequent class has ~6,800x more samples than the rarest.
*   **Asymmetric Costs:** False Negatives (missing a disease) are penalized 5x more heavily than False Positives.

---

##  Methodology

### Model Architecture
*   **Backbone:** EfficientNetB0 (Pretrained on ImageNet)
*   **Reasoning:** EfficientNet provides a superior balance of speed and accuracy. Transfer learning was essential due to the limited samples in rare pathology classes (e.g., Pneumomediastinum).
*   **Layers:** 
    *   Frozen EfficientNetB0 Base
    *   GlobalAveragePooling2D
    *   Dense (128 units, ReLU)
    *   Dropout (0.3)
    *   Dense (20 units, Softmax)

### Data Processing
*   **Input Size:** Resized from 384x384 to 224x224 (EfficientNet requirement).
*   **Normalization:** Pixel values scaled to [0, 1].
*   **Augmentation:** Horizontal flips, 10° rotations, 10% zooms, and 5% shifts. (Aggressive augmentation was avoided to preserve clinical structural meaning).
*   **Splitting:** 80/20 stratified split to maintain class distribution.

---

##  Experiments & Results


| Experiment | Configuration | Key Change | Score |
| :--- | :--- | :--- | :--- |
| **Exp 1** | LR: 0.001 + Class Weights | Initial run (contained label sorting bug) | -34.0 |
| **Exp 2** | LR: 0.001 + Capped Weights | Bug fixed; unstable training due to high weights | -5.80 |
| **Exp 3** | **LR: 0.0001 (No Weights)** | **Stable training; best performance** | **-4.72** |

### Critical Learnings:
1.  **Pipeline Integrity:** A bug in Keras' alphabetical sorting of string labels ('10' appearing before '2') initially caused a total mismatch in class assignments.
2.  **Weight Stability:** While class weights are theoretically ideal for imbalance, extreme weights (up to 510x) destabilized the optimizer. A lower learning rate without weights proved more robust.

---

##  How to Use
1.  **Clone the Repo:**
    ```bash
    git clone https://github.com
    ```
2.  **Install Dependencies:**
    ```bash
    pip install tensorflow pandas scikit-learn matplotlib
    ```
3.  **Run Training:**
    Execute the Jupyter notebook or python script provided in the `/src` folder.

---

## References
1.  **EfficientNet:** Tan, M., & Le, Q. V. (2019). Rethinking Model Scaling for CNNs.
2.  **ChestX-ray8:** Wang, X. et al. (2017). Hospital-scale Chest X-ray Database.
3.  **Frameworks:** [TensorFlow](https://www.tensorflow.org) & [Keras](https://keras.io).
