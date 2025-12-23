# Object Recognition and Tracking using Computer Vision


This project demonstrates two complementary computer vision pipelines:

1. **Real-time shape detection and tracking** using classical computer vision  
2. **Handwritten digit recognition** using a Convolutional Neural Network (CNN)

The project highlights how **interpretable classical vision methods** and **deep learning models** can be combined to build efficient, accurate, and explainable computer vision systems.

---

## Project Overview

### Part 1: Real-Time Shape Recognition and Tracking
- Detects and tracks **circles (bubbles)** and basic geometric shapes in video
- Maintains **consistent object IDs** across frames
- Handles **occlusion**, **missed detections**, and **motion**
- Designed to be **lightweight, interpretable, and real-time capable**

### Part 2: Handwritten Digit Recognition
- CNN trained on the **MNIST dataset**
- Near-perfect accuracy on standard MNIST test data
- Strong generalization to **custom external handwritten images**
- Demonstrates practical preprocessing and segmentation techniques

---

## Technologies Used

### Classical Computer Vision
- Canny Edge Detection  
- Hough Circle Transform  
- CIELAB color space  
- Hungarian Algorithm (assignment problem)

### Deep Learning
- Convolutional Neural Networks (CNNs)  
- TensorFlow / Keras  
- MNIST dataset  

---

## Part 1: Shape Detection and Tracking

### Pipeline Overview

1. **Edge Detection**
   - Canny edge detector
   - RGB → CIELAB conversion for robustness to lighting
2. **Circle Detection**
   - Hough Circle Transform (OpenCV)
   - Extracts center coordinates and radius
3. **Tracking Across Frames**
   - Cost matrix based on:
     - Euclidean distance between centers
     - Difference in radius
   - Optimal matching via Hungarian algorithm
4. **Occlusion Handling**
   - Missing detections interpolated using past and future frames
5. **Outputs**
   - Annotated video with object IDs
   - CSV file with `(x, y, radius, ID, frame)`

### Matching Cost Function

\[
Cost_{i,j} = \alpha \cdot \|C_i - C_j\|^2 + \beta \cdot (r_i - r_j)^2
\]

Where:
- \(C_i, C_j\): centers of detected circles
- \(r_i, r_j\): radii
- \(\alpha = 0.1\)
- \(\beta = 0.9\) (enforces strict radius consistency)

---

### Tracking Results

| Metric | Performance |
|------|------------|
| Tracking accuracy | > 80% |
| Detection rate | ~87% per frame |
| ID switches | < 30 across 1530 bubble instances |

**Qualitative Results**
- Stable tracking with minimal drift
- Clear visualization of detected and predicted objects

---

## Part 2: Digit Recognition using CNN

### Dataset
- **MNIST**
  - 60,000 training images
  - 10,000 test images
- **Custom external image**
  - 100 handwritten digits arranged in rows

---

### CNN Architecture

| Layer | Details |
|------|--------|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Output | 10 units, Softmax |

### Training Configuration
- Optimizer: Adam  
- Loss: Categorical Cross-Entropy  
- Epochs: 20  
- Batch size: 32  

---

### Digit Recognition Results

| Dataset | Accuracy |
|-------|----------|
| MNIST test set | ~99% |
| External handwritten image | 97% (97/100 correct) |

**Key Observations**
- Strong generalization beyond MNIST
- Preprocessing (thresholding, centering, padding) is crucial

---

## Applications

### Object Tracking
- Precision agriculture
- Industrial monitoring
- Traffic management
- Biological and laboratory experiments

### Digit Recognition
- Postal and ZIP code recognition
- Bank cheque processing
- Form and document digitization
- Automated exam grading
- Financial data entry systems

---



