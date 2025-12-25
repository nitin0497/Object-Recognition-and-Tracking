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

## Part 1: Object Detection and Tracking

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

For circle *i* in frame *t* and circle *j* in frame *t + 1*:

```
Cost(i, j) = α · ||C_i − C_j||² + β · (r_i − r_j)²
```

**Where:**
- `C_i` and `C_j` are the centers of circles *i* and *j* in consecutive frames  
- `||C_i − C_j||` denotes the Euclidean distance between the two circle centers  
- `r_i` and `r_j` are the corresponding radii  
- `α = 0.1` controls the contribution of spatial distance  
- `β = 0.9` enforces strict radius matching


---

### Tracking Results

| Metric | Performance |
|------|------------|
| Tracking accuracy | > 80% |
| Detection rate | ~87% per frame |
| ID switches | < 30 across 1530 bubble instances |

**Qualitative Results**
- Stable tracking with minimal drift
- Clear visualisation of detected and predicted objects

### Shape Tracking Demo

[![Bubble Tracking Demo](Bubble%20Tracking%20CV/output/Final_bubble_tracking_output.gif)](Bubble%20Tracking%20CV/output/Final_bubble_tracking_output.mp4)



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

<img width="778" height="812" alt="image" src="https://github.com/user-attachments/assets/447e2cea-f3d3-498e-8d6d-4597f73ad4e7" />


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



