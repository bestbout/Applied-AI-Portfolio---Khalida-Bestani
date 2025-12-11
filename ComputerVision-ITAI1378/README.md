# Museum Theft Prevention System  
**Real-Time Computer Vision for Protecting Artifacts**

## Problem Statement
Museums lost over $5 billion last year to thefts and damages. Traditional security systems are too expensive and cannot monitor every exhibit at all times and respond fast enough to prevent loss or damage. There is an urgent need for affordable real-time security that prevents loss and damage before it happens for institutions of all sizes.

## Solution Overview
THEFT PREVENTION using a Computer Vision Solution that spots when someone is getting too close to the exhibits or is acting suspicious. The model identifies threatening or sketchy behavior, filters out false alarms while verifying actual threats to then alert security in real time. Object Detection (OD) and Action Recognition (AR) to detect people and identify their behavior. Just watching people is not enough. OD to detect people and products. AR to identify suspicious behavior, "theft" and track the suspects throughout the building(s). Neuroevolution to evolve from new data and improve continuously. Evolutionary Video Deep Learning learns patterns that evolves better architecture or strategies for learning those patterns. Deep spatiotemporal learning and continual learning through: Convolutional 3D Neural Networks, Recurrent Neural Networks and Video Transformers.

## Technical Approach
### Core Techniques
Object Detection & Classification, Action Recognition & Spatiotemporal Analysis, Deep 3D Convolutional Networks, RNNs, and Video Transformers, Neuroevolution for evolving better architectures, Continual learning to adapt to new museum environments.

### Main Model
YOLOv8 (Ultralytics) – real-time object detection and behavior classification. Runs at 30+ FPS with no lag.

### Framework & Libraries
PyTorch + torchvision – core deep learning framework, Ultralytics YOLOv8 – object detection and tracking, OpenCV – video processing and frame extraction, PyTorchVideo – video sequence analysis, Transformers (CLIP & BLIP) – reduce false alarms by understanding context and intent, scikit-learn – evaluation metrics and confusion matrices, Matplotlib & Seaborn – visualization of results.

## Dataset Plan
We combine multiple sources to build a robust, bias-free dataset. A strong foundation is built for our behavioral theft detection system using multiple sources to avoid bias by combining large-scale datasets for training and specialized behavioral data for rare theft events used for fine-tuning. UCF-101 has more than 13,000 videos that cover 101 action categories that has demonstrated 96% accuracy with YOLO architectures which brings us to believe is ideal for the initial transfer learning. UCF-Crime or Anomaly Detection datasets has actual suspicious behavior videos. COCO has people, bags, products (good starting point). Roboflow for "shoplifting detection" or "retail security" datasets. Unity Perception synthetic datasets that allow us to generate correct labels that would be difficult to get in actual museums. Can build "scenes" for bounding box and labeler to capture boundaries, person crossing boundaries, hands, gestures, artifacts, reaching, bags, etc. Unity Perception open-source package provides synthetic data for computer vision by training the synthetic datasets on a 2D object detection model. UFC-Crime Dataset for anomaly detection to identify and label abnormal and normal behavior. The system learns to detect anomalies by comparing patterns and context to what is "normal." Target: 10,000–20,000 high-quality labeled data images needed to train, validate and test model.

**Key sources & links**:  
- Roboflow Universe: https://universe.roboflow.com/search?q=class%3Ashoplifting  
- Kaggle Datasets:  
  - https://www.kaggle.com/datasets/kipshidze/shoplifting-video-dataset  
  - https://www.kaggle.com/datasets/gti-upm/leapgestrecog  
  - https://www.kaggle.com/datasets/mateohervas/dcsass-dataset  
  - https://www.kaggle.com/datasets/momanyc/museum-collection  
  - https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset  

## Metrics
Primary Metric: Recall – prioritized because failing to identify actual thefts costs 5-20x less than looking into false alarms. False negatives result in money lost with every undetected theft, legal issues and possible regulatory penalties. False positives create friction among the staff, can cause alarm fatigue then real threats get ignored or missed.  
Secondary Metrics: Precision target of 80-90% is achievable and realistic by training on high-quality labeled datasets of behavioral anomalies, general surveillance, and theft detection images. We also focus on speed to maximize the efficiency from real-time detection with 50-500 4K cameras achieving 30 FPS continuous recording.

## Resources Needed
Google Colab L4 GPU to handle required video sequencing needed for this project that performs great with YOLOv8 for training. The project calls for no less than 30 frames per second (fps) and batch sizes can range between 16-32 for faster training which L4 GPU excels in and makes perfect for this project. Google Colab T4 GPU for primary platform to conduct initial testing. Kaggle for backup or more time if needed. UCF-Crime datasets for actual footage of criminal behaviors with a "normal" baseline to compare against. UCF requires a request form at no cost. COCO datasets to provide a baseline for detecting people near museum exhibits. Human pose gestures dataset from MPI. PyTorch framework with YOLOv8 for object detection to handle desired 30 fps.
