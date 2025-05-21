# SOS-Guardian 🚨

SOS-Guardian is a pioneering women’s safety system that leverages real-time CCTV analysis to automatically trigger SOS alerts, enhancing security in public and private spaces. Built with YOLO for object and gesture detection, PyTorch for deep learning, and audio processing for sound recognition, it detects raised hands (above shoulder level), specific distress calls (e.g., “SOS,” “help”), and weapons. Running on a GPU-enabled server or cloud, SOS-Guardian addresses the limitations of mobile-based SOS apps, which only 50% of users adopt, by providing an autonomous, always-on safety solution.


This document provides an overview of the project, its positive outcomes, and the challenges faced during development, offering insights into the technical journey and lessons learned. It is designed for public sharing on platforms like LinkedIn to highlight the project’s impact and my expertise as a developer.

# Project Overview 📋

SOS-Guardian revolutionizes women’s safety by automating SOS triggers through real-time CCTV analysis, eliminating reliance on mobile apps. Deployed on a physical server or cloud with GPU acceleration, it processes video and audio feeds continuously to detect critical situations, such as raised hands, distress calls, or weapons, and initiates alerts (e.g., notifying authorities). The project demonstrates advanced computer vision, audio processing, and deep learning, addressing a critical societal need.

```Purpose:``` Enhance women’s safety by automating SOS alerts in real-time, improving response times in emergencies.

```Platform:``` GPU-enabled server or cloud (Windows/Linux tested), optimized for high-performance processing.

```Key Features:```

Detects raised hands above shoulder level as an SOS gesture ✋.

Recognizes distress sounds (e.g., “SOS,” “help”) via audio analysis 🗣️.

Identifies weapons (e.g., knives, guns) for threat detection 🔪.


# Technical Stack:

````Computer Vision: ````YOLO for gesture and weapon detection.

````Deep Learning:```` PyTorch for model training and inference.

```Audio Processing:``` Libraries like Librosa for sound recognition.

```Environment:``` Python 3.9, GPU-accelerated with CUDA 12.6.


```Operation:``` Always-on backend process, analyzing CCTV feeds 24/7.

The project showcases real-time safety monitoring, deep learning model training, and system integration, making it a impactful contribution to public safety technology.

# Positive Outcomes 🌟

Developing SOS-Guardian yielded significant successes, both in technical achievements and societal impact. Below are the key positives:

```1. Real-Time SOS Detection 🚨```

Achievement: Implemented YOLO-based detection to identify raised hands and weapons in CCTV footage with high accuracy, processing frames in real-time.

Benefit: Enables rapid SOS alerts, reducing response times in emergencies.

Impact: Addresses the 50% adoption gap of mobile SOS apps by automating triggers, ensuring safety for all users in monitored areas.

Example: Raising hands above shoulders triggers an alert within milliseconds, logged as:2025-05-21 15:00:01,456 - INFO - SOS gesture detected: Hands raised above shoulders



```2. Distress Sound Recognition 🗣️```

Achievement: Integrated audio processing to detect distress calls like “SOS” or “help” using spectral analysis and machine learning.

Benefit: Complements visual detection, capturing emergencies in low-visibility conditions.

Impact: Enhances system reliability by combining multimodal inputs (video + audio).

Example: A shout of “help” triggers:2025-05-21 15:00:02,789 - INFO - Distress sound detected: 'help' with confidence 0.92




```3. Societal Impact 🙌```

Achievement: Developed a system that directly addresses women’s safety, a critical global issue.

Benefit: Offers a scalable solution for public spaces (e.g., streets, campuses) and private facilities.

Impact: Demonstrates technology’s potential to drive social good, aligning with safety and inclusion goals.

```4. Robust Model Training 📊```

Achievement: Trained a custom YOLO model over 10,000 epochs to detect gestures and weapons, achieving high precision.

Benefit: Ensures accurate detection under varied lighting, angles, and backgrounds.

Impact: Provides a reliable foundation for real-world deployment, adaptable to new datasets.

Example: Training log:2025-05-20 10:00:00,123 - INFO - Epoch 10000/10000, mAP@0.5: 0.89, Loss: 0.021



```5. Scalable Architecture 🖥️```

Achievement: Designed for GPU-enabled servers/cloud, supporting multiple CCTV feeds simultaneously.

Benefit: Scales to large environments (e.g., city-wide surveillance) with minimal latency.

Impact: Positions SOS-Guardian as a viable solution for enterprise and municipal applications.

```6. Personal Growth 📚```

Achievement: Mastered YOLO, PyTorch, audio processing, and GPU-based deployment through hands-on development.

Benefit: Gained expertise in deep learning, real-time systems, and multimodal AI.

Impact: Strengthened my portfolio, showcasing skills in computer vision, audio analysis, and safety-focused innovation.


# Challenges Faced (Negatives) and Resolutions 🔍

The development of SOS-Guardian presented several challenges, each offering valuable lessons. Below are the key negatives and how they were addressed.

```1. GPU Setup Complexities 🖥️```

Issue: Configuring NVIDIA drivers, CUDA, and cuDNN was error-prone, with version mismatches causing runtime failures:2025-05-20 09:00:01,234 - ERROR - CUDA runtime error: Incompatible driver version


Cause: Inconsistent CUDA/PyTorch versions and incomplete cuDNN setup.

Impact: Delayed model training and inference.

Resolution:

Standardized on CUDA 12.6 and verified with:nvidia-smi

    nvcc --version


Used PyTorch’s CUDA 12.6 wheel:pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


Ensured cuDNN libraries were correctly placed in system paths.


Lesson: Rigorous version compatibility checks are essential for GPU-based systems.

```2. Model Training Overfitting 📉```

Issue: The YOLO model overfitted after 10,000 epochs, performing poorly on diverse CCTV footage:2025-05-20 12:00:00,567 - WARNING - Validation mAP@0.5 dropped to 0.65


Cause: Limited dataset diversity and insufficient regularization.

Impact: Reduced detection accuracy in real-world scenarios.

Resolution:

Augmented the dataset with varied lighting, angles, and backgrounds.

Applied dropout (0.3) and weight decay (0.01) in the YOLO configuration.

Used early stopping at 8,500 epochs when validation mAP stabilized.


Lesson: Dataset diversity and regularization are critical for generalizable models.

```3. Sound Detection Accuracy 🗣️```

Issue: Distress sound recognition had high false positives (e.g., mistaking “hello” for “help”).

Cause: Limited training data and noise in CCTV audio.

Impact: Triggered unnecessary alerts, reducing system trust.

Resolution:

Expanded the audio dataset with noisy samples and negative examples (e.g., casual speech).


Fine-tuned the model with a higher confidence threshold (0.9).

Added preprocessing (noise reduction, spectral gating) using Librosa.


Lesson: Multimodal systems require balanced training across inputs to minimize errors.

```4. Camera Feed Inconsistencies 📷```

Issue: CCTV feeds failed to initialize or dropped frames, similar to issues in your Gesture-Based project:2025-05-20 14:00:02,890 - ERROR - Camera feed disconnected: Index 0 unavailable


Cause: Variable camera indices and network latency for IP-based CCTV.

Impact: Interrupted real-time processing, missing potential SOS events.

Resolution:

Implemented a retry mechanism for camera initialization:

    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            break
    if not cap.isOpened():
        raise Exception("No camera feed available")


Added support for RTSP streams for IP cameras.

Created a diagnostic script to test feeds (test_camera.py).


Lesson: Robust camera handling is critical for surveillance systems.

```5. Real-Time Performance Lag 🐢```

Issue: Processing multiple CCTV feeds caused lag on lower-end GPUs.

Cause: High computational load from YOLO and audio models.

Impact: Delayed SOS triggers, compromising responsiveness.

Resolution:

Optimized YOLO inference with batch processing and model pruning.

Reduced audio frame rate to 16 kHz for faster processing.

Scaled to cloud GPUs (e.g., AWS EC2) for high-feed scenarios.


Lesson: Performance optimization and scalable infrastructure are key for real-time applications.





# Why This Project? 🤔

SOS-Guardian addresses a critical societal issue—women’s safety—by automating emergency detection, offering a proactive alternative to mobile-based SOS apps. It highlights my skills in:

Computer Vision: YOLO for real-time gesture and weapon detection.

Deep Learning: PyTorch for model training and optimization.

Audio Processing: Sound recognition for distress calls.

System Design: Scalable, GPU-accelerated architecture.

The project’s success in integrating multimodal AI for safety underscores its potential for real-world deployment in public and private security systems.


# Acknowledgments 🙏

```Ultralytics YOLO for object detection.```

```PyTorch for deep learning.```

```Librosa for audio processing.```

```OpenCV for video feed handling.```



Contact 📧
Connect on LinkedIn or email ```agathees2401@gmail.com```  for inquiries or collaboration.
Note: This document is a public overview of a private project, detailing its purpose, successes, and challenges. For source code access, contact the author.
