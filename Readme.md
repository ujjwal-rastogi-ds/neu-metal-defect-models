---
title: NEU Metal Surface Defect Detection
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# NEU Metal Surface Defect Detection

Automated detection and classification of 6 types of metal surface defects using deep learning.

## 🎯 Features

- **6 Defect Classes**: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
- **Multiple Models**: Custom CNN, VGG16, ResNet50 (Transfer Learning)
- **High Accuracy**: 95-98% test accuracy
- **Real-time Prediction**: Upload images for instant defect detection
- **Batch Processing**: Analyze multiple images at once

## 🚀 Usage

1. Navigate to "Single Image Prediction"
2. Select a model (VGG16 recommended)
3. Upload a grayscale defect image (200x200 recommended)
4. View prediction results and confidence scores

## 📊 Models

- **CustomCNN**: Custom architecture with 4 conv blocks
- **VGG16**: Transfer learning from ImageNet
- **ResNet50**: Deep residual learning

## 📚 Dataset

Based on NEU Surface Defect Database:
- 1,800 grayscale images (200×200 pixels)
- 300 samples per class
- Training/Val/Test split: 70/15/15

## 🔗 Links

- [Original Dataset](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)
- [GitHub Repository](https://github.com/yourusername/neu-defect-detection)

## 📄 License

MIT License - See LICENSE file for details
