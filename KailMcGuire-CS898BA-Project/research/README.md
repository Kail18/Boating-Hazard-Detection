# Research Directory

The purpose of this directory is to provide research papers used in determining the direction of this project. In this md file it will also be used to discuss the different papers in the papers/url directories.

## OpenCV

This project will be using OpenCV as an image-processing and visualization library. YOLO will be responsible for model training and inference while OpenCV will support the surrounding data. OpenCV will convert images from BGR to LAB to assist in the CLAHE experiment.

## YOLO (YOLO.pdf)

citation APA 7: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 779–788). https://doi.org/10.1109/CVPR.2016.91

### Introduction

The paper discusses that while YOLO makes localization errors it decreases the amount of background errors compared to R-CNN models. It also boasts that it processes images faster by using a single look at the image as a whole resulting less GPU usage and latency with streaming. The paper discusses using regression as the pipeline resulting in a simpler faster model than R-CNN's piplines.

### Detection and Design

The model uses a confidence score to determine performance. The calculation for the score is PR(object) x IOU(truth/pred). There network design used PASCAL VOC dataset with 24 convolutional layers and 2 fully connected layers. They used a 1x1 reduction layer followed by 3x3 convolutional layer. They compared this to a fast YOLO nueral network which only used 9 convolutional layers instead of 24.

### Limitations

According the paper the model struggles with groups of small objects. There is a greater effect on the IOU with errors in small objects, secondary to the loss function treating all errors as the same.

### Conclusion

YOLO excells at fast detection and larger object detection, while fast R-CNN models see better performance with small objects. Combining both R-CNN and YOLO shows a promising results for a high performing detection method.

## Faster R-CNN

citation APA 7: Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. In Advances in Neural Information Processing Systems (Vol. 28, pp. 91–99).

### Introduction
