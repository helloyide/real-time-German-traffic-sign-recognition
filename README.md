This project is a basic test for real-time german traffic sign recognition with deeplearning techniques. Use one camera inside the car, it localizes and classifies the traffic sign in real time.

There are not many open training data for german traffic signs. Here two small datasets are used: [GTSDB](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) and [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

#Problems

1. Besides the accuracy, as a real-time system the speed of inference is also important. The final goal of this project is to install it on a Raspberry Pi. So the big network architecture like Inception-ResNet and the complex algorithm like Mask R-CNN may not applicable here.
2. The datasets are quite small for deeplearning. GTSDB only contains 900 images (devided in 600 training images and 300 evaluation images). Although GTSRB has more images (50,000 images in total) but it is only for classification, traffic signs are cropped from background images. It also only contains 43 different traffic signs, common signs like "Haltverbot", "Fußgängerüberweg" and some speed limit signs are missing.

#Solution

Different than standard end-to-end object detection algorithms, in this project a pipeline is built, localization and classification tasks are handled separately.

##Localization
For localization U-Net, a convolutional neural network for semantic segmentation, is used. It is popular for biomedical image segmentation. Because biomedical training data are expensive and hard to get, this network works pretty well with little data.

This is my training result:
TODO:

##Classification
For classification a convolutional neural network calls SqueezeNet v1.1 is chosen. SqueezeNet is famous for its small size and still good enough accuracy.

This is my training result:
TODO:

##Pipeline
1. Video frames are retrieved in a worker thread (25fps), they are then sent to the U-Net, traffic sign segmentation are found and bounding boxes are calculated.  
2. Crops the image areas inside bounding boxes and resize it to 90x90 (keep width-height ratio and pad borders if need) 
3. Cropped images are sent to the classifier network, it outputs classes and confidence scores (softmax value), ignore the one with score lower than a threshold. 
4. Draw bounding boxes, classes and scores on frames.

#Issues
After the first implementation some issues are found:

1. Some bounding boxes are marked incorrectly. It also decreases the classification accuracy.
2. Bounding boxes blinks a lot.

To solve these issues, a 10x10 grid is created, frame is then divided into 100 equal areas.

TODO: example

With the help of this grid, some parts of frame are ignored, for example last two rows at the bottom because of the camera position. These areas should not contain traffic signs, it reduces some incorrect bounding boxes.  

To reduce the blink effect, for each grid cell, instead of only considering the current frame, 5 recent frames are used to calculate the mode of classes and the average of confidence scores.
The grid helps to identify the traffic sign instance in successive frames (with the assumption that each traffic sign belongs to the same cell in recent 5 frames and there is only one traffic sign in each cell)
  
The number of frames is important, if it's too large, some important frames might be lost. Actually it should be set dynamically based on the car speed, less frames should be used when it is fast.
Also the grid is divided equally, a perspective grid system might be better.

At the last, too small boxes are ignored. The classifier won't work well in this case.

#Result
The video is cut from a driving lesson video (URL), please ignore the white texts in the video.
Some traffic sign are classified incorrectly, because GTSRB dataset doesn't contain these signs

TOOD: video

#Possible improvement
* There are many different speed limit signs, besides including all of them into training data, a number recognizer could also be added in the pipeline. A small convolutional neural network works already pretty good for MNIST hand writing number recognition, the numbers on traffic signs are more formal, it should be easy to build such recognizer without introduce much time during inference. 
* The accuracy of localization network could be improved. There is an alteration of the original U-Net design, which adds dropout layers and batch normalization. Also can try to train with other traffic sign datasets, although they are not for german traffic signs, for localization task it could be helpful.
* With additional information like the distance of the object we may distinguish them better. Like this example, if we know it's far away, we wont handle it as a traffic sign: TODO
* As mentioned before, grid could be perspective and the number of frame could be dynamic.
* In general camera does not work well in bad wetter condition and light condition.
