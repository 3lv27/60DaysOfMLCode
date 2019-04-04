# 100 Days Of ML Code

100 Days of Machine Learning Code. Full instructions about the challenge [here](https://github.com/3lv27/100DaysOfMLCode/blob/master/100_Days_of_ML_Code_Instructions.md)

My intention is to divide the challenge in 50 days specifically dedicated to the field of Deep Learning, since they are the days left for the presentation of the Saturdays.ai course project.


### Day 0: March 30, 2019
##### (DL - Drone madness)

**Today's Progress**: Defined our DL based project. As a brief intro our main goal is to build a self autonomous flight drone which would be able to map an indoor space and also be able to find a person in that space. 

**Thoughts:** It is a very ambicious project based on the deadline we have but we are full of willing and very excited to see how far we can go. I still have some researchs to do about the object detection, but I expect to have this solved by tomorrow.

**Link to work:** [Project's repo](https://github.com/george-studenko/Drone-Madness)


### Day 1: March 31, 2019
##### (DL - Drone madness)

**Today's Progress**: Researched about a bunch of computer object detection models. Cleared on my mind the difference between RCNN, Fast-RCNN, Faster-RCNN, YOLO and RetinaNet. I tried to implement RetinaNet with PyTorch but finally I found and implementation with Keras which is working right. Builded Collab filtering model following lesson 4 notebook from Fast.ai course and watched again lesson 5.

**Thoughts:** I spent a few hours reading about object detection models and realized how it is like a new enetirely world inside DL itself, there's still too much stuff to learn. Thinking about it would be super nice to implement RetinaNet inside Fast.ai library. Maybe after I finished the project it is something that I'm willing to do.

**Link(s) to articles**
[Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, FPN, RetinaNet and YOLOv3)](https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359) | 
[RetinaNet Explained and Demystified](https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/) | 
[R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e) | 
[The intuition behind RetinaNet](https://medium.com/@14prakash/the-intuition-behind-retinanet-eb636755607d) | 
[Review: RetinaNet — Focal Loss (Object Detection)](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4) | 
[What’s new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

**Link to work:** [Collab Model](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson4%20-%20Collab.ipynb)


## Day 2: April 1, 2019
##### (DL - Drone madness)

**Today's Progress**: Builded a MNIST SGD and started with a NN from scratch (notebook 5 from lesson 5 of fast.ai course). I've read some PyTorch's tutorial and finally hands oh the drone project! Started to understand the sdk from our drone Dji Tello and also started to build the space recognition protocol.

**Thoughts:** First day of this log combined with my work journey. The only thing I can say is its gonna be hard as f***! Is almost 12am o'clock and I still have a bunch of things to do. I'm not gonna have time to finish the things I started today so looking for finish them tomorrow.

**Link to work:** [SGD](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson5-SGD.ipynb)


## Day 3: April 2, 2019
##### (DL - Drone madness)

**Today's Progress**: Finished the MNIST NN from scratch (notebook 5 from lesson 5 of fast.ai course). Continued reading PyTorch's tutorial and viewed some ML videos while having dinner (trying to maximize as much as I can my free time). Today we had our first meeting project! We agreed on trying Tiny Yolo as our object detection model, I will try to have it done this week. Almost done with the space recognition protocol, having some issues with the camera.

**Thoughts:** Today my working journey was longer than expected, arrived at home at 21:30 and we had the meeting at 22. Did some magic tricks to be able to finish the notebook, progress with the SRP and learn something new about ML. Looking forward to start the weekend an have more free time.

**Link to work:** [NN](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson5-SGD.ipynb)


## Day 4: April 3, 2019
##### (DL - Drone madness)

**Today's Progress**: Read a couple of articles, one about NN in general and other one about RL. Started to review Lesson 6 from fast.ai course and also played a bit with data augmentation for computer vision.

**Thoughts:** One of my main interest is to mix Deep Learning with Reinforcement Learning, I think it could be a key piece in our project aswell so I started to research a bit about this field.

**Link(s) to articles**
[Neural Networks: All YOU Need to Know](https://towardsdatascience.com/nns-aynk-c34efe37f15a) | 
[Applications of Reinforcement Learning in Real World](https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12)

**Link to work:** [Data Augmentation](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson6-DataAugmentation.ipynb)

## Day 5: April 4, 2019
##### (DL - Drone madness)

**Today's Progress**: Read a great article about NN. I have fought with the tiny-yolo implementation because of the maxpooling but without enough time to finished.

**Thoughts:** Today I just was able to sleep 3-4h, so it has been a pretty hard day. I have not been able to do all the things I planned to do. Looking forward for tomorrow.

**Link(s) to article**
[The Neural Networks Zoo](http://www.asimovinstitute.org/neural-network-zoo/) | 
