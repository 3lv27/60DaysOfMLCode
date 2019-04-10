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
[The Neural Networks Zoo](http://www.asimovinstitute.org/neural-network-zoo/)


## Day 6: April 5, 2019
##### (DL - Drone madness)

**Today's Progress**: Rewatched lesson 6 from fastai course and did the notebook which was about to create a CNN and play a bit with its concepts. I continued aswell with the Pytorch's tutorial.

**Thoughts:** Today was a busy day but I wanted to have also a bit of free time to clear my head. Looking forward for tomorrow to spend the whole day working with the team.

**Link to work:** [CNN](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson6-DataAugmentation.ipynb)


## Day 7: April 6, 2019
##### (DL - Drone madness)

**Today's Progress**: Spend the morning working with the team. We tried to run our DJI Tello with the mavlink protocol but it doesn't have support. I was able to run the Tello with PoseNet, so the next step is to buil the integration with YOLO. Found and read an interesting about SLAM article which could be usefull.

**Thoughts:** Time flies when you are working on something you are passionate about and even more when your are working surrounded by great people.

**Link(s) to article**
[PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://arxiv.org/pdf/1505.07427.pdf)


## Day 8: April 7, 2019
##### (DL - Drone madness)

**Today's Progress**: Today I started the morning looking at the YOLO's implementation code we have willing to integrate with the drone's streaming camera, but at the end I've decied to make a step backwards and start from the very basics. I really want to understand every little detail about what is happening whit the object detection thing. Since I've never worked with Python and its ecosystem before, started to play with OpenCV, from now on I will spend time understanding this awesome tool. I cleared on my mind which is the path I'm gonna follow the next two weeks. I also read an interesting article about data augmentation.

**Thoughts:** Sometimes, willing to run is not the best strategy if you doesn't learnt how to walk first. I'm feeling very passionate about computer vision so I want to understand what's going on in every lilttle step. Commited to amplify my knowledge about object detection. Today I also decided on buying a new PC, since I'm a mac guy and a GPU is needed if you want to work in this field.. I spent time researching about what configuration will fit my needs.  

**Link(s) to article**
[When Conventional Wisdom Fails: Revisiting Data Augmentation for Self-Driving Cars](https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509)

**Link to work:** [OpenCV Basics](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/OpenCV/OpenCV-Basics.ipynb)


## Day 9: April 8, 2019
##### (DL - Drone madness)

**Today's Progress**: Hardly without time.. read an article about object detection, continued playing with OpenCV and started to code the official PyTorch's tutorial.

**Thoughts:** Today was the hardest day since the challenge began, to keep on track with it, was a very busy day (work, meetings..). Looking forward for tomorrow to see the Lesson 1 from fast.ai cutting edge deep learning tutorial which is about object detection.

**Link(s) to article**
[Deep Learning for Object Detection: A Comprehensive Review](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)

**Link to work:** [OpenCV UI](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/OpenCV/OpenCV-UI.ipynb) | [PyTorch's tutorial](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/PyTorch1.ipynb) 


## Day 10: April 9, 2019
##### (DL - Drone madness)

**Today's Progress**: Today I saw the first full lesson about object detection from fast.ai (2h of video), two more left to go. I read a really interesting article about GANs, I'm pretty impressed about what GANs can do. Last but not least, researched a bit about drone firmwares, hardwares and protocols to be able to decide in the team meeting if we continue with the DJI protocol or we go for an open source one.

**Thoughts:** The more I learn, the more passion about this entirely world, I feel there's too much interesting stuff to learn. Here is a quote I want to share with you: **"6 month of hardcore focus and alignment can put you 5 years ahead in life. Don't underestimate the power of consistency and desire. You have what it takes to become the best that you can be. Don't ever doubt yourself. Harness your power. Exceed your expectations."**

**Link(s) to article**
[The Rise of Generative Adversarial Networks](https://blog.usejournal.com/the-rise-of-generative-adversarial-networks-be52d424e517)


## Day 11: April 10, 2019
##### (DL - Drone madness)

**Today's Progress**: I started to watch the next object detection's lesson from fast.ai. I started to work on the lesson's notebook with PASCAL VOC's dataset with a great progression, spent a couple of hours. And as always read another usefull article.

**Thoughts:** Why the hell the day only has 24h, I would like to spend at least 3h more playing with the notebook but unfortunately humans, unlike computers, needs to  rest. Today I also made some calls to drone's stores in order to know where we can find a drone with the px4 controller ready to buy but it is shameful the lack of information in the online stores but even worse is how the employees of physical drone stores doesn't know nothing except Dji. Because of the remaining time we have maybe we should stick to Tello's SDK.

**Link to work:** [ObjectDetection](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson8-ObjectDetection.ipynb)

**Link(s) to article**
[From Exploration to Production — Bridging the Deployment Gap for Deep Learning (Part 1)](https://towardsdatascience.com/from-exploration-to-production-bridging-the-deployment-gap-for-deep-learning-8b59a5e1c819)

