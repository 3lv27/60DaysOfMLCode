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


## Day 12: April 11, 2019
##### (DL - Drone madness)

**Today's Progress**: I continued working on the lesson's 8 notebook. Meeting with the team and as always read another article.

**Thoughts:** Today I have barely been able to spend a couple of hours, was a really busy day. Working on the notebook is taking much longer than expected, since the code is written on the fast.ai oldest version and I have to translate and research how to do everything with the new version. Good news is that I'm learning more this way. Looking forward for tomorrow since tomorrow I'm gonna have the whole afternoon free to focus on ML.

**Link to work:** [ObjectDetection](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson8-ObjectDetection.ipynb)

**Link(s) to article**
[TOP 10 Machine Learning Algorithms](https://blog.goodaudience.com/top-10-machine-learning-algorithms-2a9a3e1bdaff)


## Day 13: April 12, 2019
##### (DL - Drone madness)

**Today's Progress**: I finished the Pytorch's tutorial notebook, read a few articles and watched and interesting video about AlphaStar and its architecture.

**Thoughts:** Feeling really tired today after all the week. Couldn't spend as much time as I exoected working on ML today but enough to keep on track. From now on I have one week of holidays so you can expect a lot of progress in the coming days.

**Link to work:** [PyTorch - Autograd](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/PyTorch1.ipynb)

**Link(s) to article**
[An Introduction to Deep Learning for Tabular Data](https://www.fast.ai/2018/04/29/categorical-embeddings/) | [A Neural Network in PyTorch for Tabular Data with Categorical Embeddings](https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/) | [ResNet for Traffic Sign Classification With PyTorch](https://towardsdatascience.com/resnet-for-traffic-sign-classification-with-pytorch-5883a97bbaa3) | [AlphaStar video](https://www.youtube.com/watch?v=GmRNpvASiPk)


## Day 14: April 13, 2019
##### (DL - Drone madness)

**Today's Progress**: Spent the morning working with the team. We were able to run the DSO paper and processed and already recorded drone's video, great news! I'm fighting with the drone's video streaming to be able to process YOLO in real time.

**Thoughts:** It is quite frustating to spent too much time in something not related with DL directly, but that's what makes this project really really challenging and interesting. In fact no one of the team members haven't had any kind of contact with a drone before and in my case nor even with video or video streaming, but hopefully I will have everything sorted by tomorrow.


**Link(s) to article**
[DSO Paper](http://vladlen.info/papers/DSO.pdf)


## Day 15: April 14, 2019
##### (DL - Drone madness)

**Today's Progress**: Finally managed to understand how the whole process of the Tello's video streaming works. Read a few pages of the Deep Learning's bible by Fellow, I. et al. Played around with the ZARA's challenge dataset in order to improve my EDA. Read an article from a classmate.

**Thoughts:** One thought: Just realized how tedious EDA can be when working with huge datasets.


**Link(s) to article**
[My data science template for Python](https://medium.com/saturdays-ai/my-data-science-template-for-python-59a67cba4290?sk=01dd86a6655d8c0bc7fa264ed5eb46d5) | [Deep Learning's Bible](https://www.deeplearningbook.org/)



## Day 16: April 15, 2019
##### (DL - Drone madness)

**Today's Progress**: Continued with the lecture of the DL's bible, read another article and went back to the last lesson of ther first part of fast.ai course since this was the only one left to watch for second time. Started the notebook corresponding to that lesson.

**Thoughts:** Sometimes I thought that my learning path could be seen from outside as a completely mess, but it just a reflect of the desire about the will of learning all the things I want to learn, and after all I'm learning a lot, so it means there's no a specific path or a perfect strategy to learn something rather than find your own perfect strategy based on your own personality.


**Link(s) to article**
[Finding Lane Lines — Simple Pipeline For Lane Detection.](https://towardsdatascience.com/finding-lane-lines-simple-pipeline-for-lane-detection-d02b62e7572b)

**Link to work:** [Resnet Mnist](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson7-Resnet-Mnist.ipynb)



## Day 17: April 16, 2019
##### (DL - Drone madness)

**Today's Progress**: Finished lesson 7 video and notebook's resnet. Continued reading DL bible's and played a bit with the drone trying to improve the video streaming system. Clarified a couple of concepts about cnn reading an article.

**Thoughts:** Little by little getting the picture of how a NN and its variations works on its roots. Just figuring out how much stuff you need to learn as a Data Scientist to be able to do a good job. As I said before I'm felling more comfortable little by little within the depp learning field but since the other day I was playing around doing EDA, I realized sometime I will need to focus on improving this absolutely necessary skill.


**Link(s) to article**
[Guide To Understanding Convolutional Neural Networks Part 2](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

**Link to work:** [Resnet Mnist](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson7-Resnet-Mnist.ipynb)



## Day 18: April 17, 2019
##### (DL - Drone madness)

**Today's Progress**: Spent the morning working / fighting on GAN's notebooks from lesson 7. Since I'm a mac guy with a non-existing macbook's GPU, I was having a lot of problems related with the workers from PyTorch's library I spent a couple of hours trying to fix it in order to be able to continue working with my mac with no success. Finally I end up working on google's collab to be bale to finish the notebook.

**Thoughts:** Thinking on accelerate my plans on buying a brand new PC with a powerfull GPU.

**Link to work:** [GAN](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Fastai-dl1-nbs/Lesson7-GAN.ipynb)


## Day 19: April 18, 2019
##### (DL - Drone madness)

**Today's Progress**: Continued with the PyTorch's tutorial, created a NN using pure PyTorch and read some docs aswell. REfresh some Math concepts in Khan academy and read one article.

**Thoughts:** Having a lot less time than expected. Extending the day till 2 - 3 am to be able to keeping up with the challenge.

**Link to work:** [PyTorch's NN](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/PyTorch-NN.ipynb)

**Link(s) to article:**
[Kaggle CareerCon 2019 competition report (#121 of 1449, top 9%)](https://medium.com/saturdays-ai/kaggle-careercon-2019-competition-report-121-of-1449-top-9-21a1b7901af7)


## Day 20: April 19, 2019
##### (DL - Drone madness)

**Today's Progress**: Today I started a new course from Udemy called "The complete NN bootcamp" from which I was able to finish the first section. From now it was all theory but it has some practical lessons coming. Moreover the course a complete YOLO's specific section, that's why I was convinced to buy the course.

**Thoughts:** It is really usefull to start all the theory from scratch since I'm solidifying a lot of concepts, but looking forward to start more advanced practical lessons. I didn't wrote a single line of code today, it was all about theory but tomorrow I will return to work on the drone's project hoping to finally implement object's recognition in the video streaminh pipeline.

**Link(s) to article:**
[Create a complete Machine learning web application using React and Flask](https://towardsdatascience.com/create-a-complete-machine-learning-web-application-using-react-and-flask-859340bddb33)


## Day 21: April 20, 2019
##### (DL - Drone madness)

**Today's Progress**: Continued with the new course I started yesterday, completing section 2 and 3, then I jumped to section 6 which is where the practice starts. I'm clarifying a lot of concepts thanks to the course. At this point I'm starting to feel that I'm really grasping what's really going on behind the scenes as far as NN's are concerned.

**Thoughts:** Since I've been trying to complete the course asap, finally I didn't have time to get down to work on the project. For sure tomorrow I will be playing around with the drone.

**Link(s) to article:**
[Review: WRNs — Wide Residual Networks (Image Classification)](https://towardsdatascience.com/review-wrns-wide-residual-networks-image-classification-d3feb3fb2004)


## Day 22: April 21, 2019
##### (DL - Drone madness)

**Today's Progress**: Kept on with the course, starting finally the practice but beginning with the very basics about PyTorch. Played a bit with the drone, but as usually having troubles with the video streaming thing. At the moment I don't get why sometimes seems like  everything is working well and sometimes it isn't working at all. The good news is that finally I found a couple of libraries which handles all the lower processes about the streaming.

**Thoughts:** I never would have imagined how hard could be to work with the drone's streaming camera. At the moment, since all I've investigated I'm able to work having the data of different type of cameras, but the drone's camera is another thing. I'm gonna try tomorrow to work with the libraries I found, hping to have better results once and for all.

**Link(s) to article:**
[The Real Reason behind all the Craze for Deep Learning](https://towardsdatascience.com/decoding-deep-learning-a-big-lie-or-the-next-big-thing-b924298f26d4)

**Link to work:** [NN Bootcamp - PyTorch's Basics](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/Tensors%20and%20Operations.ipynb)



## Day 23: April 22, 2019
##### (DL - Drone madness)

**Today's Progress**: Finished the practice of the course related to PyTorch's basics, did another notebook aswell from the course about building a NN with PyTorch vanilla and visulize its process. Also finished section 4 and 5 from the theory, so far now I'm exactly at the 1/3 of the whole course.

**Thoughts:** Happy with the path I'm doing the course, expecting to reach YOLO's section this week to be able to apply all the knowledge in the project. I was thinking I'm gonna focus on the course to be able to reach the goal and spend the whole weekend with the drone. The Elephant goal by the end of the week is to have YOLO in real time processing drone's video stream. 

**Link(s) to article:**
[L1 and L2 Regularization Methods](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)

**Link to work:** [NN Bootcamp - PyTorch's Basics](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/Tensors%20and%20Operations.ipynb) | [NN Bootcamp - PyTorch's NN + Learning Visualization](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/Learning%20Process%20Visualization.ipynb)



## Day 24: April 23, 2019
##### (DL - Drone madness)

**Today's Progress**: Completed section 8 and 9 from the Udemy's course. I've did a couple of notebooks today, in one I've built a NN with plain python and in the other one a FFNN with PyTorch using the MNIST dataset.

**Thoughts:** Today I've reached 1/4 of the challenge. Since I started the challenge I'm investing much more time studying than before, somedays it is a bit tough but I'm having pretty much fun at all. Enjoying so much this journey.

**Link(s) to article:**
[Analyze video to detect players, teams, and who attempted the basket](https://towardsdatascience.com/march-madness-analyze-video-to-detect-players-teams-and-who-attempted-the-basket-8cad67745b88)

**Link to work:** [NN Bootcamp - Python NN](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/Vanilla-NN.ipynb) | [NN Bootcamp - PyTorch FFNN](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/FFNN-MNIST.ipynb)



## Day 25: April 24, 2019
##### (DL - Drone madness)

**Today's Progress**: Completed section 10 and 11 from the Udemy's course. I've been working on a notebooks with a csv dataset building a simple NN. Also saw an interesting video about reinforcement learning (in Spanish) which I'll provide the link below.

**Thoughts:** I'm a couple of lessons behind I would like to be from the course right now, but I would try anyway to reach section 18 by Saturday morning.

**Link(s) to video:**
[Montezuma's Revenge](https://www.youtube.com/watch?v=DBJh4cfq0ro)

**Link to work:** [NN CSV Dataset - Python NN](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/Diabetes-NN.ipynb) 



## Day 26: April 25, 2019
##### (DL - Drone madness)

**Today's Progress**: Completed section 12 from the Udemy's course. Read one article and saw a couple of videos (Spanish).

**Thoughts:** Today was a really hard busy day, I haven't been able to fullfill what I had set out to do today, literally I haven't had time but I will try to catch up tomorrow and be ready for make a couple of steps forward on staurday with the drone's madness project. 

**Link(s) to article:**
[Facebook Says Developers Will Love PyTorch 1.0](https://medium.com/syncedreview/facebook-says-developers-will-love-pytorch-1-0-ba2f89ebc9cc)

**Link(s) to video:**
[The algorithms behind the first image of a black hole](https://www.youtube.com/watch?v=pTXCs3A6NEM) | [AlphaStar, la IA que domina el STARCRAFT II](https://www.youtube.com/watch?v=GmRNpvASiPk)



## Day 27: April 26, 2019
##### (DL - Drone madness)

**Today's Progress**: Watched section 14 and 15 about CNN's architectures and RNN. Built a CNN with PyTorch based on the MNIST dataset and started to watch the YOLO's section

**Thoughts:** Today I had more time to study as usuall and really enjoyed it. Hopefully tomorrow I will finish the section about YOLO and play around with the drone.

**Link(s) to article:**
[What I have learned after several AI projects](https://medium.com/predict/what-i-have-learned-after-several-ai-projects-131e345ac5cd)

**Link to work:** [CNN MNIST pt. 1](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/CNN-MNIST.ipynb)



## Day 28: April 27, 2019
##### (DL - Drone madness)

**Today's Progress**: Spent the morning with the team talking about the direction of the project, we are  at 15 days left aprox to the show day. Started to thinking about our performance to that day. I've continued with the YOLO's lessons aswell. Since in my job we are making bets about the final season of GOT (who lives, who dies and who becomes wight) I was thinking to build a NN death's character predictor so I've been looking for an api which provides the info I need and it looks like I'm finally found one.

**Thoughts:** Today I decided I'm gonna buy a new racing drone because if we want to go further with the project we need to work with an open source software and also be able to have control over the hardware. 

**Link(s) to article:**
[The Ultimate Game of Thrones Dataset](https://medium.com/@jeffrey.lancaster/the-ultimate-game-of-thrones-dataset-a100c0cf35fb)



## Day 29: April 28, 2019
##### (DL - Drone madness)

**Today's Progress**: Continued with the DL's bible reading, read about Jacobian and Hessian matrices. Related to the NN course: progressed with the YOLO's section, finished section 14 about CNN's normalization and built the corresponding notebook.

**Thoughts:** Normally on sundays I'm able to do more work than I did today, but today I felt sick the whole day so I'm really happy to at leat have been able to enjoy, learn and progress more in this fascinating field. 

**Link(s) to article:**
[Simplifying Deep Learning with Fast.ai](https://towardsdatascience.com/simplifying-deep-learning-with-fast-ai-37aa0d321f5e)

**Link to work:** [CNN MNIST pt.2](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/CNN-MNIST2.ipynb)


## Day 30: April 29, 2019
##### (DL - Drone madness)

**Today's Progress**: Continued a bit with the YOLO's section of the course and also watched a few videos from another course that I've already bought about Reinforcement Leaning. Yes, I admit curiosity beat me and I couldn't wait to take a look at the course. 

**Thoughts:** I'm already at 70% of the course and I'm thinking about next step. RL is something that is already in my path but I want to consolidate more all the knowledge  about NN before switching to RL.

**Link(s) to article:**
[Everything you need to know to master CNN](https://medium.freecodecamp.org/everything-you-need-to-know-to-master-convolutional-neural-networks-ef98ca3c7655)



## Day 31: April 30, 2019
##### (DL - Drone madness)

**Today's Progress**: Started the transfer learning's section of the course and its notebook. Read a few articles. 

**Thoughts:** I wish I could have done more than I actually did. I would like to finish the course before the weekend and then focus 100% on the project since we are at a couple of weeks to the demo day.

**Link(s) to article:**
[Probability concepts explained: Introduction](https://towardsdatascience.com/probability-concepts-explained-introduction-a7c0316de465) | [Probability concepts explained: Maximum likelihood estimation](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1) | [Why your personal Deep Learning Computer can be faster than AWS and GCP](https://medium.com/the-mission/why-your-personal-deep-learning-computer-can-be-faster-than-aws-2f85a1739cf4)

**Link to work:** [Transfer Learning](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/Transfer%20Learning.ipynb)


## Day 32: May 1, 2019
##### (DL - Drone madness)

**Today's Progress**: Ended with the transfer learning's section of the course and its notebook. I also did another transfer learning notebook from PyTorch's tutorial in order to see another way to the the same thing.

**Thoughts:** Today I was able to dedicate more than 6 hours studying and learning, so I'm really happy with today's progress.

**Link(s) to article:**
[Transfer learning: the dos and don’ts](https://medium.com/starschema-blog/transfer-learning-the-dos-and-donts-165729d66625) 

**Link to work:** [Transfer Learning NN Botcamp](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/NN%20Bootcamp/Transfer%20Learning.ipynb) | [Transfer Learning PyTorch's tutorial](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/Transfer%20Learning.ipynb)


## Day 33: May 2, 2019
##### (DL - Drone madness)

**Today's Progress**: Finished YOLO's course section. Followed with the PyTorch's official tutorial reviewing the lesson about how to build a CNN and the next which was about training a classifier and wrote down the notebook aswell. Also found an interesting project related with AI called OpenMined and researched a bit about it. 

**Thoughts:** Today we had meeting with the team and decided the next steps about the drone's project, next saturday will be critical to see how far we will be able to go with it.

**Link(s) to article:**
[Python at Netflix](https://medium.com/netflix-techblog/python-at-netflix-bba45dae649e) 

**Link to work:**
[Training a classifier](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/Training_a_classifier.ipynb)


## Day 34: May 3, 2019
##### (DL - Drone madness)

**Today's Progress**: Today I was focus on researching about tabular data, so I started a notebook in order to create a NN specially to process tabular data. Also read again a couple of articles related to this field that I'd read weeks ago. I'm going to participate in a datathon in a couple of weeks so from now on I will be focused more on that field.

**Thoughts:** Barely without time I was able to dedicate a bit more than an hour. Lokkinf forward for tomorrow to work with the team in the drone,

**Link(s) to article:**
[An Introduction to Deep Learning for Tabular Data](https://www.fast.ai/2018/04/29/categorical-embeddings/) | [A Neural Network in PyTorch for Tabular Data with Categorical Embeddings](https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/)

**Link to work:**
[NN for Tabular data](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Misc/TabularData.ipynb)



## Day 35: May 4, 2019
##### (DL - Drone madness)

**Today's Progress**: Had a lot of fun at the morning with the team working on the project, we started to ensamble the pieces we have been developing. At noon I was reviewing some math concepts (chain rule, leibniz notation, etc..), reading about how to put a model in prod with PyTorch and finished the notebook I started yesterday but I still need to improve some things.

**Thoughts:** About the project looks like we finally are gonna have something to show, I'm really excited about the demo day but keeping in mind that this is just the begining of this awesome project.

**Link(s) to article:**
[TORCH.ONNX](https://pytorch.org/docs/master/onnx.html) | [Which is better suited for deploying deep learning models into production, PyTorch or TensorFlow?](https://www.quora.com/Which-is-better-suited-for-deploying-deep-learning-models-into-production-PyTorch-or-TensorFlow)

**Link to work:**
[NN for Tabular data](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Misc/TabularData.ipynb)



## Day 36: May 5, 2019
##### (DL - Drone madness)

**Today's Progress**: Today I've been focused more on practice than theory. I was playing around bulding a churn predictor, end up with a pretty good accuracy (81,5%).

**Thoughts:** I built the churn predictor using fastai, tomorrow I would like to play with different architectures using plain PyTorch.

**Link(s) to article:**
[Parallel and Distributed Deep Learning: A Survey](https://towardsdatascience.com/parallel-and-distributed-deep-learning-a-survey-97137ff94e4c)

**Link to work:**
[Churn predictor](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Misc/Churn-predictor.ipynb)


## Day 37: May 6, 2019
##### (DL - Drone madness)

**Today's Progress**: Researched about data manipulation and how to deal with imbalanced data. Thinking about it I end up with the idea of oversampling helped by a GAN so I started to research more about GAN's. Reviwed the fastai's lesson about GANs, read a couple of articles, dive into forum topics and select a couple of papers to read in the next days. Also started to write the churn predictor notebook just with PyTorch.

**Thoughts:** I'm feeling pretty exited to start working on a field where there's no too much work already done. I think being able to generate accurate fake tabular data could be a huge step for do thing inside a company. Looking forward to start coding.

**Link(s) to article:**
[How to Handle Imbalanced Data: An Overview](https://www.datascience.com/blog/imbalanced-data) | [Comprehensive Introduction to Turing Learning and GANs: Part 1](https://towardsdatascience.com/comprehensive-introduction-to-turing-learning-and-gans-part-1-81f6d02e644d)

**Link to work:**
[Churn predictor - PyTorch](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Misc/ChurnPredictor-PyTorch.ipynb)


## Day 38: May 7, 2019
##### (DL - Drone madness)

**Today's Progress**: Today was all about GANs. Started to read a paper that I want to implement in PyTorch, it is pretty extense and dense and since I'm trying to understand everything in deep I still have left the half of it. Regarding to practice I came back to the PyTorch's tutorial and built a DCGAN. Also viewed a yotube video about news in the world of GANs.

**Thoughts:** I'm feeling pretty good since I'm understanding all the theory behind GANs and feeling pretty exited to be able to work with such and incredible algorithm. I wrote down all the code but I still need to train the model but I'm guessing it's goona take a lot of tie so I'm gonna do it by tomorrow

**Link(s) to article:**
[Data Synthesis based on Generative Adversarial Networks](https://arxiv.org/abs/1806.03384) 

**Link to work:**
[DCGAN - PyTorch](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/DCGAN.ipynb)


## Day 39: May 9, 2019
##### (DL - Drone madness)

**Today's Progress**: In order to train the GAN I was forced to use a Google VM, I tried to train it in my mac but it was taking 3h just to complete a 2%. With GPU at the end it tooks like 1h and few. Related to the drone's project I was researching about how to map the 3rd dimension, got some clues.

**Thoughts:** Today I was spending too much time configurating the VM and downloading the dataset into the machine, maybe tomorrow I'm gonna try to train the model longer to see how far it can get better generating images. 

**Link(s) to article:**
[Open Questions about Generative Adversarial Networks](https://distill.pub/2019/gan-open-problems/#tradeoffs) 

**Link to work:**
[DCGAN Trained - PyTorch](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/DCGAN-Trained.ipynb)


## Day 40: May 9, 2019
##### (DL - Drone madness)

**Today's Progress**: Trained the DCGAN model triple the time than yesterday and get much better results. Viewed a yputube video about GANs and continued with the reading of the paper, trying to figure it out how to implement some things.

**Thoughts:** Today was one of those days barely without time, but had fun training the model and undertanding in deep the algorithm. Hoping to start next week experimenting with GANs and tabular data.

**Link(s) to article:**
[Live Object Detection](https://towardsdatascience.com/live-object-detection-26cd50cceffd) 

**Link to work:**
[DCGAN Trained x3 - PyTorch](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/DCGAN-Trainedx3.ipynb)


## Day 41: May 10, 2019
##### (DL - Drone madness)

**Today's Progress**: Researched about Adversarial Networks and found out with another algorithm that I didn't knew about it called FGSM. Started to code a notebook about this architecture. 

**Thoughts:** Learning more about this kind of algorithms, looking forward next week to try to implement some adversarial network with tabular data.

**Link(s) to article:**
[A detailed example of how to generate your data in parallel with PyTorch](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel) 

**Link to work:**
[FGSM](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/FGSM.ipynb)


## Day 42: May 11, 2019
##### (DL - Drone madness)

**Today's Progress**: Great day working with the team! for first time I'm starting to see a clear path on what we want to achieve. Today we found the way to create the navigation algorithm based on the depth recognized on an image. We were able to write a test and it worked!

**Thoughts:** Really happy about the progress we made today, looking forward to implement some things I have in mind tomorrow. 

**Link(s) to article:**
[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)


## Day 43: May 12, 2019
##### (DL - Drone madness)

**Today's Progress**: Finished the FGSM's notebook. Researched a bit about adversdarial attacks and defence. Worked on the drone's project cleaning data and re-structuring some thingd. Found an implementation of the tableGAN paper with tensorflow, I would like to it with PyTorch among this week.

**Thoughts:** Today I was expecting to had much more time to focus on the project, code and research but I had some dutties to do today. Thinking about the learning schedule of the next week.

**Link(s) to article:**
[Adversarial Attacks and Defences for Convolutional Neural Networks](https://medium.com/onfido-tech/adversarial-attacks-and-defences-for-convolutional-neural-networks-66915ece52e7) 

**Link to work:**
[FGSM](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/FGSM.ipynb)



## Day 44: May 13, 2019
##### (DL - Drone madness)

**Today's Progress**: Started a notebook from PyTorch's tutorial called PyTorch by example. It is really nice to review and solidify some concepts. Found a couple of month ago bought an ebook about DL which I didn't remember and started to read it.

**Thoughts:** Still thinking on the next steps of my learning path. I found a couple of interesting courses but they teach TensorFlow. I think maybe is too really to switch to another framework, I would prefer to stick with PyTorch till I begin feeling pretty comfortable with it.

**Link(s) to article:**
[Facebook is Making Deep Learning Experimentation Easier With These Two New PyTorch-Based Frameworks](https://towardsdatascience.com/facebook-is-making-deep-learning-experimentation-easier-with-these-two-new-pytorch-based-frameworks-5e29754bb8de) 

**Link to work:**
[PyTorch by Example](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/PyTorch-Examples.ipynb)


## Day 45: May 14, 2019
##### (DL - Drone madness)

**Today's Progress**: Finished the notebook from PyTorch's tutorial. Started a course called "Mathematical foundation dor AI and machine learning". I'm gonna try to do this course 2-3 times per week to review and solidify some math's concepts.

**Thoughts:** Since I've already gain depth understanding on PyTorch, tomorrow I would like to come back to the Tabular scripts ans try to improve its performance.

**Link(s) to article:**
[Implementing SPADE using fastai](https://towardsdatascience.com/implementing-spade-using-fastai-6ad86b94030a) 

**Link to work:**
[PyTorch by Example](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/PyTorch-Examples.ipynb)


## Day 46: May 15, 2019
##### (DL - Drone madness)

**Today's Progress**: Started another notebook from Pytorch's tutorials. Yesterday I bought a bunch of books about AI, so I've been taking a look to see which ones can be added to my learning path in the next days.

**Thoughts:** Since I want to understand every single detail in depth, I'm aware math should be one of my strongest skills so I'm thinking to reviwed all the lessons about maths I took in the university. In the incoming days I want to do focus on that. 

**Link(s) to article:**
[The Blunt Guide to Mathematically Rigorous Machine Learning](https://medium.com/technomancy/the-blunt-guide-to-mathematically-rigorous-machine-learning-c53263d45c7b) 

**Link to work:**
[PyTorch by Example](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/Torchnn.ipynb)


## Day 47: May 16, 2019
##### (DL - Drone madness)

**Today's Progress**: Back to research about tabular data in DL since this weekend I'm going to participate in a datathon. Read a paper about the winning algorithm in a kaggle competition and some articles and post forums. Continued a bit with the note I started yesterday.

**Thoughts:** Today we had an event at the offices so I arrived really late at night with almost no time to code but at least I was able to understand some key concepts about tabular data and NN.

**Link(s) to article:**
[Neural Network Embeddings Explained](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526) 

**Link to work:**
[PyTorch by Example](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/Torchnn.ipynb)


## Day 48: May 17, 2019
##### (DL - Drone madness)

**Today's Progress**: Since tomorrow starts the hackathon weekend I couldn't extend my day till late. Continued with the notebook but had some weird error I couldn't figured it out what was going on. Thanks to that error I researched in depth the graph computation of PyTorch being able to understand what's going on under the hood.

**Thoughts:** Not much to say, looking forward to start a super AI weekend coding all day-night long.
 

**Link to work:**
[Torch nn module in depth](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/Torchnn.ipynb)



## Day 49: May 18, 2019

**Today's Progress**: First day of the datahon! It is based on Survival Analysis, an approach I'd never heard before, in order to predict and classify better melanoma's cancer. We had a very interesting few talks about the topic in order to better understand what is the goal. We started with an initial script an a example of a model. Spent the morning understanding the code and researching about survival analysis to better understand the possibilities we had to proceed.

**Thoughts:** It is my first datathon, I had pretty much fun the first day and also, most important. Since I'm in into ML I've benn focused on DL leaving aside the basics algorithms and other super important skills such as cleaning data, eda, etc. so I improved a lot this skills.
 

**Link to work:**
[Datathon Work](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Datathon)



## Day 50: May 19, 2019

**Today's Progress**: Last day of the datahon! Had pretty much fun!. The results weren't like expected since I had a huge faith that DL maybe was the saint grail but.. I spent a lot of time undertanding the Survival analysis approach in order to implement a NN, actually I was able to code the architecture and when I was researching on how to code the accuracy I found a specific library for survical analysis working with PyTorch, so I switched to work with this library called PySurvival, I tried 2 different models (NN && RF) but didn't got a better accuracy that we had before. Finally we ended up around the middle of the classification (6/14).

**Thoughts:** I had a lot of fun thsi weekend and earnt many many things. I understood the importance of the data analysis and experienced how the life as a data scientist should be and I liked it. I arrived to the conclussion that I need to focus as well in the other skills I didn't focus before. Pretty excited thinking about the way awaits me in this amazing field.
 

**Link to work:**
[Datathon Work](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/Datathon)



## Day 51: May 20, 2019

**Today's Progress**: Started AI for everyone course from Coursera by Andrew Ng, already did week 1 and 2. Solved the problem I had the other day with the torchnn's notebook and continued progressing with it.

**Thoughts:** Planning to enroll to DataCamp for one year in order to improve my EDA an Data Cleaning skills and also to improve my knowledge outside the area of DL, that's why I also started the Andrew Ng's course which I'm planning to end it by wednesday since it is just video theorical lessons.
 

**Link to work:**
[Torch nn module in depth](https://github.com/3lv27/100DaysOfMLCode/blob/master/notebooks/PyTorch/Torchnn.ipynb)


## Day 52: May 21, 2019

**Today's Progress**: Viewed a bunch of videos about decision trees, random forest, adaboost, gradient boosting, etc. and read related articles. Started a notebook about survival analisis.

**Thoughts:** Started to learn about the classical approaches of ML, I was wrong thinking that classical ML should have less complexity than NN, there a lot of new concepts to learn, and when I say a lot it means a looooot. Also I'm improving my data cleaning skills wprking with non public datasets.
 
**Link(s) to article:**
[CatBoost vs. Light GBM vs. XGBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db) | [Two New Frameworks that Google and DeepMind are Using to Scale Deep Learning Workflows](https://towardsdatascience.com/two-new-frameworks-that-google-and-deepmind-are-using-to-scale-deep-learning-workflows-33500b05b3f7)
