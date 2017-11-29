# Behavioral Cloning 

Shon Katzenberger  
shon@katzenberger-family.com  
December 2, 2017  
October, 2017 class of SDCND term1

## Assignment

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/gui_00.png "GUI"
[image2]: ./images/steer_left_00.jpg "Steer Left"
[image3]: ./images/steer_right_00.jpg "Steer Right"
[image4]: ./images/uncropped_center.jpg "Uncropped Center"
[image5]: ./images/cropped_left.jpg "Cropped Left"
[image6]: ./images/cropped_center.jpg "Cropped Center"
[image7]: ./images/cropped_right.jpg "Cropped Right"

## Submission Details

The submission consists of the following core files:
* WRITEUP.md: This file.
* model.py: Contains the script to load data and create and train the model.
* drive.py: Provided by Udacity (unchanged), for driving the car in autonomous mode.
* model.h5: The final trained model.
* video.mp4: A video of the model completing several laps around track one.

I've included a few additional files that may be interesting (particular the first):
* gui_show.py: This implements a GUI app to view either a training dataset (generated via the simulator), or a directory containing images. See below for more detail.
* notes.txt: Contains some notes that I took while training models. Includes some level of detail of the contents of the models folder.
* models: This folder contains several models. All successfully navigate track one, although the second one crosses some lines. See below for more detail.
* images: This folder contains images used by WRITEUP.md.
* .pylintrc: The pylint settings file.
* README.md, video.py, writeup_template.md: These are the unaltered files supplied by Udacity.

### Model Architecture

The model architecture is defined in the `buildModel` function in `model.py`.

The architecture I use is the one introduced in the lectures as being published by the NVidia Autonomous Vehicles Team.
It consists of three 5 x 5 convolutions with stride 2, followed by two 3 x 3 convolutions with stride 1, followed by three
fully connected layers. The convolutional layers do not use padding. Each layer except the last uses `relu` for activation.

I add dropout before all but the first layer, with the convolutional layers sharing one droput rate
and the fully connected layers sharing another drop out rate. For the final model, I used a dropout rate of 0.1 for convolutional
layers and 0.5 for fully connected layers. Note that the dropout
for the convolutional layers are spatial, meaning that they keep or drop an entire channel, not individual cells within a channel.

Before the first convolutional layer, the model crops to remove the top 42% and bottom 15% of the image. This helps focus the
model on the road, eliminating non-essential detail like the position of particular trees, hills, or other landscape features.
It also removes the hood of the vehicle, so the model cannot easily distinguish between the different camera images (center, left,
and right). Here is an example of a center camera image, followed by corresponding cropped left, center, and right camera images:

![alt text][image4]

![alt text][image5]
![alt text][image6]
![alt text][image7]

Notice that the hood is not visible in the cropped images.

After the cropping, but before the first convolutional layer, I normalize the input via the affine function mapping the interval [0, 255]
to the interval [-0.5, 0.5].

Here's a summary of the layers, their shapes, and descriptions:

* Input: (160, 320, 3)
* Crop: (69, 320, 3), removes the top 42% and bottom 15% of the input image
* Lambda: (69, 320, 3), converts to float32 and maps values to the interval [-0.5, +0.5]
* Convolution: (33, 159, 24), has 24 feature maps of size 5 x 5, with stride (2, 2).
* Convolution: (15, 77, 36), has 36 feature maps of size 5 x 5, with stride (2, 2).
* Convolution: (6, 37, 48), has 48 feature maps of size 5 x 5, with stride (2, 2).
* Convolution: (4, 35, 64), has 64 feature maps of size 3 x 3, with stride (1, 1).
* Convolution: (2, 33, 64), has 64 feature maps of size 3 x 3, with stride (1, 1).
* Fully Connected: (100,)
* Fully Connected: (50,)
* Fully Connected: (1,)

The total number of trainable parameters (weights and biases) is 558,949.

### Data

#### Generating and Cleaning Data

I had very little success generating quality data using the simulator. I recorded several laps of data, and had my video-game-playing
eldest son record some laps as well. Eventually I turned to using the sample data, augmented with some short corrective clips. To get
high quality corrective clips, I manually tweaked many of the angles recorded in the .csv files. This was a laborious process and
very time consuming.

To help with reviewing the quality of clips, I wrote a GUI application to visualize datasets. Here's a screen shot:

![alt text][image1]

The code is in `gui_show.py` and I encourage the reviewer to try it out. To load a driving log, click on `Load Data Set...` and
select the `driving_log.csv` file. To load a directory of images (for example, the output from running in autonomous mode), click
on `Load Pictures...` and select the directory. In either case, checking `Run` will rapidly scroll through the images (faster
than holding down the scroll bar arrow). When a dataset is loaded, the steering angle is displayed next to `Extra:`. I didn't
bother making this show the other two camera images.

Note that this application was seeded from code I'd written outside of this class, and didn't take long to adapt for this project.

Now, let's return to discussing data. The sample data turned out to be quite complete, but had some very poor labels (especially
near the beginning). I deleted several rows where the angles were not good and fixed up other angles. For example, I corrected
the labels of a segment on the bridge (starting at line 91 in training_log.csv), where the labels suggest a hard right turn for
several frames, followed by several 0-angle frames. This encouraged the model to steer hard toward the right rail of the bridge.
Fixing these labels made a marked improvement. There were a few portions of the example data that I simply deleted, because the
labels were bad, and similar situations were covered by other clips.

Most of the corrective clips that I recorded were generated before I discovered the RGB/BGR issue discussed below. Once that
was corrected, most of them were no longer needed. I did end up using a dataset with several clips moving from an edge of
the road toward the center. This dataset involved a fair amount of manual tweaking of angle values in the .csv file. Here are a couple
example images with steering angles of roughly -0.18 and +0.11, respectively:

![alt text][image2]

![alt text][image3]

---

There is an alternative approach to data generation that would have been much more efficient and would have produced much higher quality data.
This approach involves modifying the simulator to add an "automatic training" mode. In this mode, the simulator, which knows the
path of the road, would place the vehicle at a random selection of positions and orientations on the track, and would compute the ideal
steering angle from that information. It would then render and save the image(s) along with the ideal angle. Computing these ideal angles
would be much simpler than manually wrestling the simulator controls to get the vehicle into a comprehensive set of positions/orientations
with ideal steering angles.

Since data collection / generation was so laborious, I didn't have time to use the second track. If I had more time to spend on this,
I would invest in modifying the simulator to have the automatic training mode, and run it on many tracks.

#### Loading the Data

I wrote code to load all the data into numpy arrays, as well as code implementing a shuffling generator. I trained on my MSI laptop
that has 64GB of RAM, so never had to use the generator for actual training. See the `loadImage`, `loadCsv`, `loadDataSet`,
`loadDataSetGenerator`, and `loadMultipleDataSets` functions in `model.py`.

Initially, my code to load images followed the advice of the lectures and used `cv2.imread`. This was a mistake. Only after
spending tons of time training models (some of which are in the models folder), did I realize that `driver.py` invokes the model
with RGB images! Of course, `cv2.imread` produces BGR images, so my models were being trained on very different data than what
`driver.py` was invoking them on! I had assumed (silly me) that since the lecture suggested using `cv2.imread` that `driver.py`
would be invoking the model with BGR images. Honestly, I'm a bit annoyed that the provided `driver.py` used RGB, since this cost
me many hours.

Interestingly, I was able to train several models on BGR images that worked on RGB images. In particular, the first three models in
the models folder (r1.h5, r2.h5, and r3.h5) were trained on BGR images. They all successfully navigate the track, although r2.h5 crosses
lines a few times.

Once I changed my image loading code to use RGB images, I didn't need most of the corrective clips to generate successful models.

The data loading code supports horizontally flipping images (negating the label), as well as using all three camera angles
(with settable constant label adjustment). For the final model, I used both techniques, with an angle adjustment of 0.1 when using
the side cameras.

#### Training

The final model training used:

* the `adam` optimizer (with automatic learning rate).
* mean squared error loss function.
* five epochs.
* both horizontal flipping of images and side cameras with an angle adjustment of 0.1.
* the modified example dataset (with 8005 rows) plus a supplementary dataset (with 2770 rows).
* an 80% / 20% split for train / validation data.

The cobmined dataset consisted of `6(8005 + 2770) = 64,650` images. This resulted in `51,720` training images and `12,930` validation
images. I realized (by reading the keras code) that keras' `model.fit` function doesn't shuffle before splitting the data into train and
validation portions, so I manually shuffled the data before invoking `model.fit`. Note that the ideal would be to use entirely separate (but
somewhat complete) clips for validation data, since otherwise, the training and validation data can easily contain almost identical
examples (eg, adjacent frames when the vehicle is stationary or moving very slowly). Since data generation was so time consuming, I didn't pursue this.

Here's the console output, showing training and validation loss, from training of the final model:

```
(carnd-term1) F:\sdc\sdc_proj_3>python model.py
Loading dataset from: ./data1\driving_log.csv
  Adding side camera images with angle adjustment: 0.1
Loading dataset from: ./data2\driving_log.csv
  Adding side camera images with angle adjustment: 0.1
Image info (64650, 160, 320, 3) uint8
Angle info (64650,) float32
Using TensorFlow backend.
Train on 51720 samples, validate on 12930 samples
Epoch 1/5
2017-12-02 15:19:53.872165: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-12-02 15:19:53.872452: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-12-02 15:19:54.157655: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:955] Found device 0 with properties:
name: GeForce GTX 1070
major: 6 minor: 1 memoryClockRate (GHz) 1.645
pciBusID 0000:02:00.0
Total memory: 8.00GiB
Free memory: 6.65GiB
2017-12-02 15:19:54.157808: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:976] DMA: 0
2017-12-02 15:19:54.161223: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:986] 0:   Y
2017-12-02 15:19:54.161352: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:02:00.0)
51720/51720 [==============================] - 58s - loss: 0.0113 - val_loss: 0.0095
Epoch 2/5
51720/51720 [==============================] - 55s - loss: 0.0099 - val_loss: 0.0086
Epoch 3/5
51720/51720 [==============================] - 55s - loss: 0.0096 - val_loss: 0.0080
Epoch 4/5
51720/51720 [==============================] - 55s - loss: 0.0094 - val_loss: 0.0081
Epoch 5/5
51720/51720 [==============================] - 56s - loss: 0.0092 - val_loss: 0.0078
```

Five epochs were sufficient to get a successful model.

The `note.txt` file shows similar output for the models in the `models` folder.
