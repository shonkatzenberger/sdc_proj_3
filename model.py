""" Data loading and model building for Behaviorial Cloning project.
Authored by Shon Katzenberger.
"""
import os
import sys
import csv
import numpy as np

import PIL.Image as _img

# pylint: disable=C0103

def loadImage(dataDir, name):
  """ Load an image, given the directory containing the .csv and the name of the image file. """
  path = os.path.join(dataDir, 'IMG', name)
  # NOTE: The lecture suggested using cv2.imread to read the images. That loads the images
  # as BGR images, while the driver.py file invokes the model on RGB images. Unfortunately,
  # I wasted a lot of time training models before realizing this.
  raw = _img.open(path)
  if raw.mode != 'RGB':
    print("WARNING: Converting {} from {} to {}".format(name, raw.mode, 'RGB'))
    raw = raw.convert('RGB')
  image = np.asarray(raw)
  return image

def loadCsv(dataDir, fileName='driving_log.csv', multiCameras=True, adjust=0.01):
  """ Loads the dataset csv. Returns the row-count, image shape, image dtype, image file names, and angles.
  This supports using all three cameras (when multiCameras is True) or just the center camera. In the former
  case, the angles corresponding to the side cameras are adjusted by the given 'adjust' value.
  """

  path = os.path.join(dataDir, fileName)
  print("Loading dataset from: {}".format(path))
  with open(path, 'r') as fp:
    reader = csv.reader(fp)
    lines = list(reader)
  count = len(lines)

  angles = list(float(ln[3]) for ln in lines)
  angles = np.array(angles, dtype=np.float32)
  assert angles.ndim == 1 and angles.shape[0] == count

  imageFileNames = list(os.path.basename(ln[0]) for ln in lines)
  assert len(imageFileNames) == count

  # Get the first image to determine the image shape and dtype. This assumes that all images have
  # the same shape and dtype.
  imageFirst = loadImage(dataDir, imageFileNames[0])
  assert isinstance(imageFirst, np.ndarray)
  assert imageFirst.ndim == 3

  if multiCameras:
    # Use all three cameras, so glue the side cameras on. The order matter since we'll be shuffling
    # during training. Of course, the file names and angles need to be in the same order!
    print("  Adding side camera images with angle adjustment: {}".format(adjust))
    count *= 3
    imageFileNames.extend(os.path.basename(ln[1]) for ln in lines)
    imageFileNames.extend(os.path.basename(ln[2]) for ln in lines)
    angles = np.concatenate((angles, np.add(angles, adjust), np.add(angles, -adjust)), axis=0)

  # Verify consistency of output sizes/shapes.
  assert angles.ndim == 1 and angles.shape[0] == count
  assert len(imageFileNames) == count

  return count, imageFirst.shape, imageFirst.dtype, imageFileNames, angles

def loadDataSet(dataDir, includeFlips=True, multiCameras=True, adjust=0.01):
  """ Loads the entire dataset as parallel numpy arrays. Returns images and angles. """
  count, shapeImage, dtypeImage, imageFileNames, angles = loadCsv(dataDir, multiCameras=multiCameras, adjust=adjust)

  # Load all the images into a single numpy array.
  images = np.zeros(shape=(count,) + shapeImage, dtype=dtypeImage)
  for i in range(count):
    image = loadImage(dataDir, imageFileNames[i])
    assert image.shape == shapeImage
    assert image.dtype == dtypeImage
    images[i] = image

  if includeFlips:
    # Include flipped images, negating the corresponding angles.
    images = np.concatenate((images, images[:, :, ::-1, :]), axis=0)
    angles = np.concatenate((angles, np.negative(angles)), axis=0)
  return images, angles

# NOTE: I trained and ran everything on an MSI laptop that has 64GB of RAM and two
# GTX 1070 GPUs. I never trained on more than 20 GB of data, so didn't need the generator,
# but wrote it anyway.
def loadDataSetGenerator(dataDir, includeFlips=True, multiCameras=True, adjust=0.01):
  """ Loads the csv, and returns the row-count, image shape, image dtype, and a shuffling generator function
  that yields (image, angle) pairs.
  """
  count, shapeImage, dtypeImage, imageFileNames, angles = loadCsv(dataDir, multiCameras=multiCameras, adjust=adjust)

  num = 2 * count if includeFlips else count

  def _do(rand=None):
    if rand is None:
      indices = range(num)
    else:
      assert isinstance(rand, np.random.RandomState)
      indices = rand.permutation(num)

    for i in indices:
      j = i % count
      image = loadImage(dataDir, imageFileNames[j])
      assert image.shape == shapeImage
      assert image.dtype == dtypeImage
      angle = angles[j]

      if j != i:
        # Flip the image and angle.
        image = image[:, ::-1, :]
        angle = -angle
      yield image, angle

  return num, shapeImage, dtypeImage, _do

def loadMultipleDataSets(dataDirs, includeFlips=True, multiCameras=True, adjust=0.01):
  """ Load and concatenate multiple datasets. """
  assert isinstance(dataDirs, tuple) and len(dataDirs) > 0
  imagesList = []
  anglesList = []
  for d in dataDirs:
    images, angles = loadDataSet(d, includeFlips=includeFlips, multiCameras=multiCameras, adjust=adjust)
    assert images.shape[0] == angles.shape[0]
    imagesList.append(images)
    anglesList.append(angles)
  images = np.concatenate(imagesList, axis=0)
  angles = np.concatenate(anglesList, axis=0)
  return images, angles

def buildModel(shape, dr1=0.1, dr2=0.5):
  """ Build a keras model to be trained. This uses the architecture discussed in the lecture
  that is said to be published by the NVidia Autonomous Vehicle Team.

  'shape' is the input shape, assumed to be 3 dimensional.
  'dr1' is the drop out rate for the convolutional layers.
  'dr2' is the drop out rate for the fully connected layers.
  """
  assert len(shape) == 3

  # We import keras here to avoid importing it (and a ton of other stuff) when running
  # the 'show_gui.py' script (which imports this script).
  import keras.models as _kmod
  import keras.layers as _klay

  model = _kmod.Sequential()

  # First crop and normalize the image(s).
  # Note that this is part of the model, and not part of loading the data, since it
  # needs to be done when the model is invoked by the simulator (in drive.py), and I didn't
  # want to modify drive.py and try to keep it in sync with this.

  # Ignore the top 42% and the bottom 15%.
  cropTop = int(shape[0] * 0.42)
  cropBot = int(shape[0] * 0.15)
  model.add(_klay.Cropping2D(cropping=((cropTop, cropBot), (0, 0)), input_shape=shape))

  # Use very basic image normalization to get values between -0.5 and 0.5.
  model.add(_klay.Lambda(lambda x: x / 255.0 - 0.5))

  # Do three 5x5 convolutions with stride 2.
  model.add(_klay.Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
  model.add(_klay.SpatialDropout2D(dr1))
  model.add(_klay.Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
  model.add(_klay.SpatialDropout2D(dr1))
  model.add(_klay.Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))

  # Do two 3x3 convolutions with stride 1
  model.add(_klay.SpatialDropout2D(dr1))
  model.add(_klay.Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
  model.add(_klay.SpatialDropout2D(dr1))
  model.add(_klay.Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))

  # Do three fully connected layers.
  model.add(_klay.Flatten())
  model.add(_klay.Dropout(dr2))
  model.add(_klay.Dense(100, activation='relu'))
  model.add(_klay.Dropout(dr2))
  model.add(_klay.Dense(50, activation='relu'))
  model.add(_klay.Dropout(dr2))
  model.add(_klay.Dense(1))

  return model

def _run(_):
  # Select the dataset(s) to train on.
  dataDirs = (
    '../DrivingData/data1',
    '../DrivingData/data2',
  )

  # Load the datasets and print some basic information. Don't bother using all three cameras.
  images, angles = loadMultipleDataSets(dataDirs, includeFlips=True, multiCameras=True, adjust=0.1)
  print("Image info", images.shape, images.dtype)
  print("Angle info", angles.shape, angles.dtype)

  # This code was used to test the generator code. It isn't needed for training and saving the model.
  testGenerator = False
  if testGenerator:
    count, shapeImage, dtypeImage, gen = loadDataSetGenerator('./data1')
    print("Info", count, shapeImage, dtypeImage)

    # Without shuffling.
    num = 0
    for image, _ in gen():
      assert image.shape == shapeImage
      assert image.dtype == dtypeImage
      num += 1
    assert num == count
    print("Number of rows", num)

    # With shuffling.
    num = 0
    for image, _ in gen(rand=np.random.RandomState(42)):
      assert image.shape == shapeImage
      assert image.dtype == dtypeImage
      num += 1
    assert num == count
    print("Number of rows", num)

  # Train and save the model.
  train = True
  if train:
    model = buildModel(images.shape[1:])
    model.compile(loss='mse', optimizer='adam')

    # Note that model.fit doesn't shuffle before splitting off the validation data,
    # which is really silly of it. So we have to shuffle before passing it to model.fit.
    indices = np.random.RandomState(seed=57).permutation(images.shape[0])
    imagesUse = images[indices]
    anglesUse = angles[indices]
    model.fit(imagesUse, anglesUse, nb_epoch=5, validation_split=0.2, shuffle=True)

    model.save('model.h5')

if __name__ == "__main__":
  _run(sys.argv[1:])
