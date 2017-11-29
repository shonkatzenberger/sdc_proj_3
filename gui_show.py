#!/usr/bin/env python

""" Gui app for displaying images from a directory, or images + angles from a driving log.
"""

# This is based on code that I wrote for other image-related tasks, which is why it's a bit
# more general (in some places) than this project warrants.

# Note that this is useful, both for reviewing training datasets, and for viewing images
# saved by drive.py (before making the video). For the latter, checking the 'Run' box
# is particularly useful.
#
# I used this to review the contents of datasets and to locate poorly labeled images.
# I either manually tweaked bad labels or deleted the row from the csv file.
# I didn't bother deleting the corresponding images.

import os
import sys
import logging

import tkinter as _tk
import tkinter.filedialog as _dlg

import PIL.Image as _img
import PIL.ImageTk as _imgtk
import numpy as np

# Import the model file so we can load datasets.
import model as _mod

# pylint: disable=missing-docstring

class Application(_tk.Frame):
  """ The application class. """

  def __init__(self, logger, master=None):
    _tk.Frame.__init__(self, master)

    self._logger = logger

    # Image and canvas size fields.
    self._dyMin = 160
    self._dxMin = 320
    self._dy = self._dyMin
    self._dx = self._dxMin

    # The ImageData object.
    self._data = None

    # Millisecond delay when 'Run' is checked.
    self._delay = 1

    # Position within the dataset and size of the dataset. Used for scrolling.
    self._pos = -1
    self._count = -1

    self.grid()
    self._createWidgets()
    self._idImage = None

  def _createWidgets(self):
    """ Create the widgets. """
    padx = 10
    pady = 5

    frame = _tk.Frame(master=self, borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=0)

    self.quitButton = _tk.Button(master=frame, text='Quit', command=self.quit)
    self.loadDataSetButton = _tk.Button(master=frame, text='Load Data Set...', command=self.loadDataSet)
    self.loadPicturesButton = _tk.Button(master=frame, text='Load Pictures...', command=self.loadPictures)

    self.quitButton.pack(side=_tk.LEFT, padx=padx, pady=pady)
    self.loadDataSetButton.pack(side=_tk.LEFT, padx=padx, pady=pady)
    self.loadPicturesButton.pack(side=_tk.LEFT, padx=padx, pady=pady)

    frame = _tk.LabelFrame(master=self, text='Item', borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=pady)

    frame = _tk.Frame(master=self, borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=0)

    self.runVar = _tk.IntVar(master=frame, value=0)
    self.runCheck = _tk.Checkbutton(master=frame, variable=self.runVar, text='Run', command=self._runToggle)

    self.runCheck.pack(side=_tk.LEFT, padx=padx, pady=pady)

    frame = _tk.LabelFrame(master=self, text='Item', borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=pady)

    row = 0

    _tk.Label(master=frame, text='Index:').grid(row=row, column=0, padx=padx, sticky=_tk.E)
    self.indexLabel = _tk.Label(master=frame, text='', anchor=_tk.W)
    self.indexLabel.grid(row=row, column=1, sticky=_tk.W)
    row += 1

    _tk.Label(master=frame, text='Id:').grid(row=row, column=0, padx=padx, sticky=_tk.E)
    self.idLabel = _tk.Label(master=frame, text='', anchor=_tk.W)
    self.idLabel.grid(row=row, column=1, sticky=_tk.W)
    row += 1

    _tk.Label(master=frame, text='Dimensions:').grid(row=row, column=0, padx=padx, sticky=_tk.E)
    self.dimsLabel = _tk.Label(master=frame, text='', anchor=_tk.W)
    self.dimsLabel.grid(row=row, column=1, sticky=_tk.W)
    row += 1

    _tk.Label(master=frame, text='Extra:').grid(row=row, column=0, padx=padx, sticky=_tk.E)
    self.extraLabel = _tk.Label(master=frame, text='', anchor=_tk.W)
    self.extraLabel.grid(row=row, column=1, sticky=_tk.W)
    row = None

    frame = _tk.Frame(master=self, borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=pady)

    # Create a sub-frame around the scroll bar and another frame that sets the width.
    # REVIEW shonk: Is there a better way?
    sub = _tk.Frame(master=frame)
    sub.pack(side=_tk.LEFT, padx=padx, pady=pady)
    _tk.Frame(master=sub, height=0, width=400).pack()
    self.dataScroll = _tk.Scrollbar(master=sub, jump=1, orient=_tk.HORIZONTAL, command=self.scrollData)
    self.dataScroll.pack(fill=_tk.X)

    self.dataLabel = _tk.Label(master=frame, text='', anchor=_tk.W, width=20)
    self.dataLabel.pack(side=_tk.LEFT, padx=padx, pady=pady)

    self.canvas = _tk.Canvas(master=self, height=self._dy, width=self._dx, borderwidth=3)
    self.canvas.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=pady)

  def close(self):
    self._stop()
    if self._data is not None:
      self._data.close()
      self._data = None

  def quit(self):
    self._logger.debug("Quitting...")
    _tk.Frame.quit(self)

  def loadDataSet(self):
    f = _dlg.askopenfilename(
      parent=self, title='Choose a driving_log.csv file', filetypes=(("Driving Log", ".csv"),))
    if f is None or not os.path.isfile(f):
      return
    data = ImageDataFromCsv(self._logger, f)
    self._setDataSet(data)

  def loadPictures(self):
    d = _dlg.askdirectory(
      parent=self, title='Choose an image directory',
      mustexist=True)
    if d is None or not os.path.isdir(d):
      return
    data = ImageDataFromPictures(self._logger, d)
    self._setDataSet(data)

  def _setDataSet(self, data):
    assert isinstance(data, ImageData)
    if data.count == 0:
      return
    if self._data is not None:
      self._data.close()
      self._data = None
    self._data = data
    self.jumpData(0)

  @property
  def _running(self):
    return self.runVar.get() != 0

  def _runToggle(self):
    if self._data is not None:
      self._data.setRunning(self._running)
    if self._running:
      self.after(self._delay, self._fetch)

  def _stop(self):
    self.runVar.set(0)
    if self._data is not None:
      self._data.setRunning(self._running)

  def _fetch(self):
    if not self._running:
      return

    if not self._data.next():
      self._stop()
      return

    self._setImage()
    self.after(self._delay, self._fetch)

  def _setImage(self):
    image = self._data.getImage()
    if self._idImage is not None:
      self.canvas.delete(self._idImage)
      self._idImage = None
    dy, dx, _ = self._data.pixels.shape
    dyUse = max(self._dyMin, dy)
    dxUse = max(self._dxMin, dx)
    if dyUse != self._dy or dxUse != self._dx:
      self.canvas.config(height=dyUse, width=dxUse)
      self._dy = dyUse
      self._dx = dxUse
    self._idImage = self.canvas.create_image(self._dx // 2, self._dy // 2, image=image)

    pos = self._data.index
    count = self._data.count
    if count <= 0:
      count = -1
      self.dataLabel.config(text="")
      # Position the scroll bar.
      self.dataScroll.set(0, 1)
    else:
      self.dataLabel.config(text="{0} of {1}".format(pos + 1, count))
      # Position the scroll bar.
      x = pos * 7 / (8.0 * max(1, count - 1))
      self.dataScroll.set(x, x + 0.125)

    self._pos = pos
    self._count = count

    self.indexLabel.config(text="{}".format(pos))
    self.idLabel.config(text="{}".format(self._data.id))
    self.dimsLabel.config(text="{0} by {1}".format(dx, dy))

    extra = self._data.extra
    assert extra is None or isinstance(extra, tuple)
    if extra is None or len(extra) == 0:
      self.extraLabel.config(text="")
    else:
      text = ', '.join('{}'.format(x) for x in extra)
      self.extraLabel.config(text=text)

  def nextData(self):
    self._data.next()
    self._setImage()

  def prevData(self):
    self._data.prev()
    self._setImage()

  def jumpData(self, index):
    self._data.jumpTo(index)
    self._setImage()

  def scrollData(self, kind, value, unit=None):
    count = self._count
    if count <= 0:
      return

    pos = self._pos

    if kind == _tk.SCROLL:
      v = int(value)
      assert v == -1 or v == 1, "Unexpected value: {0}".format(v)
      if unit == _tk.UNITS:
        if v < 0 and pos > 0:
          self.prevData()
        elif v > 0 and pos < count - 1:
          self.nextData()
      elif unit == _tk.PAGES:
        inc = max(1, int(round(count / 10)))
        posNew = max(0, min(count - 1, pos + v * inc))
        if posNew != pos:
          self.jumpData(posNew)
    elif kind == _tk.MOVETO:
      v = float(value)
      posNew = max(0, min(count - 1, int(v * (count - 1) * 8 / 7.0)))
      self.jumpData(posNew)

class ImageData(object):
  """ Base class for an image collection. """
  def __init__(self):
    super(ImageData, self).__init__()

    self._index = -1
    self._id = None
    self._image = None
    self._imagetk = None
    self._pixels = None
    self._extra = None
    self._seekable = True

  def close(self):
    pass

  def setRunning(self, running):
    pass

  @property
  def count(self):
    if not self._seekable:
      return -1
    return self._countCore

  @property
  def _countCore(self):
    pass

  @property
  def index(self):
    return self._index

  # pylint: disable=C0103
  @property
  def id(self):
    return self._id

  @property
  def pixels(self):
    return self._pixels

  @property
  def extra(self):
    return self._extra

  def getImage(self):
    self._image = _img.fromarray(self._pixels, 'RGB')
    self._imagetk = _imgtk.PhotoImage(self._image)
    return self._imagetk

  def jumpTo(self, index):
    if self._seekable:
      index = max(0, min(self.count - 1, index))
    return self._loadPixels(index)

  def next(self):
    index = self._index + 1
    if self._seekable:
      index = min(self.count - 1, index)
    return self._loadPixels(index)

  def prev(self):
    index = max(0, self._index - 1)
    return self._loadPixels(index)

  def _loadPixels(self, index):
    pass

class ImageDataFromCsv(ImageData):
  """ Loads a driving log .csv file, together with the associated images. The images are loaded lazily (on-demand). """
  def __init__(self, logger, path):
    super(ImageDataFromCsv, self).__init__()

    logger.debug("Loading .csv from: '%s'", path)

    self._logger = logger
    self._rootDir = os.path.dirname(path)
    self._csvFileName = os.path.basename(path)
    self._count, self._shapeImage, self._dtypeImage, self._imageFileNames, self._angles = _mod.loadCsv(
      self._rootDir, fileName=self._csvFileName, multiCameras=False)

  @property
  def _countCore(self):
    return self._count

  def _loadPixels(self, index):
    assert 0 <= index < self.count
    if index == self._index:
      assert self._pixels is not None
      return False

    self._logger.debug("Loading: '%s'", self._imageFileNames[index])
    pixels = _mod.loadImage(self._rootDir, self._imageFileNames[index])

    self._pixels = pixels
    self._id = self._imageFileNames[index]
    self._extra = (self._angles[index],)
    self._index = index

    return True

class ImageDataFromPictures(ImageData):
  """ Loads all images in a directory. The images are loaded lazily (on-demand). """
  def __init__(self, logger, path):
    super(ImageDataFromPictures, self).__init__()

    logger.debug("Loading pictures from: '%s'", path)

    self._logger = logger
    self._dir = None
    self._files = None
    self._findFiles(path)

  def _findFiles(self, path):
    self._dir = path
    self._files = list()
    for name in os.listdir(self._dir):
      if name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png'):
        self._files.append(name)

  @property
  def _countCore(self):
    return len(self._files)

  def _loadPixels(self, index):
    assert 0 <= index < self.count
    if index == self._index:
      assert self._pixels is not None
      return False

    self._logger.debug("Loading: '%s'", self._files[index])
    raw = _img.open(os.path.join(self._dir, self._files[index]))
    if raw.mode != 'RGB':
      self._logger.debug("Converting from %s to %s", raw.mode, 'RGB')
      raw = raw.convert('RGB')

    self._pixels = np.asarray(raw)
    self._id = self._files[index]
    self._index = index

    return True

def _run(logger):
  app = Application(logger)
  app.master.title('Image viewing application')

  # Load initial data set, once the mainloop is spun up.
  app.after(1, app.loadDataSet)

  logger.debug('Entering mainloop')
  app.mainloop()
  logger.debug('Exited mainloop')

  app.close()
  logger.debug('Closed')

def initConsoleLogger(loggerName, verbosity=1):
  """ Set up console-based logging and return the logger. """

  logging.basicConfig(
    level=logging.DEBUG if verbosity >= 2 else logging.INFO if verbosity >= 1 else logging.WARN,
    format='[%(asctime)s %(levelname)s %(name)s]  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
  logger = logging.getLogger(loggerName)
  logger.info("Python version: %s", sys.version)

  return logger

def run(_):
  """ Run the script. """
  logger = initConsoleLogger('gui_show', verbosity=1)
  _run(logger)

if __name__ == "__main__":
  run(sys.argv[1:])
