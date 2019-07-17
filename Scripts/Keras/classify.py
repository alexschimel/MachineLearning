import numpy
import tensorflow as tf
from tensorflow import keras
import argparse
import cv2
import glob
import pandas

def loadImages(filenames):
	"""
	Load image files as grey data arrays
	@param filenames list of jpg file names
	@return array of grey pixel data (1=white, 0=black)
	"""
	# open first file to get the image size
	im = cv2.imread(filenames[0])
	n0, n1 = im.shape[:2]
	numImages = len(filenames)
	inputData = numpy.zeros((numImages, n0*n1), numpy.float32)
	for i in range(numImages):
		im = cv2.imread(filenames[i])
		inputData[i,:] = (im.mean(axis=2)/255.).flat
	return inputData

def getImageSizes(filename):
	"""
	Get the number of x and y pixels
	@parameter filename file name
	@return nx, ny
	"""
	im = cv2.imread(filename)
	return im.shape[:2]


dataDir = '../../Data/Synthetic/Dots/'
trainingDir = dataDir + 'train/'

df = pandas.read_csv(trainingDir + 'train.csv')
categories = df['numberOfDots'].unique()
minNumDots = min(categories)
maxNumDots = max(categories)
numCategories = maxNumDots - minNumDots + 1
# labels start at zero
trainingOutput = numpy.array(df['numberOfDots']) - minNumDots
trainingInput = loadImages(glob.glob(trainingDir + 'img*.jpg'))

testingDir = dataDir + 'test/'
df = pandas.read_csv(testingDir + 'test.csv')
numCategories = len(categories)
# labels start at zero
testingOutput = numpy.array(df['numberOfDots']) - minNumDots
testingInput = loadImages(glob.glob(testingDir + 'img*.jpg'))

# train the model
n0, n1 = getImageSizes(trainingDir + 'img1.jpg')
print('Number of training images: {}'.format(trainingInput.shape[0]))
print('Number of testing images : {}'.format(testingInput.shape[0]))
print('Image size               : {} x {}'.format(n0, n1))
print('Categories               : {}'.format(categories))

clf = keras.Sequential([
    #keras.layers.Flatten(input_shape=(n0, n1)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(numCategories, activation=tf.nn.softmax)
])

clf.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
# now train
clf.fit(trainingInput, trainingOutput, epochs=5)

# test
predictions = clf.predict(testingInput)

# predictions returns an array of probabilities for each label
bestGuessInds = numpy.argmax(predictions, axis=1)
cats = numpy.array([minNumDots + i for i in range(numCategories)])
numDots = cats[bestGuessInds]

# compute score
diffs = (numDots - testingOutput)**2
score = diffs.sum()
numFailures = (diffs != 0).sum()

print('score = {} number of failures = {}'.format(score, numFailures))






