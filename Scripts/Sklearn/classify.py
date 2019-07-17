import numpy
from sklearn import svm
import argparse
import cv2
import glob
import pandas

def loadImages(filenames):
	"""
	Load image files as grey flat data arrays
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


dataDir = '../../Data/Synthetic/Dots/'
trainingDir = dataDir + 'train/'

trainingLabels = pandas.read_csv(trainingDir + 'train.csv')
trainingOutput = trainingLabels['numberOfDots'][:]
trainingInput = loadImages(glob.glob(trainingDir + 'img*.jpg'))

# now train the data
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(trainingInput, trainingOutput)

# test
testingDir = dataDir + 'test/'
testingLabels = pandas.read_csv(testingDir + 'test.csv')
testingOutput = testingLabels['numberOfDots'][:]
testingInput = loadImages(glob.glob(testingDir + 'img*.jpg'))

numDots = clf.predict(testingInput)

# compute score
diffs = (numDots - testingOutput)**2
score = diffs.sum()
numFailures = (diffs != 0).sum()

print('score = {} number of failures = {}'.format(score, numFailures))






