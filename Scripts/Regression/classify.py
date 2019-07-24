import numpy
from sklearn import linear_model, metrics
import argparse
import cv2
import glob
import pandas
import re

parser = argparse.ArgumentParser(description='Count the dots.')
parser.add_argument('--seed', type=int, default=13435, 
                    help='Random seed')
parser.add_argument('--trainDir', default='../../Data/Synthetic/Dots/train/',
                    help='Path to the training data directory')
parser.add_argument('--testDir', default='../../Data/Synthetic/Dots/test/',
                    help='Path to the testing data directory')
parser.add_argument('--plot', action='store_false',
                    help='Turn off plotting')


args = parser.parse_args()


numpy.random.seed(args.seed)


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
    inputData = numpy.zeros((numImages, n0, n1), numpy.float32)
    for i in range(numImages):
        fn = filenames[i]
        # extract the index from the file name, note: the index starts with 1
        index = int(re.search(r'img(\d+).jpg', fn).group(1)) - 1
        im = cv2.imread(fn)
        inputData[index,...] = im.mean(axis=2) / 255.
    return inputData

def getImageSizes(filename):
    """
    Get the number of x and y pixels
    @parameter filename file name
    @return nx, ny
    """
    im = cv2.imread(filename)
    return im.shape[:2]


trainingDir = args.trainDir

df = pandas.read_csv(trainingDir + 'train.csv')
categories = df['numberOfDots'].unique()
categories.sort()
minNumDots = min(categories)
maxNumDots = max(categories)
numCategories = maxNumDots - minNumDots + 1
# labels start at zero
trainingOutput = (numpy.array(df['numberOfDots'], numpy.float32) - minNumDots)/(maxNumDots - minNumDots)
trainingInput = loadImages(glob.glob(trainingDir + 'img*.jpg'))

testingDir = args.testDir

df = pandas.read_csv(testingDir + 'test.csv')
numCategories = len(categories)
# labels start at zero
testingOutput = (numpy.array(df['numberOfDots'], numpy.float32) - minNumDots)/(maxNumDots - minNumDots)
testingInput = loadImages(glob.glob(testingDir + 'img*.jpg'))

# train the model
n0, n1 = getImageSizes(trainingDir + 'img1.jpg')
print('Number of training images: {}'.format(trainingInput.shape[0]))
print('Number of testing images : {}'.format(testingInput.shape[0]))
print('Image size               : {} x {}'.format(n0, n1))
print('Categories               : {} min/max = {}/{}'.format(categories, minNumDots, maxNumDots))


clf = linear_model.LinearRegression()

# now train
clf.fit(trainingInput.reshape(-1, n0*n1), trainingOutput)

# test
predictions = clf.predict(testingInput.reshape(-1, n0*n1))

numPredictions = predictions.shape[0]
predictedNumDots = (maxNumDots - minNumDots)*predictions + minNumDots
exactNumDots = (maxNumDots - minNumDots)*testingOutput + minNumDots

# compute varError
diffs = (numpy.round(predictedNumDots) - exactNumDots)**2
varError = diffs.sum()
numFailures = (diffs != 0).sum()

print('sum of errors squared = {} number of failures = {} ({} %)'.format(varError, numFailures, 100*numFailures/numPredictions))

print('known number of dots for the first 5 images   : {}'.format(exactNumDots[:5]))
print('inferred number of dots for the first 5 images: {}'.format(predictedNumDots[:5]))

# plot training/test dataset
from matplotlib import pylab
n = 50
for i in range(n):
    pylab.subplot(n//10, 10, i + 1)
    pylab.imshow(testingInput[i,...])
    titleColor = 'black'
    if int(exactNumDots[i]) != numpy.round(predictedNumDots[i]):
        titleColor = 'red'
    pylab.title('{} ({:.1f})'.format(int(exactNumDots[i]), predictedNumDots[i]),
        fontsize=8, color=titleColor)
    pylab.axis('off')
if args.plot:
    pylab.show()
else:
    pylab.savefig('someResults.png', dpi=300)
