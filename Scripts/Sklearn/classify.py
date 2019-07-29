import numpy
from sklearn import svm, metrics
import argparse
import cv2
import glob
import pandas
import re

parser = argparse.ArgumentParser(description='Count features.')
parser.add_argument('--seed', type=int, default=13435, 
                    help='Random seed')
parser.add_argument('--trainDir', default='../../Data/Synthetic/Dots/train',
                    help='Path to the training data directory')
parser.add_argument('--testDir', default='../../Data/Synthetic/Dots/test',
                    help='Path to the testing data directory')
parser.add_argument('--save', default='',
                    help='Turn plot in file')

args = parser.parse_args()


numpy.random.seed(args.seed)

def loadImages(filenames):
    """
    Load image files as grey data arrays
    @param filenames list of image files
    @return array of grey pixel data (1=white, 0=black)
    """
    # open first file to get the image size
    im = cv2.imread(filenames[0])
    n0, n1 = im.shape[:2]
    numImages = len(filenames)
    inputData = numpy.zeros((numImages, n0*n1), numpy.float32)
    index = 0
    for fn in filenames:
        im = cv2.imread(fn)
        # average the R, G, B channels and flatten array
        inputData[index,:] = (im.mean(axis=2)/255.).flat
        index += 1
    return inputData

def getImageSizes(filenames):
    """
    Get the number of x and y pixels
    @parameter filenames file names
    @return nx, ny
    """
    im = cv2.imread(filenames[0])
    return im.shape[:2]


trainingDir = args.trainDir

df = pandas.read_csv(trainingDir + '/train.csv')
categories = df['numberOfFeatures'].unique()
categories.sort()
minNumFeatures = min(categories)
maxNumFeatures = max(categories)
numCategories = maxNumFeatures - minNumFeatures + 1
# labels start at zero
trainingOutput = numpy.array(df['numberOfFeatures']) - minNumFeatures
filenames = glob.glob(trainingDir + '/img*.???')
trainingInput = loadImages(filenames)

testingDir = args.testDir

df = pandas.read_csv(testingDir + '/test.csv')
numCategories = len(categories)
# labels start at zero
testingOutput = numpy.array(df['numberOfFeatures']) - minNumFeatures
testingInput = loadImages(glob.glob(testingDir + '/img*.???'))

# train the model
n0, n1 = getImageSizes(filenames)
print('Number of training images: {}'.format(trainingInput.shape[0]))
print('Number of testing images : {}'.format(testingInput.shape[0]))
print('Image size               : {} x {}'.format(n0, n1))
print('Categories               : {} min/max = {}/{}'.format(categories, minNumFeatures, maxNumFeatures))

#clf = svm.SVC(kernel='rbf', gamma='scale', verbose=True, random_state=567)
clf = svm.NuSVC(kernel='rbf', gamma='scale', decision_function_shape='ovo', verbose=True, random_state=567)


# now train
clf.fit(trainingInput, trainingOutput)

# test
prediction = clf.predict(testingInput)
numFeatures = prediction + minNumFeatures
numFeatureExact = testingOutput + minNumFeatures

# compute score
diffs = (numFeatures - numFeaturesExact)**2
score = diffs.sum()
numFailures = (diffs != 0).sum()

numTestingImages = prediction.shape[0]
print('sum of errors squared = {} number of failures = {} ({} %)'.format(score, numFailures, 100*numFailures/numTestingImages))

nImages = 20
print('known number of features for the first {} testing images: {}'.format(nImages, numFeaturesExact[:nImages]))
print('inferred number features for the first {} testing images: {}'.format(nImages, numFeatures[:nImages]))

# plot training/test dataset
import matplotlib
if args.save:
    # does not require X
    matplotlib.use('Agg')
from matplotlib import pylab
n = 50
for i in range(n):
    pylab.subplot(n//10, 10, i + 1)
    pylab.imshow(testingInput[i,...].reshape(n0, n1))
    titleColor = 'black'
    if int(numFeaturesExact[i]) != numpy.round(numFeatures[i]):
        titleColor = 'red'
    pylab.title('{} ({})'.format(numFeaturesExact[i], numFeatures[i]),
        fontsize=8, color=titleColor)
    pylab.axis('off')
if not args.save:
    pylab.show()
else:
    pylab.savefig(args.save, dpi=300)
