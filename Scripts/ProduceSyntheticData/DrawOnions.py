import numpy
import random
from matplotlib import pylab
import os
import pandas
import argparse

parser = argparse.ArgumentParser(description='Create onion cross sections.')
parser.add_argument('--seed', type=int, default=13435, 
                    help='Random seed')
parser.add_argument('--numberOfImages', type=int, default=100, 
                    help='Number of images')
parser.add_argument('--minRange', type=int, default=1, 
                    help='Min number of circles')
parser.add_argument('--maxRange', type=int, default=5, 
                    help='Max number of circles')
parser.add_argument('--nt', type=int, default=128, 
                    help='Number of points used for each circle')
parser.add_argument('--outputDir', default='../../Data/Synthetic/Onions/',
                    help='Output directory')
parser.add_argument('--csvFile', default='train.csv',
                    help='Set CSV file name containing number of features for each image')
args = parser.parse_args()


numpy.random.seed(args.seed)

def rotate(xys, angle=0.0):
    n = xys.shape[1]
    xyg = xys.sum(axis=1)/float(n)
    dx = xys[0] - xyg[0]
    dy = xys[1] - xyg[1]
    cosa, sina = numpy.cos(alpha), numpy.sin(alpha)
    xsPrime =  cosa*dx + sina*dy
    ysPrime = -sina*dx + cosa*dy
    return xsPrime, ysPrime


class Onion(object):

    def __init__(self, nr, nrMax, nt=128):
        self.ts = numpy.linspace(0., 2*numpy.pi, nt + 1)
        self.elong = 1.0
        self.triang = 0.0
        self.nr = nr
        self.dr = 1.0/float(nrMax)

    def setProperties(self, **kw):
        self.elong = kw.get('elong', 1.0)
        self.triang = kw.get('triang', 0.0)

    def getXY(self, i):
        r = numpy.sqrt((i + 0.5) * self.dr)
        xys = numpy.zeros((2, self.ts.shape[0]), numpy.float64)
        xys[0,:] = 1.0 + r*numpy.cos(self.ts)
        xys[1,:] = r*self.elong*numpy.sin(self.ts + self.triang*numpy.sin(self.ts))
        return xys

    def saveImage(self, filename, alpha=0.0):
        for i in range(self.nr):
            xs, ys = rotate(self.getXY(i), alpha)
            pylab.plot(xs, ys, 'k-')
        pylab.axis([-1.5, 1.5, -1.5, 1.5])
        pylab.axis('off')
        pylab.savefig('{}'.format(filename), dpi=40)
        pylab.close()


try:
    os.makedirs('{}'.format(args.outputDir))
except:
    pass


imageId = numpy.zeros((args.numberOfImages,), numpy.int)
numberOfRings = numpy.zeros((args.numberOfImages,), numpy.int)
for i in range(args.numberOfImages):

    nr = int(args.minRange + (args.maxRange + 0.99 - args.minRange)*numpy.random.random())
    elong = 0.8 + numpy.random.random()
    triang = 0.0 + 0.3*numpy.random.random()
    alpha = 0.0 + 2*numpy.pi*numpy.random.random()

    on = Onion(nr=nr, nrMax=args.maxRange, nt=args.nt)
    on.setProperties(elong=elong, triang=triang)
    filename = '{}/img{}.png'.format(args.outputDir, i)
    print('saving file {}...'.format(filename))
    on.saveImage(filename)

    imageId[i] = i
    numberOfRings[i] = nr

df = pandas.DataFrame(list(zip(imageId, numberOfRings)), columns=['imageId', 'numberOfFeatures'])
csvFile = args.outputDir + '/' + args.csvFile
print('saving {}...'.format(csvFile))
df.to_csv(csvFile)
