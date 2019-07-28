import numpy
import random
from matplotlib import pylab
import os
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
args = parser.parse_args()


numpy.random.seed(args.seed)

def rotate(xys, angle=0.0):
    cosa, sina = numpy.cos(alpha), numpy.sin(alpha)
    xsPrime = cosa*xys[0] + sina*xys[1]
    ysPrime = -sina*xys[0] + cosa*xys[1]
    return xsPrime, ysPrime


class Onion(object):

    def __init__(self, nr, nt=128):
        self.ts = numpy.linspace(0., 2*numpy.pi, nt + 1)
        self.elong = 1.0
        self.triang = 0.0
        self.nr = nr
        self.dr = 1.0/float(nr)

    def setProperties(self, **kw):
        self.elong = kw.get('elong', 1.0)
        self.triang = kw.get('triang', 0.0)

    def getXY(self, i):
        r = numpy.sqrt((i + 0.5) * self.dr)
        xs = 1.0 + r*numpy.cos(self.ts)
        ys = r*self.elong*numpy.sin(self.ts + self.triang*numpy.sin(self.ts))
        return xs, ys

    def saveImage(self, filename, alpha=0.0):
        for i in range(self.nr):
            xs, ys = rotate(self.getXY(i), alpha)
            pylab.plot(xs, ys, 'k-')
            pylab.axis('equal')
            pylab.axis('off')
        pylab.savefig('{}'.format(filename), dpi=40)
        pylab.close()


for i in range(args.minRange, args.maxRange + 1):
    try:
        os.makedirs('{}/{}'.format(args.outputDir, i))
    except:
        pass

for i in range(args.numberOfImages):

    nr = args.minRange + int((args.maxRange - args.minRange)*numpy.random.random() + 0.5)
    elong = 0.8 + numpy.random.random()
    triang = 0.0 + 0.3*numpy.random.random()
    alpha = 0.0 + 2*numpy.pi*numpy.random.random()

    on = Onion(nr=nr, nt=args.nt)
    on.setProperties(elong=elong, triang=triang)
    filename = '{}{}/img{}.png'.format(args.outputDir, on.nr, i)
    print('saving file {}...'.format(filename))
    on.saveImage(filename)
