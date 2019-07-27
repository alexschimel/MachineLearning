import numpy
import random
from matplotlib import pylab
import argparse



class Onion(object):

    def __init__(self, nr, nt):
        self.ts = numpy.linspace(0., 2*numpy.pi, nt + 1)
        self.elong = 1.0
        self.triang = 0.0
        self.nr = nr
        self.dr = 1.0/float(nr)
        self.x0 = 1.0

    def setProperties(self, **kw):
        self.elong = kw.get('elong', 1.0)
        self.triang = kw.get('triang', 0.0)
        self.x0 = kw.get('x0', 0.0)

    def getXY(self, i):
        r = i * self.dr
        xs = self.x0 + r*numpy.cos(self.ts)
        ys = r*self.elong*numpy.sin(self.ts - self.triang*numpy.sin(self.ts))
        return xs, ys

    def saveImage(self, filename):
        pylab.figure()
        for i in range(self.nr):
            xs, ys = self.getXY(i)
            pylab.plot(xs, ys, 'k-')
            pylab.axis('equal')
            pylab.axis('off')
        pylab.savefig(filename)


if __name__ == '__main__':
    on = Onion(10, 32)
    on.setProperties(elong=1.6, triang=0.3, x0=2.0)
    on.saveImage('toto.png')
