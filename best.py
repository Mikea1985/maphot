#!/usr/bin/python
"""maphot is a wrapper for easily running trippy on a bunch of images
of a single object on a single night.
Usage: (-h gives this line as well)
maphot.py -c <coofile> -f <fitsfile> -v False -. True -o False -r True -a 07
Defaults are:
inputFile = 'a100.fits'  # Change with '-f <filename>' flag
coofile = 'coords.in'  # Change with '-c <coofile>' flag
verbose = False  # Change with '-v True' or '--verbose True'
centroid = True  # Change with '-. False' or  --centroid False'
overrideSEx = False  # Change with '-o True' or '--override True'
remove = True  # Change with '-r False' or '--remove False'
aprad = 0.7  # Change with '-a 1.5' or '--aprad 1.5'
coordsfile is a file that contains:
x1 y1 MJD1
x2 y2 MJD2
ie. the position of the TNO at two times, and those times in MJD format.
The script then reads the MJD keyword from the input files and extrapolates
in order to predict the location in that image.
"""

from __future__ import print_function, division
import numpy as np
import astropy.io.fits as pyf
import maphot.maphot
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')


def findBestImage(imageArray):
  """findBestImage finds the best image among a list.
  Not yet tested.
  """
  zeros = np.zeros(len(imageArray))
  for frameID in imageArray:
    hans = pyf.open(frameID + '.fits')
    headers = hans[0].header
    zeros = headers['MAGZERO']
  bestzeroid = np.argmax(zeros)
  bestfullcat = maphot.getCatalogue(imageArray[bestzeroid])
  return imageArray[bestzeroid], bestfullcat

#bestImage, bestCatalogue = findBestImage(['a' + str(ii)
#                                          for ii in range(100, 123)])

#
