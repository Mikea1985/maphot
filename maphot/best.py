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
from trippy import psfStarChooser, scamp
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')


def getCatalogue(file_start):
  """getCatalog checks whether a catalog file already exists.
  If it does, it is read in. If not, it runs SExtractor to create it.
  """
  outfile = open(file_start + ".trippy", 'w')
  try:
    fullcatalog = scamp.getCatalog(file_start + '_fits.cat',
                                   paramFile='def.param')
  except IOError:
    try:
      scamp.makeParFiles.writeSex(file_start + '_fits.sex', minArea=3.0,
                                  threshold=5.0, zpt=26.0, aperture=20.,
                                  min_radius=2.0, catalogType='FITS_LDAC',
                                  saturate=55000)
      scamp.makeParFiles.writeConv()
      scamp.makeParFiles.writeParam(numAps=1)
      scamp.makeParFiles.writeSex(file_start + '_ascii.sex', minArea=3.0,
                                  threshold=5.0, zpt=26.0, aperture=20.,
                                  min_radius=2.0, catalogType='ASCII',
                                  saturate=55000)
      scamp.makeParFiles.writeConv()
      scamp.makeParFiles.writeParam(numAps=1)
      scamp.runSex(file_start + '_fits.sex', file_start + '.fits',
                   options={'CATALOG_NAME': file_start + '_fits.cat'})
      scamp.runSex(file_start + '_ascii.sex', file_start + '.fits',
                   options={'CATALOG_NAME': file_start + '_ascii.cat'})
      fullcatalog = scamp.getCatalog(file_start + '_fits.cat',
                                     paramFile='def.param')
    except IOError as error:
      print("IOError: ", error)
      print("You have almost certainly forgotten to activate Ureka!")
      raise
  except UnboundLocalError:
    print("\nData error occurred!\n")
    raise
  ncat = len(fullcatalog['XWIN_IMAGE'])
  print("\n", ncat, " catalog stars\n")
  outfile.write("\n{} catalog stars\n".format(ncat))
  return fullcatalog


def findBestImage(imageArray):
  """findBestImage finds the best image among a list.
  Not yet tested.
  """
  zeros = np.zeros(len(imageArray))
  for ii, frameID in enumerate(imageArray):
    hans = pyf.open(frameID + '.fits')
    headers = hans[0].header
    zeros[ii] = headers['MAGZERO']
  bestzeroid = np.argmax(zeros)
  bestfullcat = getCatalogue(imageArray[bestzeroid])
  return imageArray[bestzeroid], bestfullcat
#bestImage, bestCatalogue = findBestImage(['a' + str(ii)
#                                          for ii in range(100, 123)])


def getAllCatalogues(imageArray):
  """Grab the SExtractor catalogue for all the images.
  """
  catalogueArray = []
  for ii, frameID in enumerate(imageArray):
    catalogueArray.append(getCatalogue(imageArray[ii]))
  return catalogueArray


def findSharedCatalogue(catalogueArray):
  """Compare catalogues and create a master catalogue of only 
  stars that are in all images
  """
  ntimes = len(catalogueArray)
  xstar = [list(catalogueArray[ii]['XWIN_IMAGE']) for ii in range(ntimes)]
  ystar = [list(catalogueArray[ii]['YWIN_IMAGE']) for ii in range(ntimes)]
  magstar = [list(catalogueArray[ii]['MAG_AUTO']) for ii in range(ntimes)]
  nobjmax = np.sum([len(xstar[tt]) for tt in np.arange(ntimes)])
  master = np.zeros([nobjmax, 2 + ntimes])
  n0 = len(xstar[0])
  master[0:n0, 0:3] = np.array([xstar[0], ystar[0], magstar[0]]).T
  nobjec = n0
  for tt in np.arange(1, ntimes):
    for ss in np.arange(len(xstar[tt])):
      distsquare = ((master[:nobjec, 0] - xstar[tt][ss]) ** 2 +
                    (master[:nobjec, 1] - ystar[tt][ss]) ** 2)
      idx = distsquare.argmin()
      if distsquare[idx] < 25:  # if match previous star, add mag to its array
        master[idx, 2 + tt] = magstar[tt][ss]
      else:  # else add a new star entry
        master[nobjec, 2 + tt] = magstar[tt][ss]
        master[nobjec, 0:2] = xstar[tt][ss], ystar[tt][ss]
        nobjec += 1
  trimlist = []
  for ss in np.arange(nobjec):
    if len(np.where(master[ss] == 0)[0]) == 0:
      trimlist.append(ss)
  sharedCatalogue = master[trimlist][:, 0:2]
  """Only the X and Y of the shared catalogue is returned, even though
  other information, like magnitude, from SExtractor could easily be added. 
  This is because we don't need the magnitudes, we'll do better with TrIPPy.
  """
  return sharedCatalogue




#
