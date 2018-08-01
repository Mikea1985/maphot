#!/usr/bin/python
"""
Module for:
 finding the best image,
 finding catalogue of stars visible in all,
 finding the best images from that catalogue.
"""

from __future__ import print_function, division
import getopt
import sys
import pickle
import numpy as np
import astropy.io.fits as pyf
from maphot_functions import (getSExCatalog, inspectStars, findSharedCatalogue)
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')


def findBestImage(imageArray,
                  SEx_params=np.array([2.0, 2.0, 27.8, 10.0, 2.0, 2.0]),
                  **kwargs):
  """findBestImage finds the best image among a list.
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  zeros = np.zeros(len(imageArray))
  fluxlim = np.zeros(len(imageArray))
  for ii, frameID in enumerate(imageArray):
    hans = pyf.open(frameID + '.fits')
    headers = hans[0].header
    try:
      zeros[ii] = headers['MAGZERO']  # Subaru Hyper-Suprime
    except KeyError:
      try:
        zeros[ii] = headers['PHOT_C']   # CFHT MegaCam
      except KeyError:
        zeros[ii] = -666
        print("Frame " + str(frameID) + " did not have MAGZERO keyword."
              if verbose else "")
    try:
      fluxlim[ii] = headers['FLUXLIM']  # Subaru Hyper-Suprime
    except KeyError:
      fluxlim[ii] = headers['MAG_LIM']  # CFHT MegaCam
  if np.max(zeros) > -666:
    bestZeroID = np.argmax(zeros)
  else:
    print("No frames had MAGZERO keyword." if verbose else "")
    bestZeroID = np.argmax(fluxlim)
  bestFullCatalogue = getSExCatalog(imageArray[bestZeroID], SEx_params)
  print("Image" + str(bestZeroID) + " is best.")
  return bestZeroID, bestFullCatalogue


def getAllCatalogues(imageArray,
                     SEx_params=np.array([2.0, 2.0, 27.8, 10.0, 2.0, 2.0]),
                     **kwargs):
  """Grab the SExtractor catalogue for all the images.
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  catalogueArray = []
  for ii, dummy in enumerate(imageArray):
    catalogueArray.append(getSExCatalog(imageArray[ii], SEx_params))
  print((len(catalogueArray), len(catalogueArray[0])) if verbose else "")
  return catalogueArray


def getArguments(sysargv):
  """Get arguments given when this is called from a command line"""
  filenameFile = 'files.txt'  # Change with '-f <filename>' flag
  repfactor = 10
  verbose = False
  try:
    options, dummy = getopt.getopt(sysargv[1:], "f:v:h:r",
                                   ["ifile=", "verbose=", "repfactor="])
  except TypeError as error:
    print(error)
    sys.exit()
  except getopt.GetoptError as error:
    print(" Input ERROR! ")
    print('best -f <filename>')
    sys.exit(2)
  else:
    for opt, arg in options:
      if opt in ("-v", "-verbose"):
        if arg == '0' or arg == 'False':
          arg = False
        elif arg == '1' or arg == 'True':
          arg = True
        else:
          print(opt, arg, np.array([arg]).dtype)
          raise TypeError("-v flags must be followed by " +
                          "0/False/1/True")
      if opt == '-h':
        print('best -f <filename>')
      elif opt in ('-f', '--ifile'):
        filenameFile = arg
      elif opt in ('-v', '--verbose'):
        verbose = arg
      elif opt in ('-r', '--repfactor'):
        repfactor = arg
    imageArray = np.genfromtxt(filenameFile, usecols=(0), dtype=str)
  return imageArray, repfactor, verbose


def pickleCatalogue(catalogue, filename, **kwargs):
  """Write a catalogue to an ascii file."""
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  outfile = open(filename, 'wb')
  pickle.dump(catalogue, outfile, pickle.HIGHEST_PROTOCOL)
  print("Catalogue pickled." if verbose else "")
  outfile.close()


def unpickleCatalogue(filename, **kwargs):
  """Write a catalogue to an ascii file."""
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  outfile = open(filename, 'rb')
  catalogue = pickle.load(outfile)
  outfile.close()
  print("Catalogue unpickled." if verbose else "")
  return catalogue


def best(imageArray, repfactor, **kwargs):
  """This function can be run to do all of the above.
  This is called automatically if this is main."""
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  print(imageArray if verbose else "")
  bestID, bestCatalogue = findBestImage(imageArray)
  print(bestID if verbose else "")
  catalogueArray = getAllCatalogues(imageArray)
  sharedCatalogue = findSharedCatalogue(catalogueArray, bestID)
  bestCatalogue = inspectStars(imageArray[bestID], sharedCatalogue, repfactor,
                               SExCatalogue=True, noVisualSelection=False)
  pickleCatalogue(bestCatalogue, 'best.cat')
  return bestID, bestCatalogue

if __name__ == '__main__':
  images, repfact, verbatim = getArguments(sys.argv)
  bestImage, bestCat = best(images, repfact, verbose=verbatim)
  print('Best catalogue:')
  print(bestCat)
  print('Best image #: ' + str(bestImage))


# End of file.
# Nothing to see here.
