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
from trippy import psfStarChooser, scamp, psf
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')


def getCatalogue(file_start, **kwargs):
  """getCatalog checks whether a catalog file already exists.
  If it does, it is read in. If not, it runs SExtractor to create it.
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
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
  print("\n" + str(ncat) + " catalog stars\n" if verbose else "")
  return fullcatalog


def findBestImage(imageArray, **kwargs):
  """findBestImage finds the best image among a list.
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  zeros = np.zeros(len(imageArray))
  for ii, frameID in enumerate(imageArray):
    hans = pyf.open(frameID + '.fits')
    headers = hans[0].header
    zeros[ii] = headers['MAGZERO']
  bestZeroID = np.argmax(zeros)
  bestFullCatalogue = getCatalogue(imageArray[bestZeroID])
  print("Image" + str(bestZeroID) + " is best." if verbose else "")
  return bestZeroID, bestFullCatalogue
#bestImage, bestCatalogue = findBestImage(['a' + str(ii)
#                                          for ii in range(100, 123)])


def getAllCatalogues(imageArray, **kwargs):
  """Grab the SExtractor catalogue for all the images.
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  catalogueArray = []
  for ii, dummy in enumerate(imageArray):
    catalogueArray.append(getCatalogue(imageArray[ii]))
  print((len(catalogueArray), len(catalogueArray[0])) if verbose else "")
  return catalogueArray


def findSharedCatalogue(catalogueArray, useIndex, **kwargs):
  """Compare catalogues and create a master catalogue of only
  stars that are in all images
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  ntimes = len(catalogueArray)
  keys = catalogueArray[0].keys()
  nkeys = len(keys)
  xkey = np.where(np.array(keys) == 'XWIN_IMAGE')[0][0]
  ykey = np.where(np.array(keys) == 'YWIN_IMAGE')[0][0]
  xstar = [list(catalogueArray[ii]['XWIN_IMAGE']) for ii in range(ntimes)]
  ystar = [list(catalogueArray[ii]['YWIN_IMAGE']) for ii in range(ntimes)]
  nobjmax = len(xstar[useIndex])
  master = np.zeros([nobjmax, nkeys + ntimes])
  master[:, 0:nkeys] = np.array([list(catalogueArray[useIndex][key])
                                 for key in keys]).T
  for tt in np.arange(0, ntimes):
    for ss in np.arange(len(xstar[tt])):
      distsquare = ((master[:, xkey] - xstar[tt][ss]) ** 2 +
                    (master[:, ykey] - ystar[tt][ss]) ** 2)
      idx = distsquare.argmin()
      if distsquare[idx] < 25:  # if match previous star, add to its array
        master[idx, nkeys + tt] = 1
      else:  # else do nothing
        pass
  trimlist = []
  for ss in np.arange(nobjmax):
    if len(np.where(master[ss][nkeys:] == 0)[0]) == 0:
      trimlist.append(ss)
  sharedCatalogue = {}
  for kk, key in enumerate(keys):
    sharedCatalogue[key] = master[trimlist][:, kk]
  print((len(sharedCatalogue), len(sharedCatalogue[0])) if verbose else "")
  """This should return a catalogue dictionary formatted in the same
  way as the original catalogue, allowing us to do anything with it that
  we could do with any other catalogue.
  """
  return sharedCatalogue


def inspectStars(file_start, catalogue, repfactor, **kwargs):
  """Run psfStarChooser, inspect stars, generate PSF and lookup table.
  """
  SExCatalogue = kwargs.pop('SExCatalogue', False)
  noVisualSelection = kwargs.pop('noVisualSelection', True)
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  try:
    with pyf.open(file_start + '.fits') as han:
      data = han[0].data
    starChooser = psfStarChooser.starChooser(data, catalogue['XWIN_IMAGE'],
                                             catalogue['YWIN_IMAGE'],
                                             catalogue['FLUX_AUTO'],
                                             catalogue['FLUXERR_AUTO'])
    (goodFits, goodMeds, goodSTDs
     ) = starChooser(30, 100,  # (box size, min SNR)
                     initAlpha=3., initBeta=3.,
                     repFact=repfactor,
                     includeCheesySaturationCut=False,
                     noVisualSelection=noVisualSelection,
                     verbose=False)
    print(("\ngoodFits = ", goodFits, "\n") if verbose else "")
    print(("\ngoodMeds = ", goodMeds, "\n") if verbose else "")
    print(("\ngoodSTDs = ", goodSTDs, "\n") if verbose else "")
    goodPSF = psf.modelPSF(np.arange(61), np.arange(61), alpha=goodMeds[2],
                           beta=goodMeds[3], repFact=repfactor)
    fwhm = goodPSF.FWHM()  # this is the pure moffat FWHM
    print("fwhm = " + str(fwhm) if verbose else "")
    goodPSF.genLookupTable(data, goodFits[:, 4], goodFits[:, 5], verbose=False)
    goodPSF.genPSF()
    fwhm = goodPSF.FWHM()  # this is the FWHM with lookuptable included
    print("fwhm = " + str(fwhm) if verbose else "")
  except UnboundLocalError:
    print("Data error occurred!")
    raise
  if SExCatalogue:
    bestCatalogue = extractGoodStarCatalogue(catalogue,
                                             goodFits[:, 4], goodFits[:, 5])
    return bestCatalogue
  else:
    return goodFits, goodMeds, goodSTDs, goodPSF, fwhm


def extractGoodStarCatalogue(startCatalogue, xcat, ycat, **kwargs):
  """extractBestStarCatalogue crops a Catalogue,
  leaving only the stars that match a given x & y list.
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  trimCatalogue = {'XWIN_IMAGE': xcat,
                   'YWIN_IMAGE': ycat}
  print(trimCatalogue if verbose else "")
  goodCatalogue = findSharedCatalogue([startCatalogue, trimCatalogue], 0)
  print((len(goodCatalogue), len(goodCatalogue[0])) if verbose else "")
  return goodCatalogue


def getArguments(sysargv, **kwargs):
  """Get arguments given when this is called from a command line"""
  filenameFile = 'files.txt'  # Change with '-f <filename>' flag
  repfactor = 10
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  try:
    options, dummy = getopt.getopt(sysargv[1:], "f:v:h:",
                                   ["ifile=i", "verbose"])
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
    imageArray = getFileNames(filenameFile)
  return imageArray, repfactor, verbose


def getFileNames(filenameFile, **kwargs):
  """Reads a file that has a list of filenames, one per line."""
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  imageArray = np.genfromtxt(filenameFile, usecols=(0), dtype=str)
  print((filenameFile, imageArray) if verbose else "")
  return imageArray


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


#
