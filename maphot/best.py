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
from datetime import datetime
import warnings
import pickle
import numpy as np
import astropy.io.fits as pyf
from astropy.table.table import Table
from maphot_functions import (getSExCatalog, inspectStars,
                              queryPanSTARRS, readPanSTARRS, PS1_vs_SEx,
                              getDataHeader, findSharedPS1Catalogue,
                              saveStarMag, trimCatalog)
from __version__ import __version__
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')


def findBestImage(imageArray,
                  SEx_params=np.array([2.0, 2.0, 27.8, 10.0, 2.0, 2.0]),
                  **kwargs):
  """findBestImage finds the best image among a list.
  """
  verbose = kwargs.pop('verbose', False)
  extno = kwargs.pop('extno', None)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  zeros = np.zeros(len(imageArray))
  fluxlim = np.zeros(len(imageArray))
  for ii, frameID in enumerate(imageArray):
    hans = pyf.open(frameID + '.fits')
    if extno is None:
      print('Warning: Treating this as a single extension file.')
      headers = hans.header
    else:
      headers = hans[extno].header
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
  bestFullCatalogue = getSExCatalog(imageArray[bestZeroID] + '.fits',
                                    SEx_params, extno=extno)
  print("Image" + str(bestZeroID) + " is best.")
  return bestZeroID, bestFullCatalogue


def getAllCatalogues(imageArray,
                     SEx_params=np.array([2.0, 2.0, 27.8, 10.0, 2.0, 2.0]),
                     **kwargs):
  """Grab the SExtractor catalogue for all the images.
  """
  verbose = kwargs.pop('verbose', False)
  extno = kwargs.pop('extno', None)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  catalogueArray = []
  for ii, dummy in enumerate(imageArray):
    catalogueArray.append(getSExCatalog(imageArray[ii] + '.fits', SEx_params,
                                        extno=extno))
  print((len(catalogueArray), len(catalogueArray[0])) if verbose else "")
  return catalogueArray


def getArguments(sysargv):
  """Get arguments given when this is called from a command line"""
  filenameFile = 'files.txt'  # Change with '-f <filename>' flag
  repfactor = 10
  verbose = False
  ignoreWarns = False
  extno = None
  try:
    options, dummy = getopt.getopt(sysargv[1:], "f:v:h:r:e:i:",
                                   ["filenamefile=", "verbose=", "repfactor=",
                                    "extension=", "ignoreWarnings="])
  except TypeError as error:
    print(error)
    sys.exit()
  except getopt.GetoptError as error:
    print(" Input ERROR! ")
    print('best -f <filename>')
    sys.exit(2)
  else:
    print(options)
    for opt, arg in options:
      if opt in ("-v", "--verbose", "-i", "--ignoreWarnings"):
        if arg == '0' or arg == 'False':
          arg = False
        elif arg == '1' or arg == 'True':
          arg = True
        else:
          print(opt, arg, np.array([arg]).dtype)
          raise TypeError("-v and -i flags must be followed by " +
                          "0/False/1/True")
      if opt == '-h':
        print('best -f <filenamefile> -e <extension> -i <ignoreWarnings>')
      elif opt in ('-f', '--filenamefile'):
        filenameFile = arg
      elif opt in ('-v', '--verbose'):
        verbose = arg
      elif opt in ('-r', '--repfactor'):
        repfactor = arg
      elif opt in ('-e', '--extension'):
        extno = int(arg)
      elif opt in ('-i', '--ignoreWarnings'):
        ignoreWarns = arg
    imageArray = np.array([ia.replace('.fits', '')
                           for ia in np.genfromtxt(filenameFile,
                                                   usecols=(0), dtype=str)])
  print(imageArray)
  return imageArray, repfactor, verbose, extno, ignoreWarns


def pickleCatalogue(catalogue, filename, **kwargs):
  """Write a catalogue to an ascii file."""
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  outfile = open(filename, 'wb')
  pickle.dump(Table(catalogue, masked=False), outfile, pickle.HIGHEST_PROTOCOL)
  outfile.close()
  print("Catalogue pickled." if verbose else "")
  return


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


def PanSTARRSStuff(SExCatalogArray, bestID):
  """Load the PS1 catalogue, identify PS1 stars in the SExtractor catalog.
  """
  #Get the PS1 catalog for the area around the TNO.
  #Only do this if the catalog hasn't already been downloaded once.
  obsRA = np.nanmedian(SExCatalogArray[bestID]['X_WORLD'])
  obsDec = np.nanmedian(SExCatalogArray[bestID]['Y_WORLD'])
  RADecString = '{0:05.1f}_{1:+4.1f}'.format(obsRA, obsDec)
  try:
    PS1Catalog = readPanSTARRS('panstarrs_' + RADecString + '.xml',
                               PSF_Kron=0.4)
  except IOError:
    queryPanSTARRS(obsRA, obsDec, rad_deg=0.3,
                   catalog_filename='panstarrs_' + RADecString + '.xml')
    PS1Catalog = readPanSTARRS('panstarrs_' + RADecString + '.xml',
                               PSF_Kron=0.4)
  print('A Pan-STARRS catalog has been loaded with '
        + '{} entries.'.format(len(PS1Catalog)))
  #Match the PS1 sources to the SExtractor catalog. Only keep matched pairs.
  PS1CatArray = []
  for SExCatalog in SExCatalogArray:
    PS1CatArray.append(PS1_vs_SEx(PS1Catalog, SExCatalog, maxDist=2.5,
                                  appendSEx=False))
  PS1SharedCat = findSharedPS1Catalogue(PS1CatArray)
  return PS1SharedCat


def best(imageArray, repfactor, **kwargs):
  """This function can be run to do all of the above.
  This is called automatically if this is main."""
  extno = kwargs.pop('extno', None)
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  print(__version__ if verbose else "")
  print(imageArray if verbose else "")
  #bestID, bestSExCat = findBestImage(imageArray, extno=extno)
  #print(bestID if verbose else "")
  catalogueArray = getAllCatalogues(imageArray, extno=extno)
  nCatMembers = [len(cat['XWIN_IMAGE']) for cat in catalogueArray]
  bestID = np.argmax(nCatMembers)
  bestSExCat = catalogueArray[bestID]
  (bestData, _, _, _, _, MJDm, _, NAXIS1, NAXIS2, _, _,INS
   ) = getDataHeader(imageArray[bestID] + '.fits', extno=extno)
  print("The best image is {}".format(imageArray[bestID]))
  bestSExCatTrimmed = trimCatalog(bestSExCat, bestData, dcut=30, mcut=55000,
                                  snrcut=0, shapecut=5,  # basically no cuts
                                  naxis1=NAXIS1, naxis2=NAXIS2)
  catalogueArray[bestID] = bestSExCatTrimmed  # lazy workaround
  PS1SharedCat = PanSTARRSStuff(catalogueArray, bestID)
  print('{}'.format(len(PS1SharedCat)) +
        ' PS1 sources are visible in all images.')
  timeNow = datetime.now().strftime('%Y-%m-%d/%H:%M:%S')
  if verbose:
    saveStarMag('PS1SharedCat.txt', PS1SharedCat,
                timeNow, __version__, 'All images', extno=extno)
  bestSharedPS1SExCat = PS1_vs_SEx(PS1SharedCat, bestSExCat,
                                   maxDist=2.5, appendSEx=True)
  if verbose:
    saveStarMag('bestSharedPS1SExCat.txt', bestSharedPS1SExCat,
                timeNow, __version__, MJDm, extno=extno)
  inspectedSExCat = inspectStars(bestData, bestSharedPS1SExCat[:],
                                 repfactor, SExCatalogue=True,
                                 noVisualSelection=False)
  inspectedPS1Cat = findSharedPS1Catalogue([PS1SharedCat, inspectedSExCat])
  saveStarMag('InspectedStars.txt', inspectedPS1Cat,
              timeNow, __version__, 'All images', extno=extno)
  print('{}'.format(len(inspectedPS1Cat)) +
        ' PS1 sources left after manual inspection.')
  bestCatName = ('best.cat' if extno is None
                 else 'best{0:02.0f}.cat'.format(extno))
  pickleCatalogue(inspectedPS1Cat, bestCatName)
  return bestID, inspectedPS1Cat


if __name__ == '__main__':
  images, repfact, verbatim, extension, ignoreWarnings = getArguments(sys.argv)
  #Ignore all Python warnings.
  #This is generally a terrible idea, and should be turned off for de-bugging.
  if ignoreWarnings:
    warnings.filterwarnings("ignore")
  bestImage, bestCat = best(images, repfact, extno=extension, verbose=verbatim)
  print('Best catalogue:')
  print(bestCat)
  print('Best image #: ' + str(bestImage))


# End of file.
# Nothing to see here.
