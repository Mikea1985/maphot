#!/usr/bin/python
"""
Module for:
 finding the best image,
 finding catalogue of stars visible in all,
 finding the best images from that catalogue.
"""

from __future__ import print_function, division
import getopt
import io
import os
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

## To adjust how strick trimming happens adjust these:
# PSF_Kron in the readPanSTARRS call:
#  limit difference between PSF and Kron mags (Galaxies have larger difference)
# maxDist in PS1_vs_SEx call:
#  limit to distance (") between PS1 and SEx coords
# dcut, mcut, snrcut and shapecut in trimCatalog call:
#  dcut; lower limit on distance to nearest neighbour.
#  mcut: maximum pixel valued allowed (should be below saturation value)
#  snrcut: lower limit for SNR of stars
#  shapecut: uper limit on A/B (long axis/short axis)
# rad_deg in queryPanSTARRS call:
#  radius of circle to get PS1 catalog for.

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
                                    SEx_params, extno=extno, verb=verbose)
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
                                        extno=extno, verb=verbose))
  print((len(catalogueArray), len(catalogueArray[0])) if verbose else "")
  return catalogueArray


def getArguments(sysargv):
  """Get arguments given when this is called from a command line"""
  filenameFile = 'files.txt'  # Change with '-f <filename>' flag
  repfactor = 10
  verbose = False
  ignoreWarns = False
  extno = None
  PSF_Kron = None
  autoTrim = False
  snrcut = 0.
  shapecut = 5.
  use1 = ('python best.py -f <filenamefile> -e <extension> ' +
          '-i <ignoreWarnings> -a <autoTrim> -k <PSF-Kron limit>')
  use2 = ('Defaults:\n' +
          '-f / --filenamefile=   files.txt\n' +
          '-v / --verbose=        False\n' +
          '-r / --repfactor=      10\n' +
          '-e / --extension=      None (must be integer otherwise)\n' +
          '-i / --ignoreWarnings= False\n' +
          '-k / --PSF_Kron=       None (otherwise float)\n' +
          '-s / --snrcut=         0. \n' +
          '-p / --shapecut=       5.\n' +
          '-a / --autoTrim=       False')
  try:
    options, dummy = getopt.getopt(sysargv[1:], "f:v:h:r:e:i:k:a:s:p:",
                                   ["filenamefile=", "verbose=", "repfactor=",
                                    "extension=", "ignoreWarnings=",
                                    "PSF_Kron=", "autoTrim=",
                                    "snrcut=", "shapecut="])
  except TypeError as error:
    print(error)
    sys.exit()
  except getopt.GetoptError as error:
    print(" Input ERROR! ")
    print(use1)
    print(use2)
    sys.exit(2)
  else:
    print(options)
    for opt, arg in options:
      if opt in ("-v", "--verbose", "-i", "--ignoreWarnings",
                 "-a", "--autoTrim"):
        if arg == '0' or arg == 'False':
          arg = False
        elif arg == '1' or arg == 'True':
          arg = True
        else:
          print(opt, arg, np.array([arg]).dtype)
          raise TypeError("-v, -i and -a flags must be followed by " +
                          "0/False/1/True")
      if opt == '-h':
        print(use1)
        print(use2)
        sys.exit(2)
      elif opt in ('-f', '--filenamefile'):
        filenameFile = arg
      elif opt in ('-r', '--repfactor'):
        repfactor = arg
      elif opt in ('-v', '--verbose'):
        verbose = arg
      elif opt in ('-e', '--extension'):
        extno = int(arg)
      elif opt in ('-i', '--ignoreWarnings'):
        ignoreWarns = arg
      elif opt in ('-k', '--PSF_Kron'):
        PSF_Kron = float(arg)
      elif opt in ('-a', '--autoTrim'):
        autoTrim = arg
      elif opt in ('-s', '--snrcut'):
        snrcut = float(arg)
      elif opt in ('-p', '--shapecut'):
        shapecut = float(arg)
    imageArray = np.array([ia.replace('.fits', '')
                           for ia in np.genfromtxt(filenameFile,
                                                   usecols=(0), dtype=str)])
  print(imageArray)
  return (imageArray, repfactor, verbose, extno, ignoreWarns, PSF_Kron,
          autoTrim, snrcut, shapecut)


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


def PanSTARRSStuff(SExCatalogArray, obsRA, obsDec, verbose=False,
                   radius=0.3, PSF_Kron=0.6):
  """Load the PS1 catalogue, identify PS1 stars in the SExtractor catalog.
  """
  #Get the PS1 catalog for the area around the TNO.
  #Only do this if the catalog hasn't already been downloaded once.
  RADecString = '{0:05.1f}_{1:+4.1f}'.format(obsRA, obsDec)
  try:
    PS1Catalog = readPanSTARRS('panstarrs_' + RADecString + '.xml',
                               PSF_Kron=PSF_Kron, verbose=verbose)
  except IOError:
    queryPanSTARRS(obsRA, obsDec, rad_deg=radius,
                   catalog_filename='panstarrs_' + RADecString + '.xml')
    PS1Catalog = readPanSTARRS('panstarrs_' + RADecString + '.xml',
                               PSF_Kron=PSF_Kron, verbose=verbose)
  print('A Pan-STARRS catalog has been loaded with '
        + '{} entries.'.format(len(PS1Catalog)))
  #Match the PS1 sources to the SExtractor catalog. Only keep matched pairs.
  PS1CatArray = []
  for SExCatalog in SExCatalogArray:
    PS1CatArray.append(PS1_vs_SEx(PS1Catalog, SExCatalog, maxDist=2.5,
                                  appendSEx=False))
  if verbose:
    with io.open('PS1CatKronTrim.cat', 'w') as sf:
      sf.write(u'RA(PS1)\tDec(PS1)\tPS1_name\n')
      [sf.write(u'{}\t{}\t{}\n'.format(PS1Catalog['raMean'][j],
                                       PS1Catalog['decMean'][j],
                                       PS1Catalog['objName'][j]))
       for j in np.arange(len(PS1Catalog['objName']))]
    for i, PC in enumerate(PS1CatArray):
      sf = io.open('PS1Cat{}.cat'.format(i), 'w')
      sf.write(u'RA(PS1)\tDec(PS1)\tPS1_name\n')
      [sf.write(u'{}\t{}\t{}\n'.format(PC['raMean'][j], PC['decMean'][j],
                                       PC['objName'][j]))
       for j in np.arange(len(PC['objName']))]
      sf.close()
  PS1SharedCat = findSharedPS1Catalogue(PS1CatArray)
  return PS1SharedCat


def calcRadius(WCS, NAXIS1, NAXIS2):
  '''Calculate the radius of the field (half maximum corner seperation).'''
  rac, decc = WCS.all_pix2world([0, 0, NAXIS1, NAXIS1],
                                [0, NAXIS2, NAXIS2, 0], 0)
  centRA = np.mean(rac)
  centDec = np.mean(decc)
  d = []
  for i in np.arange(4):
    for j in np.arange(4):
      if i != j:
        d.append(((rac[i] - rac[j]) ** 2 + (decc[i] - decc[j]) ** 2) ** 0.5)
  dmax = np.max(d) * 0.5
  print('Field radius = {} degrees.'.format(dmax))
  return dmax, centRA, centDec


def best(imageArray, repfactor, **kwargs):
  """This function can be run to do all of the above.
  This is called automatically if this is main."""
  extno = kwargs.pop('extno', None)  # Extension number for multi-ext fits
  PSF_Kron = kwargs.pop('PSF_Kron', None)  # PSF-Kron mag diff limit
  verbose = kwargs.pop('verbose', False)  # print/save more things?
  autoTrim = kwargs.pop('autoTrim', False)  # use psfStarChooser's autoTrim?
  snrcut = kwargs.pop('snrcut', 0)  # minimum snr to use star
  shapecut = kwargs.pop('shapecut', 5)  # maximum axis ratio
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  # If verbose, print some stuff
  print("{}\n".format(__version__) if verbose else "", end="")
  print("{}\n".format(imageArray) if verbose else "", end="")
  # Get all the SExtractor catalogues
  catalogueArray = getAllCatalogues(imageArray, extno=extno, verbose=verbose)
  nCatMembers = [len(cat['XWIN_IMAGE']) for cat in catalogueArray]
  bestID = np.argmax(nCatMembers)  # Best image has most sources, obviously...
  bestSExCat = catalogueArray[bestID]
  print("The best image is {}".format(imageArray[bestID]) +
        " with {} SExtractor detections.".format(len(bestSExCat['MAG_AUTO'])))
  mcut = 55000  # Near-saturation level for most imagers
  (bestData, _, _, _, _, MJDm, _, NAXIS1, NAXIS2, WCS, _, INS
   ) = getDataHeader(imageArray[bestID] + '.fits', extno=extno,
                     verbose=verbose)
  if INS == 'GMOS-N':
    mcut = 135000  # Near-saturation level for GMOS
  # Trim the SExtractor catalog. This automatically removes things near edge,
  # on non-central chip (GMOS only), saturated stars, as well as very
  # elongated sources (shapecut), faint sources (snrcut) and sources near
  # other sources (dcut). By default, dcut, snrcut and shapecut have very lax
  # constraints, so as to not accidentally remove too much.
  bestSExCatTrimmed = trimCatalog(bestSExCat, bestData, dcut=0, mcut=mcut,
                                  snrcut=snrcut, shapecut=shapecut,
                                  naxis1=NAXIS1, naxis2=NAXIS2,
                                  instrument=INS, verbose=verbose)
  print("{}".format(len(bestSExCatTrimmed['MAG_AUTO'])) +
        " SExtractor sources left after trimming garbage.")
  if verbose:  # Save the trimmed SExtractor catalog for best image
    with io.open('SExCatBestTrim.cat', 'w') as sf:
      sf.write(u'RA(SEx)\tDec(SEx)\tX(SEx)\tY(SEx)\n')
      [sf.write(u'{}\t{}\t{}\t{}\n'.format(bestSExCatTrimmed['X_WORLD'][j],
                                           bestSExCatTrimmed['Y_WORLD'][j],
                                           bestSExCatTrimmed['XWIN_IMAGE'][j],
                                           bestSExCatTrimmed['YWIN_IMAGE'][j]))
       for j in np.arange(len(bestSExCatTrimmed['X_WORLD']))]
  # Lazy workaround: replace full SEx cat of best image with trimmed cat:
  catalogueArray[bestID] = bestSExCatTrimmed
  # Calculate size of field:
  dmax, centRA, centDec = calcRadius(WCS, NAXIS1, NAXIS2)
  # Now download PS1 catalogue and identify stars in the SEx cat:
  PS1SharedCat = PanSTARRSStuff(catalogueArray, centRA, centDec,
                                verbose=verbose, radius=dmax * 1.1,
                                PSF_Kron=PSF_Kron)
  print('{}'.format(len(PS1SharedCat)) +
        ' PS1 sources are visible in all images.')
  timeNow = datetime.now().strftime('%Y-%m-%d/%H:%M:%S')
  if verbose:
    saveStarMag('PS1SharedCat.txt', PS1SharedCat, timeNow, __version__,
                'All images', None, extno=extno)
  bestSharedPS1SExCat = PS1_vs_SEx(PS1SharedCat, bestSExCat,
                                   maxDist=2.5, appendSEx=True)
  if verbose:
    saveStarMag('bestSharedPS1SExCat.txt', bestSharedPS1SExCat,
                timeNow, __version__, MJDm, None, extno=extno)
  inspectedSExCat = inspectStars(bestData, bestSharedPS1SExCat[:],
                                 repfactor, SExCatalogue=True,
                                 noVisualSelection=False, quickFit=True,
                                 autoTrim=autoTrim)
  try:
    os.rename('psfStarChooser.png', 'best_psfStarChooser.png')
  except:
    pass
  inspectedPS1Cat = findSharedPS1Catalogue([PS1SharedCat, inspectedSExCat])
  saveStarMag('InspectedStars.txt', inspectedPS1Cat,
              timeNow, __version__, 'All images', None, extno=extno)
  print('{}'.format(len(inspectedPS1Cat)) +
        ' PS1 sources left after manual inspection.')
  bestCatName = ('best.cat' if extno is None
                 else 'best{0:02.0f}.cat'.format(extno))
  pickleCatalogue(inspectedPS1Cat, bestCatName)
  return bestID, inspectedPS1Cat


if __name__ == '__main__':
  (images, repfact, verbatim, extension, ignoreWarnings, kronDifference,
   automaticTrim, SignalCut, ElongationCut) = getArguments(sys.argv)
  #Ignore all Python warnings.
  #This is generally a terrible idea, and should be turned off for de-bugging.
  if ignoreWarnings:
    warnings.filterwarnings("ignore")
  bestImage, bestCat = best(images, repfact, extno=extension, verbose=verbatim,
                            PSF_Kron=kronDifference, autoTrim=automaticTrim,
                            snrcut=SignalCut, shapecut=ElongationCut)
  print('Best catalogue:')
  print(bestCat)
  print('Best image #: ' + str(bestImage))


# End of file.
# Nothing to see here.
