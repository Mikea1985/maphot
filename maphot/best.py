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
from trippy import psfStarChooser, scamp, psf
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')


def getCatalogue(file_start):
  """getCatalog checks whether a catalog file already exists.
  If it does, it is read in. If not, it runs SExtractor to create it.
  """
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
  for ii, dummy in enumerate(imageArray):
    catalogueArray.append(getCatalogue(imageArray[ii]))
  return catalogueArray


def findSharedCatalogue(catalogueArray, useIndex):
  """Compare catalogues and create a master catalogue of only
  stars that are in all images
  """
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
  """This should return a catalogue dictionary formatted in the same
  way as the original catalogue, allowing us to do anything with it that
  we could do with any other catalogue. 
  """
  return sharedCatalogue


def inspectStars(file_start, catalogue, repfact, **kwargs):
  """Run psfStarChooser, inspect stars, generate PSF and lookup table.
  """
  SExCatalogue = kwargs.pop('return_SExtractor_catalogue', False)
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
                     repFact=repfact,
                     includeCheesySaturationCut=False,
                     verbose=False)
    print("\ngoodFits = ", goodFits, "\n")
    print("\ngoodMeds = ", goodMeds, "\n")
    print("\ngoodSTDs = ", goodSTDs, "\n")
    goodPSF = psf.modelPSF(np.arange(61), np.arange(61), alpha=goodMeds[2],
                           beta=goodMeds[3], repFact=repfact)
    fwhm = goodPSF.FWHM()  # this is the pure moffat FWHM
    print("fwhm = ", fwhm)
    goodPSF.genLookupTable(data, goodFits[:, 4], goodFits[:, 5], verbose=False)
    goodPSF.genPSF()
    fwhm = goodPSF.FWHM()  # this is the FWHM with lookuptable included
    print("fwhm = ", fwhm)
  except UnboundLocalError:
    print("Data error occurred!")
    raise
  if SExCatalogue:
    bestCatalogue = extractGoodStarCatalogue(catalogue,
                                             goodFits[:, 4], goodFits[:, 5])
    return bestCatalogue
  else:
    return goodFits, goodMeds, goodSTDs, goodPSF, fwhm


def extractGoodStarCatalogue(startCatalogue, xcat, ycat):
  """extractBestStarCatalogue crops a Catalogue,
  leaving only the stars that match a given x & y list.
  """
  trimCatalogue = {'XWIN_IMAGE': xcat,
                   'YWIN_IMAGE': ycat}
  goodCatalogue = findSharedCatalogue([startCatalogue, trimCatalogue], 0)
  return goodCatalogue



#
