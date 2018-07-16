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
sexparfile = 'sex.pars'  # Change with '-s filename' or '--sexparfile filename'
coordsfile is a file that contains:
x1 y1 MJD1
x2 y2 MJD2
ie. the position of the TNO at two times, and those times in MJD format.
The script then reads the MJD keyword from the input files and extrapolates
in order to predict the location in that image.
"""

from __future__ import print_function, division
import os
import getopt
import sys
import numpy as np
import mp_ephem
import astropy.io.fits as pyf
from astropy.io.votable import parse_single_table
from astropy.table import Column
from astropy import wcs
from trippy import scamp, MCMCfit
import requests
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')
__version__ = 0.3


def queryPanSTARRS(ra_deg, dec_deg, rad_deg=0.1, mindet=1, maxsources=10000,
                   server=('https://archive.stsci.edu/panstarrs/search.php'),
                   catalog_filename='panstarrs.xml'):
  '''
  This function is inspired by Michael Mommert's wordpress post about querying
  PanSTARRS1 from Python:
  https://michaelmommert.wordpress.com/2017/02/13/
  accessing-the-gaia-and-pan-starrs-catalogs-using-python/
  Query Pan-STARRS DR1 @ MAST
  parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field radius in degrees
              mindet: minimum number of detection (optional)
              maxsources: maximum number of sources
              server: servername
              catalog_filename: the filename to save the catalog query to.
  returns: astropy.table object
  '''
  r = requests.get(server, params={'RA': ra_deg, 'DEC': dec_deg,
                                   'SR': rad_deg, 'max_records': maxsources,
                                   'outputformat': 'VOTable',
                                   'ndetections': ('>%d' % mindet)})
  # write query data into local file
  outf = open(catalog_filename, 'w')
  outf.write(r.text)
  outf.close()
  return


def readPanSTARRS(catalog_filename='panstarrs.xml', rMin=0, gMin=0,
                  PSF_Kron=0.5):
  '''
  Read a PanSTARRS catalog from an xml file.
  Only include objects that have both g and r-band magnitudes.
  '''
  # Read an xml file and parse it into an astropy.table object.
  PS1CatDataFull = parse_single_table(catalog_filename)
  PS1All = PS1CatDataFull.to_table(use_names_over_ids=True)
  PS1Cat = PS1All[(PS1All['rMeanPSFMag'] > rMin)
                  & (PS1All['gMeanPSFMag'] > gMin)
                  & (PS1All['rMeanPSFMag'] - PS1All['rMeanKronMag'] < PSF_Kron)
                  & (PS1All['gMeanPSFMag']
                     - PS1All['gMeanKronMag'] < PSF_Kron)]
  return PS1Cat


def PS1_vs_SEx(PS1Cat, SExCat, maxDist=1):
  '''
  Match sources in the PanSTARRS and Source Extractor catalogs.
  Return only the overlapping catalog, with all columns from both catalogs.
  With this, we probably don't need to use catalogTrim. Maybe? Let's see.
  '''
  SExArgs = []
  PS1Args = []
  for ii, RADeci in enumerate(PS1Cat['raMean', 'decMean']):
    distance = ((SExCat['X_WORLD'] - RADeci[0]) ** 2 +
                (SExCat['Y_WORLD'] - RADeci[1]) ** 2) ** 0.5
    dminSExArg = np.argsort(distance)[0]
    if distance[dminSExArg] < maxDist / 3600.:
      SExArgs.append(dminSExArg)
      PS1Args.append(ii)
  PS1SExCatalog = PS1Cat[PS1Args]
  PS1SExCatalog.add_columns([Column(SExCat[key][SExArgs], key)
                             for key in SExCat.keys()])
  return PS1SExCatalog


def trimCatalog(cat, somedata, dcut, mcut, snrcut, shapecut, naxis1, naxis2):
  """trimCatalog trims the SExtractor catalogue of non-roundish things,
  really bright things and things that are near other things.
  cat = the full catalogue from SExtractor.
  dcut = the minimum acceptable distance between sources.
  mcut = maximum count (in image counts); removes really bright sources.
  snrcut = minimum Signal-to-Noise to keep; remove faint objects.
  shapecut = maximum long-axis/short-axis shape value; remove galaxies.
  naxis1, naxis2 = dimensions of the CCD/image.
  """
  good = []
  for ii in range(len(cat['XWIN_IMAGE'])):
    try:
      a = int(cat['XWIN_IMAGE'][ii])
      b = int(cat['YWIN_IMAGE'][ii])
      m = np.max(somedata[b - 4:b + 5, a - 4:a + 5])
    except:
      pass
    xi = cat['XWIN_IMAGE'][ii]
    yi = cat['YWIN_IMAGE'][ii]
    distance = np.sort(((cat['XWIN_IMAGE'] - xi) ** 2 +
                        (cat['YWIN_IMAGE'] - yi) ** 2) ** 0.5)
    d = distance[1]
    snrs = cat['FLUX_AUTO'][ii] / cat['FLUXERR_AUTO'][ii]
    shape = cat['AWIN_IMAGE'][ii] / cat['BWIN_IMAGE'][ii]
    if (cat['FLAGS'][ii] == 0
       and d > dcut
       and m < mcut
       and snrs > snrcut
       and shape < shapecut
       and xi > dcut + 1 and xi < naxis1 - dcut - 1
       and yi > dcut + 1 and yi < naxis2 - dcut - 1):
      good.append(ii)
  good = np.array(good)
  outcat = {}
  for ii in cat:
    outcat[ii] = cat[ii][good]
  return outcat


def getObservations(mpc_lines):
  '''Parces MPC lines and generates an mp_ephem observation.'''
  observationList = []
  for _, mpc_line in enumerate(mpc_lines):
    date = mpc_line[15:31]
    ra = mpc_line[32:44]
    dec = mpc_line[44:57]
    obsCode = mpc_line[-4:-1]
    observationList.append(mp_ephem.ephem.Observation(ra=ra, dec=dec,
                                                      date=date,
                                                      observatory_code=obsCode)
                           )
  return observationList


def coordRateAngle(orbit, MJDate, WCS, obs_code=568):
  '''Given an orbit and a date (MJDate),
  calculates the rate and angle of motion
  as seen from a given observatory (default is 568 - Mauna Kea, Hawai'i).'''
  orbit.predict(MJDate + 2400000.5, obs_code=obs_code)
  ra0, dec0 = orbit.coordinate.ra.degree, orbit.coordinate.dec.degree
  orbit.predict(MJDate + 2400000.5 + 1. / 24.0, obs_code=obs_code)
  ra1, dec1 = orbit.coordinate.ra.degree, orbit.coordinate.dec.degree
  #rate_deg = ((np.cos(dec0 * np.pi / 180) * (ra1 - ra0)) ** 2
  #            + (dec1 - dec0) ** 2) ** 0.5  # degrees per hour
  coords = WCS.wcs_world2pix(np.array([[ra0, dec0], [ra1, dec1]]), 1)
  rate_pix = ((coords[1, 0] - coords[0, 0]) ** 2
              + (coords[1, 1] - coords[0, 1]) ** 2) ** 0.5  # pix per hour
  angle_pix = (np.arctan2(coords[1, 1] - coords[0, 1],
                          coords[1, 0] - coords[0, 0]) * 180. / np.pi) % 180
  return coords[0, :], rate_pix, angle_pix


def writeSExParFiles(imageFileName, params):
  '''
  This writes a Source Extractor parameter file.
  '''
  sexFile = imageFileName.replace('.fits', '.sex')
  os.system('rm {}'.format(sexFile))
  os.system('rm def.param')
  os.system('rm default.conv')
  scamp.makeParFiles.writeSex(sexFile,
                              minArea=params[0], threshold=params[1],
                              zpt=params[2], aperture=params[3],
                              kron_factor=params[4], min_radius=params[5],
                              catalogType='FITS_LDAC', saturate=110000)
  scamp.makeParFiles.writeConv()
  scamp.makeParFiles.writeParam('def.param', numAps=1)


def runSExtractor(imageFileName, SExParams):
  '''
  Run Source Extractor. Provide a useful error if it fails.
  '''
  SExtractorFile = imageFileName.replace('.fits', '.sex')
  catalogFile = imageFileName.replace('.fits', '.cat')
  writeSExParFiles(imageFileName, SExParams)
  try:
    scamp.runSex(SExtractorFile, imageFileName,
                 options={'CATALOG_NAME': catalogFile})
    fullcatalog = scamp.getCatalog(catalogFile, paramFile='def.param')
  except IOError as error:
    raise IOError('\n{}\nYou have almost certainly forgotten '.format(error) +
                  'to activate Ureka or AstroConda!')
  return fullcatalog


def getSExCatalog(imageFileName, SExParams, verb=True):
  '''Checks whether a catalog file already exists.
  If it does, it is read in. If not, it runs Source Extractor to create it.
  '''
  catalogFile = imageFileName.replace('.fits', '.cat')
  try:
    fullcatalog = scamp.getCatalog(catalogFile, paramFile='def.param')
  except IOError:
    fullcatalog = runSExtractor(imageFileName, SExParams)
  except UnboundLocalError:
    print("\nData error occurred!\n")
    raise
  ncat = len(fullcatalog['XWIN_IMAGE'])
  print("\n" + str(ncat) + " catalog stars\n" if verb else "")
  return fullcatalog


def runMCMCCentroid(centPSF, centData, centxt, centyt, centm,
                    centbg, centdtransx, centdtransy,
                    repfact):
  """runMCMCCentroid runs an MCMC centroiding, fitting the TSF to the data.
  Returns the fitted centoid co-ordinates.
  """
  print("Should I be doing this?")
  print("MCMC-fitting TSF to the moving object")
  centfitter = MCMCfit.MCMCfitter(centPSF, centData)
  centfitter.fitWithModelPSF(centdtransx + centxt - int(centxt),
                             centdtransy + centyt - int(centyt),
                             m_in=centm / repfact ** 2.,
                             fitWidth=10, nWalkers=10,
                             nBurn=20, nStep=20, bg=centbg, useLinePSF=True,
                             verbose=True, useErrorMap=False)
  (centfitPars, centfitRange) = centfitter.fitResults(0.67)
# Reverse the above coordinate transformation:
  xcentroid, ycentroid = centfitPars[0:2] \
                         - [centdtransx, centdtransy] \
                         + [int(centxt), int(centyt)]  # noqa
  return xcentroid, ycentroid, centfitPars, centfitRange


def getArguments(sysargv, useage):
  """Get arguments given when this is called from a command line"""
  AinputFile = 'a100.fits'  # Change with '-f <filename>' flag
  Acoordsfile = 'coords.in'  # Change with '-c <coordsfile>' flag
  Averbose = False  # Change with '-v True' or '--verbose True'
  Acentroid = False  # Change with '-. False' or  --centroid False'
  AoverrideSEx = False  # Change with '-o True' or '--override True'
  Aremove = False  # Change with '-r False' or '--remove False'
  Aaprad = -42.
  Arepfact = 10
  Apxscale = 1.0
  AroundAperRad = 1.4
  Asexparfile = 'sex.pars'
  Aextno = 0
  try:
    options, dummy = getopt.getopt(sysargv[1:], "f:c:v:.:o:r:a:h:s:e:",
                                   ["ifile=", "coords=", "verbose=",
                                    "centroid=", "overrideSEx=",
                                    "remove=", "aprad=", "sexparfile=",
                                    "extension="])
    for opt, arg in options:
      if (opt in ("-v", "-verbose", "-.", "--centroid", "-o", "--overrideSEx",
                  "-r", "--remove")):
        if arg == '0' or arg == 'False':
          arg = False
        elif arg == '1' or arg == 'True':
          arg = True
        else:
          print(opt, arg, np.array([arg]).dtype)
          raise TypeError("-v -. -o -r flags must be followed by " +
                          "0/False/1/True")
      if opt == '-h':
        print(useage)
      elif opt in ('-f', '--ifile'):
        AinputFile = arg
      elif opt in ('-c', '--coords'):
        Acoordsfile = arg
      elif opt in ('-v', '--verbose'):
        Averbose = arg
      elif opt in ('-.', '--centroid'):
        Acentroid = arg
      elif opt in ('-o', '--overrideSEx'):
        AoverrideSEx = arg
      elif opt in ('-r', '--remove'):
        Aremove = arg
      elif opt in ('-a', '--aprad'):
        Aaprad = float(arg)
      elif opt in ('-s', '--sexparfile'):
        Asexparfile = float(arg)
      elif opt in ('-e', '--extension'):
        Aextno = float(arg)
  except TypeError as error:
    print(error)
    sys.exit()
  except getopt.GetoptError as error:
    print(" Input ERROR! ")
    print(useage)
    sys.exit(2)
  return (AinputFile, Acoordsfile, Averbose, Acentroid,
          AoverrideSEx, Aremove, Aaprad, Arepfact, Apxscale, AroundAperRad,
          Asexparfile, Aextno)


def findTNO(xzero, yzero, fullcat, outfile):
  """Finds the nearest catalogue entry to the estimated location."""
  dist = ((fullcat['XWIN_IMAGE'] - xzero) ** 2
          + (fullcat['YWIN_IMAGE'] - yzero) ** 2) ** 0.5
  args = np.argsort(dist)
  print("\n x0, y0 = ", xzero, yzero, "\n")
  outfile.write("\nx0, y0 = {}, {}\n".format(xzero, yzero))
  xtno = fullcat['XWIN_IMAGE'][args][0]
  ytno = fullcat['YWIN_IMAGE'][args][0]
  if (xtno - xzero) ** 2 + (ytno - yzero) ** 2 > 36:
    print("\n   WARNING! Object not found at", xzero, yzero, "\n")
    outfile.write("\n   WARNING! Object not found at {}, {}\n".format(xzero,
                                                                      yzero))
    xtno, ytno = xzero, yzero
  return xtno, ytno


def predicted2catalog(someCatalog, someCoords):
  '''Given a predicted coordinate and a source catalog,
  find the nearest source in the catalog and use its coordinates.
  We might need to edit this if Source Extractor doesn't always find the TNO.
  '''
  ## Check whether someCoords is a single x+y coord or an array of coords.
  coordShape = np.shape(someCoords)
  if coordShape == (2,):
    theseCoords = np.array([someCoords])
  elif (len(coordShape) == 2) & (coordShape[0] == 2):
      theseCoords = someCoords.T
  else:
    raise TypeError('Coordinates must be in an array of shape (2, ?)')
  #Find the distance between the coords and all catalog points. Pick closest.
  args = []
  minDist = []
  catCoords = []
  for cooi in theseCoords:
    distance = ((someCatalog['XWIN_IMAGE'] - cooi[0]) ** 2
                + (someCatalog['YWIN_IMAGE'] - cooi[1]) ** 2) ** 0.5
    arg = np.argsort(distance)[0]
    args.append(arg)
    minDist.append(distance[arg])
    catCoords.append([someCatalog['XWIN_IMAGE'][arg],
                      someCatalog['YWIN_IMAGE'][arg]])
  for di in np.arange(20, 5, -1):
    dsum = np.sum(minDist > di)
    if dsum:
      print('WARNING! {} sets of coordinates '.format(dsum) +
            'were shifted by more than {} pixels'.format(di))
  return np.array(catCoords).T


def saveTNOMag(image_fn, mpc_fn, headerMJD, obsMJD, SExTNOCoord, x_tno, y_tno,
               zpt, obsFILTER, FWHM, aperMulti, TNOphot,
               magCalibration, dmagCalibration, finalTNOphotINST, zptGood,
               finalTNOphotPS1, timeNow, TNObgRegion, version):
  '''Save the TNO magnitude and other information.'''
  filtStr = obsFILTER + 'RawMag'
  mag_heads = ''
  mag_strings = ''
  for ai, am in enumerate(aperMulti):
    mag_heads += filtStr + str(am) + '\td' + filtStr + str(am) + '\t'
    mag_strings += '{}\t{}\t'.format(TNOphot.magnitude[ai],
                                     TNOphot.dmagnitude[ai])
  TNOFileName = image_fn.replace('.fits', '_TNOmag.txt')
  TNOFile = open(TNOFileName, 'w')
  TNOFile.write('#Filename\tObject\tMJD\tMJD_middle\t' +
                'RA(deg)\tDec(deg)\t' +
                'x_pix\ty_pix\tZptRaw\t' +
                'Filter\tFWHM\t' + mag_heads +
                'MagCalibration\tdMagCalibration\t' +
                'GoodMag_Inst\tdGoodMag_Inst\t' +
                'ZptGood\tRunTime\t' +
                'GoodMag_PS1\tdGoodMag_PS1\t' +
                'bgXMin\tbgXMax\tbgYMin\tbgYMax\t' +
                'gphot_version\n')
  TNOFile.write('{}\t{}\t'.format(image_fn, mpc_fn) +
                '{}\t{}\t'.format(headerMJD, obsMJD) +
                '{}\t{}\t'.format(SExTNOCoord[0], SExTNOCoord[1]) +
                '{}\t{}\t{}\t'.format(x_tno[0], y_tno[0], zpt) +
                '{}\t{}\t'.format(obsFILTER, FWHM) + mag_strings +
                '{}\t{}\t'.format(magCalibration, dmagCalibration) +
                '{}\t{}\t'.format(finalTNOphotINST[0], finalTNOphotINST[1]) +
                '{}\t{}\t'.format(zptGood, timeNow) +
                '{}\t{}\t'.format(finalTNOphotPS1[0], finalTNOphotPS1[1]) +
                '{}\t{}\t{}\t{}\t'.format(*TNObgRegion) +
                '{}\n'.format(version))
  TNOFile.close()
  return


def saveStarMag(image_fn, finalCat, timeNow, version, headerMJD):
  '''Save the star magnitudes and other information.'''
  starFileName = image_fn.replace('.fits', '_starmag.txt')
  starFile = open(starFileName, 'w')
  starFile.write('#Run time: {}\n'.format(timeNow))
  starFile.write('#Image name and MJD: {} {}\n'.format(image_fn, headerMJD))
  starFile.write('#gphot version: {}\n'.format(version))
  starFile.write(''.join(['{}\t'.format(keyi) for keyi in finalCat.keys()])
                 + '\n')
  for row in finalCat:
    starFile.write(''.join(['{}\t'.format(row[keyi])
                            for keyi in finalCat.keys()]) + '\n')
  starFile.close()
  return


def PS1_to_Gemini(catalog):
  '''Transform the PS1 magnitudes of catalog stars to Gemini magnitudes,
  using the filter transforms from Schwamb et al 2018.'''
  PS1rGemini = Column(catalog['rMeanPSFMag'] - 0.052 *
                      (catalog['gMeanPSFMag'] - catalog['rMeanPSFMag']),
                      name='rMeanPSFMag_Gemini',
                      description=catalog['rMeanPSFMag'].description +
                      ' transformed to Gemini filter')
  PS1gGemini = Column(catalog['gMeanPSFMag'] + 0.0369 *
                      (catalog['gMeanPSFMag'] - catalog['rMeanPSFMag']),
                      name='gMeanPSFMag_Gemini',
                      description=catalog['gMeanPSFMag'].description +
                      ' transformed to Gemini filter')
  GeminiCat = catalog.copy()
  GeminiCat.add_columns((PS1rGemini, PS1gGemini))
  return GeminiCat


def PS1_to_CFHT(catalog, filters='griz'):
  '''Transform the PS1 magnitudes of catalog stars to CFHT magnitudes,
  using the filter transforms from CADC:
  http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html .
  Note, these are for the new 2015 filter set.'''
  PS1g = catalog['gMeanPSFMag']
  PS1i = catalog['iMeanPSFMag']
  PS1gi = PS1g - PS1i
  conpars = {'g': [0.014, 0.059, -0.00313, -0.00178],
             'r': [0.003, -0.050, 0.0125, -0.0069],
             'i': [0.006, -0.024, 0.00627, -0.00523],
             'z': [-0.016, -0.069, 0.0239, -0.0056]}
  CFHTCat = catalog.copy()
  for CFHfilt in filters:
    c = conpars[CFHfilt]
    PS1toCFHT = Column(catalog[CFHfilt + 'MeanPSFMag'] + c[0] + c[1] * PS1gi
                       + c[2] * PS1gi ** 2 + c[3] * PS1gi ** 3,
                       name=CFHfilt + 'MeanPSFMag_CFHT',
                       description=catalog[CFHfilt + 'MeanPSFMag'].description
                       + ' transformed to CFHT filter')
    CFHTCat.add_columns((PS1toCFHT))
  return CFHTCat


def CFHT_to_PS1(magnitude, mag_uncertainty, filter_name='r'):
  '''Transform CFHT magnitude to PS1 magnitudes,
  using the filter transforms from CADC:
  http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html ,
  and the median colours from Olef 2012:
  http://iopscience.iop.org/article/10.1088/0004-637X/749/1/10/meta ,
  converted to CFHT colours using the CHFT transforms.
  Note, these are for the new 2015 filter set.'''
  gr = (0.69, 0.17)  # g-r calculated from Olef 2012
  ri = (0.28, 0.17)  # r-i
  iz = (0.22, 0.20)  # i-z and std
  conpars = {'g': [gr, -0.015, -0.067, 0],
             'r': [ri, -0.002, +0.154, 0],
             'i': [ri, -0.002, +0.087, 0],
             'z': [iz, +0.001, +0.175, +0.013]}
  c = conpars[filter_name]
  conversion = c[1] + c[2] * c[0][0] + c[3] * c[0][0] ** 2
  conv_uncertainty = (c[2] + 2 * c[3] * c[0][0]) * c[0][1]
  print(mag_uncertainty, conv_uncertainty)
  PS1mag = magnitude + conversion
  PS1mag_uncertainty = (mag_uncertainty ** 2 + conv_uncertainty ** 2) ** 0.5
  return PS1mag, PS1mag_uncertainty


def addPhotToCatalog(X, Y, catTable, photDict):
  '''Match good stars from starChooser in a SExtractor catalog (astropy table).
  Trims the catalog and adds the columns for our photometry.
  Since the X and Y coordinates should already come from the catalog,
  maching should be unique '''
  tableArgs = []
  photDictArgs = []
  for i, XYi in enumerate(catTable['XWIN_IMAGE', 'YWIN_IMAGE']):
    for j in np.arange(len(X)):
      if (XYi[0] == X[j]) & (XYi[1] == Y[j]):
        tableArgs.append(i)
        photDictArgs.append(j)
  bestCat = catTable[tableArgs]
  bestCat.add_columns([Column(photDict[key][photDictArgs], key)
                       for key in np.sort(photDict.keys())])
  return bestCat


def getDataHeader(inputFile, extno=None):
  '''Reads in a fits file (or a given extension of one).
  Returns the image data, the header, and a few useful keyword values.'''
  with pyf.open(inputFile) as han:
    if extno is None:
      data = han.data
      header = han.header
    else:
      data = han[extno].data
      header = han[extno].header
    EXPTIME = header['EXPTIME']
    try:
      MAGZERO = header['MAGZERO']  # Subaru Hyper-Suprime
    except:
      try:
        MAGZERO = header['PHOT_C']   # CFHT MegaCam
      except:
        MAGZERO = 26.0
    try:
      MJD = header['MJD']  # Subaru
    except:
      try:
        MJD = header['MJDATE']  # CFHT
      except:
        MJD = header['MJD-OBS']  # Gemini/CFHT
    try:
      GAIN = header['GAINEFF']  # Subaru
    except:
      GAIN = header['GAIN']  # CFHT
    NAXIS1 = header['NAXIS1'] if header['NAXIS1'] > 128 else header['ZNAXIS1']
    NAXIS2 = header['NAXIS2'] if header['NAXIS2'] > 128 else header['ZNAXIS2']
  WCS = wcs.WCS(header)
  return data, header, EXPTIME, MAGZERO, MJD, GAIN, NAXIS1, NAXIS2, WCS


# End of file.
# Nothing to see here.
