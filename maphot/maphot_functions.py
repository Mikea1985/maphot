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
import io
import getopt
import sys
import pylab as pyl
import mp_ephem
from six.moves import input
import numpy as np
import requests
import astropy.io.fits as pyf
from astropy.visualization import interval, ZScaleInterval
from astropy.io.votable import parse_single_table
from astropy.table import Column, Table
from astropy import wcs
from trippy import scamp, MCMCfit, psf, psfStarChooser
#from stsci import numdisplay  # pylint: disable=import-error
from __version__ import __version__
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')
print("You are using maphot version: ", __version__)


def queryPanSTARRS(ra_deg, dec_deg, rad_deg=0.1, mindet=1, maxsources=50000,
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
                  PSF_Kron=0.5, verbose=False):
  '''
  Read a PanSTARRS catalog from an xml file.
  Only include objects that have both g and r-band magnitudes.
  '''
  # Read an xml file and parse it into an astropy.table object.
  PS1CatDataFull = parse_single_table(catalog_filename)
  PS1All = PS1CatDataFull.to_table(use_names_over_ids=True)
  Ktfr = (np.ones_like(PS1All['rMeanPSFMag'], dtype=bool) if PSF_Kron is None
          else (PS1All['rMeanPSFMag'] - PS1All['rMeanKronMag'] < PSF_Kron))
  Ktfg = (np.ones_like(PS1All['gMeanPSFMag'], dtype=bool) if PSF_Kron is None
          else (PS1All['gMeanPSFMag'] - PS1All['gMeanKronMag'] < PSF_Kron))
  magtfr = (PS1All['rMeanPSFMag'] > rMin)
  magtfg = (PS1All['gMeanPSFMag'] > gMin)
  alltf = magtfr & magtfg & Ktfr & Ktfg
  PS1Cat = PS1All[alltf]
  if verbose:
    nPS1 = len(PS1All['raMean'])
    print('Of {} PS1 stars,\n'.format(nPS1) +
          '{} did not have r-band PSF mag,\n'.format(nPS1 - sum(magtfr)) +
          '{} did not have g-band PSF mag,\n'.format(nPS1 - sum(magtfg)) +
          '{} did not meet r-band PSF-Kron limit,\n'.format(nPS1 - sum(Ktfr)) +
          '{} did not meet g-band PSF-Kron limit,\n'.format(nPS1 - sum(Ktfg)) +
          'leaving just {} useful PS1 stars.'.format(len(PS1Cat['raMean'])))
  return PS1Cat


def PS1_vs_SEx(PS1Cat, SExCat, maxDist=1, appendSEx=True):
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
  if appendSEx:
    PS1SExCatalog.add_columns([Column(SExCat[key][SExArgs], key)
                               for key in SExCat.keys()])
  return PS1SExCatalog


def trimCatalog(cat, somedata, dcut, mcut, snrcut, shapecut, naxis1, naxis2,
                instrument=None, verbose=False):
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
  rejected = io.open('rejected_stars.txt', 'w') if verbose else ""
  print('Treating as MODS data, trimming edges off.\n'
        if (verbose and ('MODS' in instrument)) else "", end='')
  for ii in range(len(cat['XWIN_IMAGE'])):
    try:
      a = int(cat['XWIN_IMAGE'][ii])
      b = int(cat['YWIN_IMAGE'][ii])
      m = np.max(somedata[b - 4:b + 5, a - 4:a + 5])
    except:
      m = 666 * mcut  # If try fails, is because source too near edge
    xi = cat['XWIN_IMAGE'][ii]
    yi = cat['YWIN_IMAGE'][ii]
    distance = np.sort(((cat['XWIN_IMAGE'] - xi) ** 2 +
                        (cat['YWIN_IMAGE'] - yi) ** 2) ** 0.5)
    d = distance[1]
    snrs = cat['FLUX_AUTO'][ii] / cat['FLUXERR_AUTO'][ii]
    shape = cat['AWIN_IMAGE'][ii] / cat['BWIN_IMAGE'][ii]
    flagtf = (cat['FLAGS'][ii] < 4)
    disttf = (d > dcut)
    mtf = (m < mcut)
    snrtf = (snrs > snrcut)
    shapetf = (shape < shapecut)
    buffercut = 31
    xytf = (xi > buffercut and xi < naxis1 - buffercut
            and yi > buffercut and yi < naxis2 - buffercut)
    if flagtf and disttf and mtf and snrtf and shapetf and xytf:
      if (instrument == 'GMOS-N') or (instrument == 'GMOS-S'):
        if (xi > 1046 + buffercut and xi < 2064 - buffercut
            and yi > 13 + buffercut and yi < 2271 - buffercut):
          good.append(ii)
        else:
          rejected.write(u'Star at {}, {} trimmed: '.format(xi, yi) +
                         u'not on central chip.\n') if verbose else ""
      elif 'MODS' in instrument:
        if (xi > 18 + buffercut and xi < 3049 - buffercut
            and yi > 39 + buffercut and yi < 3067 - buffercut):
          good.append(ii)
        else:
          rejected.write(u'Star at {}, {} trimmed: '.format(xi, yi) +
                         u'too near edge.\n') if verbose else ""
      else:
          good.append(ii)
    else:
      rejected.write(u'Star at {}, {} trimmed: Flag {}'.format(xi, yi, flagtf)
                     + u'dist {}, m {}, snr {}, '.format(disttf, mtf, snrtf)
                     + u'shape {}, xy {}.\n'.format(shapetf, xytf)
                     ) if verbose else ""
  rejected.close() if verbose else ""
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


def writeSExParFiles(imageFileName, minArea, threshold, zpt, aperture,
                     kron_factor, min_radius, extno=None, verbose=False):
  '''
  This writes a Source Extractor parameter file.
  '''
  if extno is None:
    print('Warning: Treating this as a single extension file.')
    sexFile = imageFileName.replace('.fits', '.sex')
  else:
    sexFile = imageFileName.replace('.fits', '{0:02.0f}.sex'.format(extno))
  os.system('rm {}'.format(sexFile))
  os.system('rm def.param')
  os.system('rm default.conv')
  if np.shape(aperture):
    aperture = list(aperture)
  else:
    aperture = [aperture]
  scamp.makeParFiles.writeSex(sexFile,
                              minArea=minArea, threshold=threshold,
                              zpt=zpt, aperture=aperture,
                              kron_factor=kron_factor, min_radius=min_radius,
                              catalogType='FITS_LDAC', saturate=120000)
  if verbose:
    scamp.makeParFiles.writeSex(sexFile + '2',
                                minArea=minArea, threshold=threshold,
                                zpt=zpt, aperture=aperture,
                                kron_factor=kron_factor, min_radius=min_radius,
                                catalogType='ASCII', saturate=120000)
  scamp.makeParFiles.writeConv()
  scamp.makeParFiles.writeParam('def.param', numAps=1)


def runSExtractor(imageFileName, SExParams, extno=None, verbose=False):
  '''  Run Source Extractor. Provide a useful error if it fails.
  '''
  if extno is None:
    print('Warning: Treating this as a single extension file.')
    SExtractorFile = imageFileName.replace('.fits', '.sex')
    catalogFile = imageFileName.replace('.fits', '.cat')
    imageFNE = imageFileName
  else:
    SExtractorFile = imageFileName.replace('.fits',
                                           '{0:02.0f}.sex'.format(extno))
    catalogFile = imageFileName.replace('.fits', '{0:02.0f}.cat'.format(extno))
    imageFNE = imageFileName + '[{}]'.format(extno)
  writeSExParFiles(imageFileName, *SExParams, extno=extno, verbose=verbose)
  try:
    scamp.runSex(SExtractorFile, imageFNE,
                 options={'CATALOG_NAME': catalogFile})
    if verbose:
      scamp.runSex(SExtractorFile + '2', imageFNE,
                   options={'CATALOG_NAME': catalogFile + '2'})
    fullcatalog = scamp.getCatalog(catalogFile, paramFile='def.param')
  except IOError as error:
    raise IOError('\n{}\nYou have almost certainly forgotten '.format(error) +
                  'to activate Ureka or AstroConda!')
  return fullcatalog


def getSExCatalog(imageFileName, SExParams, extno=None, verb=True):
  '''Checks whether a catalog file already exists.
  If it does, it is read in. If not, it runs Source Extractor to create it.
  '''
  if extno is None:
    print('Warning: Treating this as a single extension file.')
    catalogFile = imageFileName.replace('.fits', '{0:02.0f}.cat'.format(extno))
  else:
    catalogFile = imageFileName.replace('.fits', '{0:02.0f}.cat'.format(extno))
  try:
    fullcatalog = scamp.getCatalog(catalogFile, paramFile='def.param')
  except IOError:
    fullcatalog = runSExtractor(imageFileName, SExParams, extno=extno,
                                verbose=verb)
  except UnboundLocalError:
    print("\nData error occurred!\n")
    raise
  ncat = len(fullcatalog['XWIN_IMAGE'])
  print(str(ncat) + " sources in Source Extractor catalog\n"
        if verb else "", end="")
  return fullcatalog


def runMCMCCentroid(centPSF, centData, centxt, centyt, centm,
                    centbg, centdtransx, centdtransy,
                    repfact, tnotrack=False):
  """runMCMCCentroid runs an MCMC centroiding, fitting the TSF to the data.
  Returns the fitted centoid co-ordinates.
  """
  print("MCMC-fitting TSF to the moving object")
  centfitter = MCMCfit.MCMCfitter(centPSF, centData)
  centfitter.fitWithModelPSF(centdtransx + centxt,
                             centdtransy + centyt,
                             m_in=centm / repfact ** 2.,
                             fitWidth=10, nWalkers=20,
                             nBurn=20, nStep=30, bg=centbg,
                             useLinePSF=not(tnotrack),
                             verbose=False, useErrorMap=False)
  (centfitPars, centfitRange) = centfitter.fitResults(0.67)
# Reverse the above coordinate transformation:
  xcentroid = centfitPars[0] - centdtransx
  ycentroid = centfitPars[1] - centdtransy
  return xcentroid, ycentroid, centfitPars, centfitRange


def getArguments(sysargv):
  """Get arguments given when this is called from a command line"""
  useage = ('maphot -c <MPCfile> -f <imagefile> -e <extension>'
            + ' -i <ignoreWarnings> [-v <verbose>  -. <centroid> '
            + '-o <overrideSEx> -r <remove> -a <aprad> -s <sexparfile>'
            + '-t <tnotrack>]')
  AinputFile = 'a100.fits'  # Change with '-f <filename>' flag
  Acoordsfile = 'coords.in'  # Change with '-c <coordsfile>' flag
  Averbose = False  # Change with '-v True' or '--verbose True'
  Acentroid = False  # Change with '-. False' or  --centroid False'
  AoverrideSEx = False  # Change with '-o True' or '--override True'
  Aremove = False  # Change with '-r False' or '--remove False'
  Atnotrack = False  # true tracks TNO
  Aaprad = 0.7
  AroundAperRad = 0.7  # np.arange(0.5, 3.1, 0.1)
  AroundAperRad = np.arange(0.5, 3.1, 0.1)
  #AroundAperRad = np.array([0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
  #AroundAperRad = np.array([0.698, 0.699, 0.700, 0.701, 0.702, 0.8, 0.9, 1.0])
  Arepfact, Apxscale, = 10, 1.0
  Asexparfile, Aextno = None, 1
  AignoreWarnings = False
  try:
    options, dummy = getopt.getopt(sysargv[1:], "f:c:v:.:o:r:a:h:s:e:i:t:",
                                   ["imagefile=", "MPCfile=", "verbose=",
                                    "centroid=", "overrideSEx=",
                                    "remove=", "aprad=", "sexparfile=",
                                    "extension=", "ignoreWarnings=",
                                    "tracktno="])
    for opt, arg in options:
      if (opt in ("-v", "-verbose", "-.", "--centroid", "-o", "--overrideSEx",
                  "-r", "--remove", "-t", "--tracktno",
                  "-i", "--ignoreWarnings")):
        if arg == '0' or arg == 'False':
          arg = False
        elif arg == '1' or arg == 'True':
          arg = True
        else:
          print(opt, arg, np.array([arg]).dtype)
          raise TypeError("-v -. -o -r -i flags must be followed by " +
                          "0/False/1/True")
      if opt == '-h':
        print(useage)
      elif opt in ('-f', '--imagefile'):
        AinputFile = arg
      elif opt in ('-c', '--MPCfile'):
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
        Aextno = int(arg)
      elif opt in ('-i', '--ignoreWarnings'):
        AignoreWarnings = arg
      elif opt in ('-t', '--tracktno'):
        Atnotrack = arg
  except TypeError as error:
    print(error)
    sys.exit()
  except getopt.GetoptError as error:
    print(" Input ERROR! \n", useage)
    sys.exit(2)
  return (AinputFile, Acoordsfile, Averbose, Acentroid,
          AoverrideSEx, Aremove, Aaprad, Atnotrack, Arepfact, Apxscale,
          AroundAperRad, Asexparfile, Aextno, AignoreWarnings)


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
  return np.array(catCoords).T, minDist


def saveTNOMag(image_fn, mpc_fn, headerMJD, obsMJD, SExTNOCoord, x_tno, y_tno,
               zpt, obsFILTER, FWHM, aperMulti, TNOphot, TNOAperCorr,
               magCalibration, dmagCalibration, finalTNOphotINST, zptGood,
               finalTNOphotPS1, timeNow, TNObgRegion, version, extno=None):
  '''Save the TNO magnitude and other information.'''
  filtStr = obsFILTER + 'RawMag'
  mag_heads = ''
  mag_strings = ''
  cal_heads = ''
  cal_strings = ''
  good_heads = ''
  good_strings = ''
  zpt_heads = ''
  zpt_strings = ''
  try:
    print('{} is the first aperture, {} is the last'.format(aperMulti[0],
                                                            aperMulti[-1]))
  except:
    aperMulti = np.array([aperMulti])
  for ai, am in enumerate(aperMulti):
    mag_heads += filtStr + str(am) + '\td' + filtStr + str(am) + '\t'
    mag_strings += '{}\t{}\t'.format(TNOphot.magnitude[ai],
                                     TNOphot.dmagnitude[ai])
    cal_heads += 'MagCalib' + str(am) + '\tdMagCalib' + str(am) + '\t'
    cal_strings += '{}\t{}\t'.format(magCalibration[ai], dmagCalibration[ai])
    good_heads += 'GoodMag_Inst' + str(am) + '\tdGoodMag_Inst' + str(am) + '\t'
    good_strings += '{}\t{}\t'.format(finalTNOphotINST[0][ai],
                                      finalTNOphotINST[1][ai])
    zpt_heads += 'ZPTGood' + str(am) + '\t'
    zpt_strings += '{}\t'.format(zptGood[ai])
  if extno is None:
    print('Warning: Treating this as a single extension file.')
    TNOFileName = image_fn.replace('.fits', '_TNOmag.txt')
  else:
    TNOFileName = image_fn.replace('.fits',
                                   '{0:02.0f}_TNOmag.txt'.format(extno))
  TNOFile = open(TNOFileName, 'w')
  TNOFile.write('#Filename\tObject\tMJD\tMJD_middle\t' +
                'RA(deg)\tDec(deg)\t' +
                'x_pix\ty_pix\tZptRaw\t' +
                'Filter\tFWHM\t' + mag_heads + cal_heads +
                good_heads + zpt_heads + 'RunTime\t' +
                #'GoodMag_PS1\tdGoodMag_PS1\t' +
                'bgXMin\tbgXMax\tbgYMin\tbgYMax\t' +
                'maphot_version\n')
  TNOFile.write('{}\t{}\t'.format(image_fn, mpc_fn) +
                '{}\t{}\t'.format(headerMJD, obsMJD) +
                '{}\t{}\t'.format(SExTNOCoord[0], SExTNOCoord[1]) +
                '{}\t{}\t{}\t{}\t'.format(x_tno, y_tno, zpt, obsFILTER) +
                '{}\t'.format(FWHM) + mag_strings + cal_strings +
                good_strings + zpt_strings + '{}\t'.format(timeNow) +
                #'{}\t{}\t'.format(finalTNOphotPS1[0], finalTNOphotPS1[1]) +
                '{}\t{}\t{}\t{}\t'.format(*TNObgRegion) +
                '{}\n'.format(version))
  TNOFile.close()
  return


def saveTNOMag2(image_fn, mpc_fn, obsMJD, SExTNOCoord, x_tno, y_tno,
                obsFILTER, FWHM, finalTNOphotPS1, timeNow, version,
                extno=None):
  '''Save the TNO magnitude and other information.'''
  if extno is None:
    print('Warning: Treating this as a single extension file.')
    TNOFileName = 'TNOmags.txt'
  else:
    TNOFileName = 'TNOmags{0:02.0f}.txt'.format(extno)
  TNOFile = open(TNOFileName, 'a+')
  TNOFile.write('#Filename\tObject\tMJDm\t' +
                'RA(deg)\tDec(deg)\t' +
                'x_pix\ty_pix\t' +
                'Filter\tFWHM\t' +
                'GoodMag_PS1\tdGoodMag_PS1\t' +
                'RunTime\t' +
                'maphot_version\n')
  TNOFile.write(image_fn.replace('.fits', '') + '\t' +
                mpc_fn.replace('.mpc', '').replace('../MPC/', '') + '\t' +
                '{}\t'.format(obsMJD) +
                '{}\t{}\t'.format(SExTNOCoord[0], SExTNOCoord[1]) +
                '{}\t{}\t'.format(x_tno, y_tno) +
                '{}\t{}\t'.format(obsFILTER, FWHM) +
                '{}\t{}\t'.format(finalTNOphotPS1[0], finalTNOphotPS1[1]) +
                '{}\t'.format(timeNow) +
                '{}\n'.format(version))
  TNOFile.close()
  return


def saveStarMag(image_fn, finalCat, timeNow, version, headerMJD,
                sigmaclip=None, extno=None):
  '''Save the star magnitudes and other information.'''
  if extno is None:
    print('Warning: Treating this as a single extension file.')
    starFileName = image_fn.replace('.fits', '_starmag.txt')
  else:
    starFileName = image_fn.replace('.fits',
                                    '{0:02.0f}_starmag.txt'.format(extno))
  if sigmaclip is None:
    sigmaclip = np.ones(len(finalCat), dtype=bool)
  if np.sum(sigmaclip) == np.size(sigmaclip):
    print('Using all of the stars.')
  else:
    print('*** WARNING!!! Using sigma clipping! ***')
    print('** Check whether this is appropriate! **')
    print('*** WARNING!!! Using sigma clipping! ***')
  starFile = open(starFileName, 'w')
  starFile.write('#Run time: {}\n'.format(timeNow))
  starFile.write('#Image name and MJD: {} {}\n'.format(image_fn, headerMJD))
  starFile.write('#gphot version: {}\n'.format(version))
  starFile.write('#{}'.format(sigmaclip).replace('\n', '') + '\n')
  starFile.write(''.join(['{}\t'.format(keyi) for keyi in finalCat.keys()])
                 + '\n')
  for row in finalCat:
    starFile.write(''.join(['{}\t'.format(row[keyi])
                            for keyi in finalCat.keys()]) +
                   '\n')
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

def PS1_to_LBT(catalog):
  '''Transform the PS1 magnitudes of catalog stars to LBT magnitudes,
  using the filter transforms from LBT website.'''
  # note this rband is in the blue camera.
  PS1rGemini = Column(catalog['rMeanPSFMag'] + 0.016 *
                      (catalog['gMeanPSFMag'] - catalog['rMeanPSFMag']),
                      name='rMeanPSFMag_CFHT',
                      description=catalog['rMeanPSFMag'].description +
                      ' transformed to telescope filter')
  PS1gGemini = Column(catalog['gMeanPSFMag'] + 0.086 *
                      (catalog['gMeanPSFMag'] - catalog['rMeanPSFMag']),
                      name='gMeanPSFMag_CFHT',
                      description=catalog['gMeanPSFMag'].description +
                      ' transformed to telescope filter')
  PS1zGemini = Column(catalog['zMeanPSFMag'] + 0.020 *
                      (catalog['iMeanPSFMag'] - catalog['zMeanPSFMag']),
                      name='zMeanPSFMag_CFHT',
                      description=catalog['zMeanPSFMag'].description +
                      ' transformed to telescope filter')
  GeminiCat = catalog.copy()
  GeminiCat.add_columns((PS1rGemini, PS1gGemini, PS1zGemini))
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
  new_columns = []
  for key in CFHTCat.keys():
    new_columns.append(CFHTCat[key])
  for CFHfilt in filters:
    c = conpars[CFHfilt]
    PS1toCFHT = Column(catalog[CFHfilt + 'MeanPSFMag'] + c[0] + c[1] * PS1gi
                       + c[2] * PS1gi ** 2 + c[3] * PS1gi ** 3,
                       name=CFHfilt + 'MeanPSFMag_CFHT',
                       description=catalog[CFHfilt + 'MeanPSFMag'].description
                       + ' transformed to CFHT filter')
    new_columns.append(PS1toCFHT)
  newColumnsTable = Table(new_columns)
  return newColumnsTable


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


def LBT_not_to(magnitude, mag_uncertainty, filter_name='r'):
  '''don't do this here, it depends on color which we're measuring.'''
  return magnitude, mag_uncertainty


def addPhotToCatalog(X, Y, catTable, photDict):
  '''Match good stars from starChooser in a SExtractor catalog (astropy table).
  Trims the catalog and adds the columns for our photometry.
  Since the X and Y coordinates should already come from the catalog,
  maching should be unique '''
  tableArgs = []
  photDictArgs = []
  print(photDict.keys())
  for i, XYi in enumerate(catTable['XWIN_IMAGE', 'YWIN_IMAGE']):
    for j in np.arange(len(X)):
      if (XYi[0] == X[j]) & (XYi[1] == Y[j]):
        tableArgs.append(i)
        photDictArgs.append(j)
  print(photDictArgs)
  bestCat = catTable[tableArgs]
  bestCat.add_columns([Column(photDict[key][photDictArgs], key)
                       for key in np.sort(list(photDict.keys()))])
  return bestCat


def addTrippyToCat(catphotA, magStars, dmagStars, fluxStars, SNRStars,
                   bgStars, FILTER, apStr):
  """Add Trippy photometry for the stars to the catalogue."""
  magKeyName = FILTER + 'MagTrippy' + apStr
  catphotB = addPhotToCatalog(catphotA['XWIN_IMAGE'],
                              catphotA['YWIN_IMAGE'], catphotA,
                              {magKeyName: np.array(magStars),
                               'd' + magKeyName: np.array(dmagStars),
                               'TrippySourceFlux' + apStr:
                               np.array(fluxStars),
                               'TrippySNR' + apStr: np.array(SNRStars),
                               'TrippyBG' + apStr: np.array(bgStars)})
  return catphotB


def getDataHeader(inputFile, extno=None, verbose=False):
  '''Reads in a fits file (or a given extension of one).
  Returns the image data, the header, and a few useful keyword values.'''
  with pyf.open(inputFile) as han:
    if extno is None:
      print('Warning: Treating this as a single extension file.')
      data = han.data
      header = han.header
    else:
      data = han[extno].data
      header = han[extno].header
      header0 = han[0].header
  WCS = wcs.WCS(header)
  INST = header0['INSTRUME']  # Gemini, CFHT, Subaru, LBT
  if 'MODS' in INST:  # LBT MODS
    keys = ['EXPTIME', 27.8, 'MJD-OBS', 'GAINDL', 'FILTNAME']
  elif 'LBC' in INST:  # LBT LBC
    keys = ['EXPTIME', 27.8, 'MJD_OBS', 'GAIN', 'FILTER']
  elif INST == 'GMOS-N':  # Gemini N GMOS
    keys = ['EXPTIME', 27.8, 'MJD-OBS', 'GAINMULT', 'FILTER2']
  elif INST == 'MegaPrime':  # CFHT MegaPrime
    keys = ['EXPTIME', 'PHOT_C', 'MJD-OBS', 'GAIN', 'FILTER']
  elif INST == 'Hyper Suprime-Cam':  # Subaru HSC
    keys = ['EXPTIME', 'MAGZERO', 'MJD-STR', 'GAINEFF', 'FILTER']
  else:
    raise NameError('Instrument {} not recognised.'.format(INST))
  k1 = keys[1]
  EXPTIME = keyValue(header, header0, keys[0])
  MAGZERO = keyValue(header, header0, k1) if type(k1) == str else k1
  MJD = keyValue(header, header0, keys[2])
  GAIN = keyValue(header, header0, keys[3])
  FILTER = keyValue(header, header0, keys[4])[0]  # Just want first character
  MJDmid = MJD + EXPTIME / 172800.0
  NAXIS1 = header['NAXIS1'] if header['NAXIS1'] > 128 else header['ZNAXIS1']
  NAXIS2 = header['NAXIS2'] if header['NAXIS2'] > 128 else header['ZNAXIS2']
  print('{}\n'.format((EXPTIME, MAGZERO, MJD, GAIN, FILTER, MJDmid, NAXIS1,
                       NAXIS2, INST)) if verbose else '', end='')
  return(data, header, EXPTIME, MAGZERO, MJD, MJDmid,
         GAIN, NAXIS1, NAXIS2, WCS, FILTER, INST)


def keyValue(head_first, head_then, key):
  """First checks one header for a keyword; if fails, check second header."""
  try:
    value = head_first[key]
  except KeyError:
    value = head_then[key]
  return value


def inspectStars(data, catalogue, repfactor, **kwargs):
  """Run psfStarChooser, inspect stars, generate PSF and lookup table.
  """
  SExCatalogue = kwargs.pop('SExCatalogue', False)
  noVisualSelection = kwargs.pop('noVisualSelection', False)
  verbose = kwargs.pop('verbose', False)
  quickFit = kwargs.pop('quickFit', False)
  autoTrim = kwargs.pop('autoTrim', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  print(np.shape(data))
  try:
    starChooser = psfStarChooser.starChooser(data, catalogue['XWIN_IMAGE'],
                                             catalogue['YWIN_IMAGE'],
                                             catalogue['FLUX_AUTO'],
                                             catalogue['FLUXERR_AUTO'])
    (goodFits, goodMeds, goodSTDs
     ) = starChooser(30, 0,  # (box size, min SNR)
                     initAlpha=3., initBeta=3.,
                     repFact=repfactor, quickFit=quickFit,
                     noVisualSelection=noVisualSelection, autoTrim=autoTrim,
                     verbose=False, printStarInfo=True, saveFigure=True)
    print("\ngoodFits = {}\n".format(goodFits) if verbose else "", end='')
    print("\ngoodMeds = {}\n".format(goodMeds) if verbose else "", end='')
    print("\ngoodSTDs = {}\n".format(goodSTDs) if verbose else "", end='')
    goodPSF = psf.modelPSF(np.arange(61), np.arange(61), alpha=goodMeds[2],
                           beta=goodMeds[3], repFact=repfactor)
    fwhm = goodPSF.FWHM(fromMoffatProfile=True)  # the pure moffat FWHM
    print("fwhm = {}\n".format(fwhm) if verbose else "", end='')
    goodPSF.genLookupTable(data, goodFits[:, 4], goodFits[:, 5], verbose=False)
    goodPSF.genPSF()
    fwhm = goodPSF.FWHM()  # this is the FWHM with lookuptable included
    print("fwhm = {}\n".format(fwhm) if verbose else "", end='')
  except UnboundLocalError:
    print("Data error occurred!")
    raise
  if SExCatalogue:
    bestCatalogue = extractGoodStarCatalogue(catalogue,
                                             goodFits[:, 4], goodFits[:, 5])
    return bestCatalogue
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
  goodCatalogue = findSharedSExCatalogue([startCatalogue, trimCatalogue])
  print((len(goodCatalogue), len(goodCatalogue[0])) if verbose else "")
  return goodCatalogue


def findSharedSExCatalogue_old(catalogueArray, useIndex, **kwargs):
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
    #if len(np.where(master[ss][nkeys:] == 0)[0]) == 0:
    #  trimlist.append(ss)
    if not np.where(master[ss][nkeys:] == 0)[0]:
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


def findSharedSExCatalogue(catalogueArray, **kwargs):
  """Compare catalogues and create a master catalogue of only
  stars that are in all images
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  cat0 = catalogueArray[0]
  n0 = len(cat0)
  maskUse = np.ones(n0, dtype=bool)
  for catalogue in catalogueArray[1:]:
    maskNew = np.array([(cat0['XWIN_IMAGE'][i] in catalogue['XWIN_IMAGE']) &
                        (cat0['YWIN_IMAGE'][i] in catalogue['YWIN_IMAGE'])
                        for i in np.arange(n0)])
    maskUse = maskUse & maskNew
  sharedCatalogue = cat0[maskUse]
  print((len(sharedCatalogue), len(sharedCatalogue[0])) if verbose else "")
  """This should return a catalogue dictionary formatted in the same
  way as the original catalogue, allowing us to do anything with it that
  we could do with any other catalogue.
  """
  return sharedCatalogue


def findSharedPS1Catalogue(catalogueArray, **kwargs):
  """Compare catalogues and create a master catalogue of only
  stars that are in all images
  """
  verbose = kwargs.pop('verbose', False)
  if kwargs:
    raise TypeError('Unexpected **kwargs: %r' % kwargs)
  cat0 = catalogueArray[0]
  n0 = len(cat0)
  maskUse = np.ones(n0, dtype=bool)
  for catalogue in catalogueArray[1:]:
    maskNew = np.array([cat0['objName'][i] in catalogue['objName']
                        for i in np.arange(n0)])
    maskUse = maskUse & maskNew
  sharedCatalogue = cat0[maskUse]
  print((len(sharedCatalogue), len(sharedCatalogue[0])) if verbose else "")
  """This should return a catalogue dictionary formatted in the same
  way as the original catalogue, allowing us to do anything with it that
  we could do with any other catalogue.
  """
  return sharedCatalogue


def chooseCentroid(data, xt, yt, x0, y0, bg, goodPSF, NAXIS1, NAXIS2,
                   repfact=10, outfile=None, centroid=False, remove=False,
                   filename=None, tnotrack=False):
  ''' Choose between SExtractor position and predicted position.
  If desirable, use MCMC to fit the TSF to the object, thus centroiding on it.
  This is often NOT better than the SExtractor location, especially when the
  object is only barely trailed or when the sky has a gradient
  (near something bright).
  This fit is also used to remove the object from the image, later.
  fit takes time proportional to nWalkers*(2+nBurn+nStep).
  '''
  fitPars = None
  SExFoundIt = False if ((x0 == xt) & (y0 == yt)) else True
  centroid = True if not SExFoundIt else centroid  # if SEx no find TNO
  Data = (data[np.max([0, int(yt) - 200]):np.min([NAXIS2 - 1, int(yt) + 200]),
               np.max([0, int(xt) - 200]):np.min([NAXIS1 - 1, int(xt) + 200])]
          - bg)
  dtransy, dtransx = (- np.max([0, int(yt) - 200]) - 1,
                      - np.max([0, int(xt) - 200]) - 1)
  Zoom = (data[np.max([0, int(yt) - 15]):np.min([NAXIS2 - 1, int(yt) + 15]),
               np.max([0, int(xt) - 15]):np.min([NAXIS1 - 1, int(xt) + 15])]
          - bg)
  zy, zx = (int(yt) - np.max([0, int(yt) - 15]) - 1,
            int(xt) - np.max([0, int(xt) - 15]) - 1)
  m_obj = np.max(data[np.max([0, int(yt) - 5]):
                      np.min([NAXIS2 - 1, int(yt) + 5]),
                      np.max([0, int(xt) - 5]):
                      np.min([NAXIS1 - 1, int(xt) + 5])])
  xt0, yt0 = xt, yt
  while True:  # Breaks once a centroid has been selected.
    zscale = ZScaleInterval()
    (z1, z2) = zscale.get_limits(Zoom)
    normer = interval.ManualInterval(z1, z2)
    pyl.imshow(normer(Zoom), origin='lower')
    pyl.plot([zx + x0 - int(xt0)], [zy + y0 - int(yt0)], 'k*', ms=10)
    if SExFoundIt:
      pyl.plot([zx + xt0 - int(xt0)], [zy + yt0 - int(yt0)],
               'w+', ms=10, mew=2)
    if centroid or remove:
      print("Should I be doing this?")
      (xcent, ycent, fitPars, fitRange
       ) = runMCMCCentroid(goodPSF, Data, x0, y0, m_obj, 0, dtransx, dtransy,
                           repfact, tnotrack=tnotrack)
      print("\nfitPars = ", fitPars, "\nfitRange = ", fitRange, "\n")
      if outfile is not None:
        outfile.write("\nfitPars={}".format(fitPars) +
                      "\nfitRange={}".format(fitRange))
      pyl.plot([zx + xcent - int(xt0)],
               [zy + ycent - int(yt0)], 'rx', ms=10, mew=2)
      print("\n")
      print('These are your centroiding options in ' + filename
            if filename is not None else '')
      print("MCMCcentroid (green)  x,y = ", xcent, ycent)
      if SExFoundIt:
        print("SExtractor   (white)  x,y = ", xt, yt)
      print("Estimated    (black)  x,y = ", x0, y0)
      pyl.title(filename if filename is not None else '')
      pyl.show()
      if SExFoundIt:
        yn = input('Accept MCMC centroid (m or c), '
                   + 'SExtractor centroid (S), or estimate (e)? ')
      else:
        yn = input('Accept MCMC centroid (M or c), '
                   + 'or estimate (e)? ')
      if ('e' in yn) or ('E' in yn):  # if press e/E use estimate
        xt, yt = x0, y0
        break
      elif ('s' in yn) or ('S' in yn):
        break
      elif ('m' in yn) or ('M' in yn) or ('c' in yn) or ('C' in yn):
        xt, yt = xcent, ycent
        break
      else:
        if SExFoundIt:
          yn = 'S'  # else do nothing, use SExtractor co-ordinates.
        else:
          yn = 'M'  # else do nothing, use MCMC co-ordinates.
          xt, yt = xcent, ycent
        break
    else:  # if not previously centroided, check whether needed
      if (x0 == xt) & (y0 == yt):  # if SExtractor not find TNO, run centroid
        centroid = True
        SExFoundIt = False
      else:  # else pick between estimate, SExtractor and recentroiding
        print('These are your centroiding options in ' + filename
              if filename is not None else '')
        print("SExtractor   (white)  x,y = ", xt, yt)
        print("Estimated    (black)  x,y = ", x0, y0)
        pyl.title(filename if filename is not None else '')
        pyl.show()
        yn = input('Accept ' + 'SExtractor centroid (S), or estimate (e), '
                   + ' or recentroid using MCMC (m or c)? ')
        if ('e' in yn) or ('E' in yn):  # if press e/E use estimate
          xt, yt = x0, y0
          break
        elif ('m' in yn) or ('M' in yn) or ('c' in yn) or ('C' in yn):
          centroid = True
          print("Setting centroid={}, will re-centroid".format(centroid))
        else:
          yn = 'S'  # else do nothing, use SExtractor co-ordinates.
          break
  print("Coordinates chosen from this centroid: {}".format(yn))
  if outfile is not None:
    outfile.write("\nCoordinates chosen from this centroid: {}".format(yn))
  return xt, yt, yn, fitPars


def tst_fitting(goodPSF, Data, x_in, y_in, bg, repFact, tnotrack=False):
  '''Test the various ways of fitting PSF.'''
  if tnotrack:
    m_in = goodPSF.repFact ** 2 * np.sum(Data) / np.sum(goodPSF.fullPSF)
  else:
    m_in = goodPSF.repFact ** 2 * np.sum(Data) / np.sum(goodPSF.longPSF)

  fitter = MCMCfit.MCMCfitter(goodPSF, Data)
  fitter.fitWithModelPSF(x_in, y_in, m_in=m_in, bg=bg,
                         fitWidth=5, nWalkers=42, nBurn=10,
                         nStep=42, useLinePSF=True, verbose=False,
                         fitXY=True, fitM=True, fitXYM=True)
  (fitPars111, fitRange) = fitter.fitResults(0.67)

  fitter = MCMCfit.MCMCfitter(goodPSF, Data)
  fitter.fitWithModelPSF(x_in, y_in, m_in=m_in, bg=bg,
                         fitWidth=5, nWalkers=42, nBurn=10,
                         nStep=42, useLinePSF=True, verbose=False,
                         fitXY=False, fitM=True, fitXYM=True)
  (fitPars011, fitRange) = fitter.fitResults(0.67)

  fitter = MCMCfit.MCMCfitter(goodPSF, Data)
  fitter.fitWithModelPSF(x_in, y_in, m_in=m_in, bg=bg,
                         fitWidth=5, nWalkers=42, nBurn=10,
                         nStep=42, useLinePSF=True, verbose=False,
                         fitXY=True, fitM=False, fitXYM=True)
  (fitPars101, fitRange) = fitter.fitResults(0.67)

  fitter = MCMCfit.MCMCfitter(goodPSF, Data)
  fitter.fitWithModelPSF(x_in, y_in, m_in=m_in, bg=bg,
                         fitWidth=5, nWalkers=42, nBurn=10,
                         nStep=42, useLinePSF=True, verbose=False,
                         fitXY=True, fitM=True, fitXYM=False)
  (fitPars110, fitRange) = fitter.fitResults(0.67)

  fitter = MCMCfit.MCMCfitter(goodPSF, Data)
  fitter.fitWithModelPSF(x_in, y_in, m_in=m_in, bg=bg,
                         fitWidth=5, nWalkers=42, nBurn=10,
                         nStep=42, useLinePSF=True, verbose=False,
                         fitXY=False, fitM=False, fitXYM=True)
  (fitPars001, fitRange) = fitter.fitResults(0.67)

  fitter = MCMCfit.MCMCfitter(goodPSF, Data)
  fitter.fitWithModelPSF(x_in, y_in, m_in=m_in, bg=bg,
                         fitWidth=5, nWalkers=42, nBurn=10,
                         nStep=42, useLinePSF=True, verbose=False,
                         fitXY=False, fitM=True, fitXYM=False)
  (fitPars010, fitRange) = fitter.fitResults(0.67)

  fitter = MCMCfit.MCMCfitter(goodPSF, Data)
  fitter.fitWithModelPSF(x_in, y_in, m_in=m_in, bg=bg,
                         fitWidth=5, nWalkers=42, nBurn=10,
                         nStep=42, useLinePSF=True, verbose=False,
                         fitXY=True, fitM=False, fitXYM=False)
  (fitPars100, fitRange) = fitter.fitResults(0.67)

  fitter = MCMCfit.LSfitter(goodPSF, Data)
  fitPars000 = fitter.fitWithModelPSF(x_in, y_in, m_in=m_in,
                                      fitWidth=5, bg=bg,
                                      useLinePSF=True, verbose=False)

  print('111', fitPars111)
  print('011', fitPars011)
  print('101', fitPars101)
  print('110', fitPars110)
  print('001', fitPars001)
  print('010', fitPars010)
  print('100', fitPars100)
  print('000', fitPars000)
  print(x_in, y_in, m_in)


def removeTSF(data, xt, yt, bg, goodPSF, NAXIS1, NAXIS2, header, inputName,
              outfile=None, repfact=10, remove=True, verbose=False,
              fitPars=None, tnotrack=False):
  '''Remove a TSF.
  If remove=False, will not remove, just saves postage-stamp around xt, yt.'''
  Data = (data[np.max([0, int(yt) - 200]):np.min([NAXIS2 - 1, int(yt) + 200]),
               np.max([0, int(xt) - 200]):np.min([NAXIS1 - 1, int(xt) + 200])]
          - bg)
  dtransy = - np.max([0, int(yt) - 200]) - 1
  dtransx = - np.max([0, int(xt) - 200]) - 1
  m_obj = np.max(data[np.max([0, int(yt) - 5]):
                      np.min([NAXIS2 - 1, int(yt) + 5]),
                      np.max([0, int(xt) - 5]):
                      np.min([NAXIS1 - 1, int(xt) + 5])])
  #if (fitPars is None) or remove:
  if (fitPars is None) and remove:
    print("Should I be doing this?")
    fitter = MCMCfit.MCMCfitter(goodPSF, Data)
    fitter.fitWithModelPSF(dtransx + xt, dtransy + yt,
                           m_in=m_obj / repfact ** 2.,
                           fitWidth=10, nWalkers=20,
                           nBurn=20, nStep=20, bg=0, useLinePSF=not(tnotrack),
                           verbose=False, useErrorMap=False)
    (fitPars, fitRange) = fitter.fitResults(0.67)
    print("\nfitPars = ", fitPars, "\n")
    print("\nfitRange = ", fitRange, "\n")
    if outfile is not None:
      outfile.write("\nfitPars={}".format(fitPars))
      outfile.write("\nfitRange={}".format(fitRange))
  if fitPars is not None:
    removed = goodPSF.remove(fitPars[0], fitPars[1], fitPars[2],
                             Data, useLinePSF=not(tnotrack))
    zscale = ZScaleInterval()
    (z1, z2) = zscale.get_limits(removed)
    normer = interval.ManualInterval(z1, z2)
    modelImage = goodPSF.plant(fitPars[0], fitPars[1], fitPars[2], Data,
                               addNoise=False, useLinePSF=not(tnotrack),
                               returnModel=True)
    if verbose:
      pyl.imshow(normer(goodPSF.lookupTable), origin='lower')
      pyl.show()
      pyl.imshow(normer(modelImage), origin='lower')
      pyl.show()
      pyl.imshow(normer(Data), origin='lower')
      pyl.show()
      pyl.imshow(normer(removed), origin='lower')
      pyl.show()
    hdu = pyf.PrimaryHDU(modelImage, header=header)
    list = pyf.HDUList([hdu])
    list.writeto(inputName + '_modelImage.fits', overwrite=True)
    hdu = pyf.PrimaryHDU(removed, header=header)
    list = pyf.HDUList([hdu])
    list.writeto(inputName + '_removed.fits', overwrite=True)
  else:
    zscale = ZScaleInterval()
    (z1, z2) = zscale.get_limits(Data)
    normer = interval.ManualInterval(z1, z2)
    if verbose:
      pyl.imshow(normer(goodPSF.lookupTable), origin='lower')
      pyl.show()
      pyl.imshow(normer(Data), origin='lower')
      pyl.show()
  hdu = pyf.PrimaryHDU(goodPSF.lookupTable, header=header)
  list = pyf.HDUList([hdu])
  list.writeto(inputName + '_lookupTable.fits', overwrite=True)
  hdu = pyf.PrimaryHDU(Data, header=header)
  list = pyf.HDUList([hdu])
  list.writeto(inputName + '_Data.fits', overwrite=True)
  return


def calcCalib(cat, FILTER, apStr, telescope):
  """Calculate the zero-point offset and uncertainty"""
  magKeyName = FILTER + 'MagTrippy' + apStr
  if telescope == 'CFHT':
    magCalibArray = (cat[FILTER + 'MeanPSFMag_CFHT']
                     - cat[magKeyName])
    dmagCalibArray = (cat[FILTER + 'MeanPSFMagErr'] ** 2
                      + cat['d' + magKeyName] ** 2) ** 0.5
  if telescope == 'LBT':
    magCalibArray = (cat[FILTER + 'MeanPSFMag_CFHT']
                     - cat[magKeyName])
    dmagCalibArray = (cat[FILTER + 'MeanPSFMagErr'] ** 2
                      + cat['d' + magKeyName] ** 2) ** 0.5
  if telescope == 'Gemini':
    magCalibArray = (cat[FILTER[0] + 'MeanPSFMag_Gemini']
                     - cat[magKeyName])
    dmagCalibArray = (cat[FILTER[0] + 'MeanPSFMagErr'] ** 2
                      + cat['d' + magKeyName] ** 2) ** 0.5
  # First use lazy method to identify outliers:
  magCalibration = np.nanmean(magCalibArray)
  dmagCalibration = np.std(magCalibArray)
  # Calculate it again with weighted average and outlier rejection:
  sigmaclip = np.abs(magCalibArray - magCalibration) < 3 * dmagCalibration
  (magCalibration, sumOfWeights
   ) = np.average(magCalibArray[sigmaclip], returned=True,
                  weights=1. / dmagCalibArray[sigmaclip] ** 2)
  dmagCalibration = 1. / sumOfWeights ** 0.5
  return magCalibration, dmagCalibration, sigmaclip

# End of file.
# Nothing to see here.
