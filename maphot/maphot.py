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
import sys
from datetime import datetime
import warnings
from six.moves import zip
import numpy as np
import mp_ephem
#from astropy.io import fits
from trippy import psf, pill
import best
from maphot_functions import (getArguments, getObservations, coordRateAngle,
                              getSExCatalog, predicted2catalog,
                              saveTNOMag, saveStarMag,
                              getDataHeader, addPhotToCatalog, PS1_vs_SEx,
                              PS1_to_CFHT, CFHT_to_PS1, inspectStars,
                              chooseCentroid, removeTSF)
from __version__ import __version__
from pix2world import pix2world

__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')
print("You are using maphot version: ", __version__)

###############################################################################

useage = 'maphot -c <coordsfile> -f <fitsfile> -v False '\
         + '-. False -o False -r False -a 0.7'
(inputFile, coordsfile, verbose, centroid, overrideSEx, remove,
 aprad, repfact, pxscale, roundAperRad, SExParFile, extno, ignoreWarnings
 ) = getArguments(sys.argv)
#Ignore all Python warnings.
#This is generally a terrible idea, and should be turned off for de-bugging.
if ignoreWarnings:
  warnings.filterwarnings("ignore")

print("ifile =", inputFile, ", coords =", coordsfile, ", verbose =", verbose,
      ", centroid =", centroid, ", overrideSEx =", overrideSEx,
      ", remove =", remove, ", aprad =", aprad)
if verbose:
  print(np.array([centroid]).dtype, np.array([remove]).dtype)
  if centroid or remove:
    print("Will run MCMC centroiding")

# Read in the image and get the data, header and keywords needed.
(data, header, EXPTIME, MAGZERO, MJD, MJDm, GAIN, NAXIS1, NAXIS2, WCS, FILTER
 ) = getDataHeader(inputFile, extno=extno)

# Set up an output file that has all sorts of information.
# Preferably, whenever something is printed to screen, save it here too.
inputName = inputFile.replace('.fits', '{0:02.0f}'.format(extno))
outfile = open(inputName + '.trippy', 'w')
print("############################")
if extno is None:
  print("Working on ", inputFile)
else:
  print("Working on {}[{}]".format(inputFile, extno))
print("############################")
if extno is None:
  outfile.write("\nWorking on {}.\n".format(inputFile))
else:
  outfile.write("\nWorking on {}[{}].\n".format(inputFile, extno))
print("\nMJDm = ", MJDm)
outfile.write("\nMJDm = {}\n".format(MJDm))

#Get the object coordinates and rates of motion
with open(coordsfile) as han:
    mpc = han.readlines()
observations = getObservations(mpc)
TNOorbit = mp_ephem.BKOrbit(observations)
TNOpred, rate, angle = coordRateAngle(TNOorbit, MJDm, WCS)
print('TNO predicted to be at {},\nmoving at '.format(TNOpred) +
      '{} pix/hr inclined {} deg (to x+).'.format(rate, angle))
outfile.write('\nTNO predicted to be at {},\nmoving at '.format(TNOpred) +
              '{} pix/hr inclined {} deg (to x+).'.format(rate, angle))
xPred, yPred = TNOpred

#Get the Source Extractor catalogue
if SExParFile is None:
  SEx_params = np.array([2.0, 2.0, 27.8, 10.0, 2.0, 2.0])
else:
  SEx_params = np.genfromtxt(SExParFile)
fullSExCat = getSExCatalog(inputFile, SEx_params, extno=extno)

#Find the SourceExtractor source nearest to the predicted location.
TNOSEx, centroidShift = predicted2catalog(fullSExCat, TNOpred)
xSEx, ySEx = TNOSEx
#If using the Source Extractor centroid is undesirable, set overrideSEx=True
#Otherwise, the source nearest the predicted TNO location is used.
if (centroidShift > 15) | overrideSEx:
  print("SourceExtractor location no good (maybe didn't find TNO?)")
  print('Will use predicted location instead.')
  outfile.write("\nSourceExtractor location no good (maybe didn't find TNO?)")
  outfile.write('\nWill use predicted location instead.')
  centroid = True
  xUse, yUse = xPred, yPred  # Use predicted location.
else:
  xUse, yUse = xSEx, ySEx  # Use SExtractor location
print("xUse, yUse = ", xUse, yUse)
outfile.write("\nxUse, yUse = {}, {}\n".format(xUse, yUse))

# Read in catalogue of good stars
bestCatName = ('best.cat' if extno is None
               else 'best{0:02.0f}.cat'.format(extno))
try:
  bestCat = best.unpickleCatalogue(bestCatName)
  print('Success!')
except IOError:
  print('Uh oh! Unpickling unsuccesful. Does ' + bestCatName + ' exist?')
  print('If not, run best.best([fitsList]).')
  #best.best([glob.glob('*.fits')], repfact)
  #bestCat = best.unpickleCatalogue(bestCatName)
  raise IOError(bestCatName + ' missing. Run best.best')
# Match phot stars to PS1 catalog
catalog_psf = PS1_vs_SEx(bestCat, fullSExCat, maxDist=1.0, appendSEx=True)
catalog_phot = catalog_psf

# Restore PSF if exist, otherwise build it.
try:
  goodPSF = psf.modelPSF(restore=inputName + '_psf.fits')
  fwhm = goodPSF.FWHM()
  print("PSF restored from file.")
  outfile.write("\nPSF restored from file\n")
#  goodPSF.fitted=False
  print("fwhm = ", fwhm)
  outfile.write("\nfwhm = {}\n".format(fwhm))
except IOError:
  print("Could not restore PSF (Normal unless previously saved)")
  print("Making new one.")
  outfile.write("\nDid not restore PSF from file\n")
  (goodFits, goodMeds, goodSTDs, goodPSF, fwhm
   ) = inspectStars(data, catalog_psf, repfact, verbose=True)
  outfile.write("\ngoodFits={}".format(goodFits))
  outfile.write("\ngoodMeds={}".format(goodMeds))
  outfile.write("\ngoodSTDs={}".format(goodSTDs))
  outfile.write("\n fwhm = {}\n".format(goodPSF))
  outfile.write("\n fwhm = {}\n".format(fwhm))
except UnboundLocalError:
  print("Data error occurred!")
  outfile.write("\nData error occured!\n")
  raise

goodPSF.line(rate, angle, EXPTIME / 3600., pixScale=pxscale,
             useLookupTable=True)
goodPSF.computeRoundAperCorrFromPSF(psf.extent(0.7 * fwhm, 4 * fwhm, 100),
                                    display=False,
                                    displayAperture=False,
                                    useLookupTable=True)
roundAperCorr = goodPSF.roundAperCorr(roundAperRad * fwhm)
goodPSF.computeLineAperCorrFromTSF(psf.extent(0.1 * fwhm, 4 * fwhm, 100),
                                   l=(EXPTIME / 3600.) * rate / pxscale,
                                   a=angle, display=False,
                                   displayAperture=False)
goodPSF.psfStore(inputName + '_psf.fits')

# Do photometry for the trimmed catalog stars.
# This will be used to find a set of non-variable stars, in order to
# subtract fluctuations due to seeing, airmass, etc.
bgStars = []
magStars = []
dmagStars = []
fluxStars = []
SNRStars = []
print('Photometry of catalog stars')
outfile.write("\n# Photometry of catalog stars\n")
outfile.write("\n#   x       y   magnitude  dmagnitude")
for xcat, ycat in np.array(list(zip(catalog_phot['XWIN_IMAGE'],
                                    catalog_phot['YWIN_IMAGE']))):
  starPhot = pill.pillPhot(data, repFact=repfact)
  starPhot(xcat, ycat, radius=fwhm * roundAperRad, l=0.0, a=0.0,
           exptime=EXPTIME,
           #zpt=26.0, skyRadius=4 * fwhm, width=30.,
           zpt=MAGZERO, skyRadius=4 * fwhm, width=30.,
           enableBGSelection=verbose, display=verbose, backupMode="smart",
           trimBGHighPix=3., zscale=False)
  starPhot.SNR(gain=GAIN, useBGstd=True)
  print("{0:13.8f} {1:13.8f} {2:13.10f} {3:13.10f}".format(
        xcat, ycat, starPhot.magnitude - roundAperCorr, starPhot.dmagnitude))
  outfile.write("\n{0:13.8f} {1:13.8f} {2:13.10f} {3:13.10f}".format(
                xcat, ycat, starPhot.magnitude - roundAperCorr,
                starPhot.dmagnitude))
  magStars.append(starPhot.magnitude - roundAperCorr)
  dmagStars.append(starPhot.dmagnitude)
  fluxStars.append(starPhot.sourceFlux)
  SNRStars.append(starPhot.snr)
  bgStars.append(starPhot.bg)

(xUse, yUse, centroidUsed
 ) = chooseCentroid(data, xUse, yUse, xPred, yPred, np.median(bgStars),
                    goodPSF, NAXIS1, NAXIS2, outfile=outfile, repfact=repfact,
                    centroid=centroid, remove=remove)

print('\nPhotometry of moving object')
outfile.write("\nPhotometry of moving object\n")
TNOPhot = pill.pillPhot(data, repFact=repfact)
# Make sure to use IRAF coordinates not numpy/sextractor coordinates!
if aprad > 0:
  bestap = np.arange(aprad, aprad + 1)[0]  # stupid but wouldn't work otherwise
else:  # Automatically identify best aperture.
  apertures = np.arange(0.7, 2.0, 0.1)
  linedmag = np.zeros(len(apertures))
  for i, ap in enumerate(apertures):
    TNOPhot(xUse, yUse, radius=fwhm * ap, l=(EXPTIME / 3600.) * rate / pxscale,
            a=angle, skyRadius=4 * fwhm, width=6 * fwhm,
            #zpt=26.0, exptime=EXPTIME, enableBGSelection=False, display=False,
            zpt=MAGZERO, exptime=EXPTIME, enableBGSelection=False,
            display=False, backupMode="smart", trimBGHighPix=3., zscale=False)
    TNOPhot.SNR(gain=GAIN, verbose=False, useBGstd=True)
    linedmag[i] = TNOPhot.dmagnitude
  bestap = apertures[np.argmin(linedmag)]
lineAperRad = bestap
print("Aperture used= ", bestap)
outfile.write("\nBest aperture = {}".format(bestap))
lineAperCorr = goodPSF.lineAperCorr(lineAperRad * fwhm)
print("lineAperCorr, roundAperCorr = ", lineAperCorr, roundAperCorr, "\n")
outfile.write("\nlineAperCorr,roundAperCorr={},{}".format(lineAperCorr,
                                                          roundAperCorr))
TNOPhot(xUse, yUse, radius=fwhm * lineAperRad,
        l=(EXPTIME / 3600.) * rate / pxscale,
        a=angle, skyRadius=4 * fwhm, width=6 * fwhm,
        #zpt=26.0, exptime=EXPTIME, enableBGSelection=True, display=True,
        zpt=MAGZERO, exptime=EXPTIME, enableBGSelection=True, display=True,
        backupMode="smart", trimBGHighPix=3., zscale=False)
TNOPhot.SNR(gain=GAIN, verbose=True, useBGstd=True)

# Print those values
print("TNOPhot.magnitude = ", TNOPhot.magnitude)
print("TNOPhot.dmagnitude = ", TNOPhot.dmagnitude)
print("TNOPhot.sourceFlux = ", TNOPhot.sourceFlux)
print("TNOPhot.snr = ", TNOPhot.snr)
print("TNOPhot.bg = ", TNOPhot.bg)
outfile.write("\nTNOPhot.magnitude={}".format(TNOPhot.magnitude))
outfile.write("\nTNOPhot.dmagnitude={}".format(TNOPhot.dmagnitude))
outfile.write("\nTNOPhot.sourceFlux={}".format(TNOPhot.sourceFlux))
outfile.write("\nTNOPhot.snr={}".format(TNOPhot.snr))
outfile.write("\nTNOPhot.bg={}".format(TNOPhot.bg))

print("\nFINAL (non-calibrated) RESULT!")
print("#{0:12} {1:13} {2:13} {3:13} {4:13}".format(
      '   x ', '    y ', ' magnitude ', '  dmagnitude ', ' magzero '))
print("{0:13.8f} {1:13.8f} {2:13.10f} {3:13.10f} {4:13.10f}\n".format(
      xUse, yUse, TNOPhot.magnitude - lineAperCorr,
      TNOPhot.dmagnitude, MAGZERO))
outfile.write("\nFINAL (non-calibrated) RESULT!")
outfile.write("\n#{0:12} {1:13} {2:13} {3:13} {4:13}\n".format(
              '   x ', '    y ', ' magnitude ', '  dmagnitude ', ' magzero '))
outfile.write("{0:13.8f} {1:13.8f} {2:13.10f} {3:13.10f} {4:13.10f}\n".format(
              xUse, yUse, TNOPhot.magnitude - lineAperCorr,
              TNOPhot.dmagnitude, MAGZERO))

# Add photometry to the star catalog
magKeyName = FILTER + 'MagTrippy' + str(bestap)
PS1PhotCat = addPhotToCatalog(catalog_phot['XWIN_IMAGE'],
                              catalog_phot['YWIN_IMAGE'], catalog_phot,
                              {magKeyName: np.array(magStars),
                               'd' + magKeyName: np.array(dmagStars),
                               'TrippySourceFlux': np.array(fluxStars),
                               'TrippySNR': np.array(SNRStars),
                               'TrippyBG': np.array(bgStars)})
# Convert star catalog's PS1 magnitudes to CFHT magnitudes
finalCat = PS1_to_CFHT(PS1PhotCat)
# Calculate magnitude calibration factor
magCalibArray = (finalCat[FILTER + 'MeanPSFMag_CFHT']
                 - finalCat[magKeyName])
magCalibration = np.median(magCalibArray)
dmagCalibration = np.std(magCalibArray)

# Correct the TNO magnitude and zero point
finalTNOphotCFHT = (TNOPhot.magnitude - lineAperCorr + magCalibration,
                    (TNOPhot.dmagnitude ** 2
                     + dmagCalibration ** 2) ** 0.5)
zptGood = MAGZERO + magCalibration
finalTNOphotPS1 = CFHT_to_PS1(finalTNOphotCFHT[0], finalTNOphotCFHT[1], FILTER)

TNOCoords = WCS.all_pix2world(xUse, yUse, 1)
#Save TNO magnitudes neatly.
timeNow = datetime.now().strftime('%Y-%m-%d/%H:%M:%S')
saveTNOMag(inputFile, coordsfile, MJD, MJDm, TNOCoords, xUse, yUse,
           MAGZERO, FILTER, fwhm, bestap, TNOPhot,
           magCalibration, dmagCalibration, finalTNOphotCFHT, zptGood,
           finalTNOphotPS1, timeNow, np.array(TNOPhot.bgSamplingRegion),
           __version__, extno=extno)
saveStarMag(inputFile, finalCat, timeNow, __version__, MJD, extno=extno)

# You could stop here.
# However, to confirm that things are working well,
# let's generate the trailed PSF and subtract the object out of the image.
#if centroid and (('e' in centroidUsed) or ('E' in centroidUsed) or
#                 ('s' in centroidUsed) or ('S' in centroidUsed)):
#  remove = True
removeTSF(data, xUse, yUse, TNOPhot.bg, goodPSF, NAXIS1, NAXIS2, header,
          inputName + '{0:02.0f}'.format(extno), outfile=outfile,
          repfact=repfact, remove=remove)

#Run function to save photometry in MPC format
pix2world(inputFile, EXPTIME, MJD, finalTNOphotPS1[0], xUse, yUse, FILTER, extno)

print('Done with ' + inputFile + '!')
outfile.close()
# End of file.
# Nothing to see here.
