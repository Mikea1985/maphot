#!/usr/bin/python
"""maphot is a wrapper for easily running trippy on a bunch of images
of a single object on a single night.
Usage: (-h gives this line as well)
maphot.py -c <coofile> -f <fitsfile> -v False -. True -o False -r True
          -a 0.7 -t False -e 0
Defaults are:
inputFile = 'a100.fits'  # Change with '-f <filename>' flag
coofile = 'coords.in'  # Change with '-c <coofile>' flag
verbose = False  # Change with '-v True' or '--verbose True'
centroid = True  # Change with '-. False' or  --centroid False'
overrideSEx = False  # Change with '-o True' or '--override True'
remove = True  # Change with '-r False' or '--remove False'
aprad = 0.7  # Change with '-a 1.5' or '--aprad 1.5'
extno = 0  # Change with '-e 1' or '--extno 1'
tnotrack = False  # Change with '-t True'
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
import dill
import mp_ephem
from six.moves import zip
import numpy as np
#from astropy.io import fits
from trippy import psf, pill
import best
from maphot_functions import (getArguments, getObservations, coordRateAngle,
                              getSExCatalog, predicted2catalog,
                              saveTNOMag, saveStarMag, saveTNOMag2,
                              getDataHeader, PS1_vs_SEx,
                              PS1_to_Gemini, PS1_to_CFHT,  # CFHT_to_PS1,
                              inspectStars, chooseCentroid, removeTSF,
                              addTrippyToCat, calcCalib,
                              PS1_to_LBT, extractGoodStarCatalogue, LBT_not_to)
from __version__ import __version__
from pix2world import pix2MPC

__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')
print("You are using maphot version: ", __version__)

###############################################################################

useage = 'maphot -c <coordsfile> -f <fitsfile> -v False '\
         + '-. False -o False -r False -a 0.7 -t False'
(inputFile, coordsfile, verbose, centroid, overrideSEx, remove, aprad,
 tnotrack, repfact, pxscale, roundAperRad, SExParFile, extno, ignoreWarnings
 ) = getArguments(sys.argv)
#Ignore all Python warnings.
#This is generally a terrible idea, and should be turned off for de-bugging.
if ignoreWarnings:
  warnings.filterwarnings("ignore")

#roundAperRad = aprad
print(aprad, roundAperRad)
starAperRad = roundAperRad
tnoAperRad = roundAperRad

print("ifile =", inputFile, ", coords =", coordsfile, ", verbose =", verbose,
      ", centroid =", centroid, ", overrideSEx =", overrideSEx,
      ", remove =", remove, ", aprad =", aprad, ", tracking TNO ", tnotrack)
if verbose:
  print(np.array([centroid]).dtype, np.array([remove]).dtype)
  if centroid or remove:
    print("Will run MCMC centroiding")

# Read in the image and get the data, header and keywords needed.
(data, header, EXPTIME, MAGZERO, MJD, MJDm, GAIN, NAXIS1, NAXIS2, WCS, FILTER,
 INST) = getDataHeader(inputFile, extno=extno)

# Which telescope is this?
teles = 'LBT'
if INST == 'MegaPrime':
  teles = 'CFHT'
if INST == 'GMOS-N':
  teles = 'Gemini'
print("Using telescope: " + teles)

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
# photometry parameters based on sidereal/non-sidereal tracking
if tnotrack:
  ltno = 0.0
  atno = 0.0
  lstar = (EXPTIME / 3600.) * rate / pxscale
  astar = angle
else:
  lstar = 0.0
  astar = 0.0
  ltno = (EXPTIME / 3600.) * rate / pxscale
  atno = angle

#Get the Source Extractor catalogue
if SExParFile is None:
  SEx_params = np.array([2.0, 2.0, 27.8, 10.0, 2.0, 2.0])
else:
  SEx_params = np.genfromtxt(SExParFile)
fullSExCat = getSExCatalog(inputFile, SEx_params, extno=extno)

#Find the SourceExtractor source nearest to the predicted location.
TNOSEx, cShift = predicted2catalog(fullSExCat, TNOpred)
xSEx, ySEx = TNOSEx[:, 0]
centroidShift = cShift[0]
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
  print('Success! Unpickled the best catalogue.')
  outfile.write('\nSuccess! Unpickled the best catalogue.')
except IOError:
  print('Uh oh! Unpickling unsuccesful. Does ' + bestCatName + ' exist?')
  print('If not, run best.best([fitsList]).')
  raise IOError(bestCatName + ' missing. Run best.best')
# Match phot stars to PS1 catalog
catalog_psf = PS1_vs_SEx(bestCat, fullSExCat, maxDist=1.0, appendSEx=True)

# Restore PSF if exist, otherwise build it.
try:
  goodStarFile = open(inputName + '_goodStars.pickle', 'rb')
  (goodFits, goodMeds, goodSTDs, goodPSF, starAperCorr, tnoAperCorr
   ) = dill.load(goodStarFile)
  goodStarFile.close()
  print("PSF restored from file.")
  outfile.write("\nPSF restored from file\n")
  fwhm = goodPSF.FWHM()
  print("fwhm = ", fwhm, ' restored')
  outfile.write("\nfwhm = {}\n".format(fwhm))
except IOError:
  print("Could not restore PSF (Normal unless previously saved)")
  print("Making new one.")
  outfile.write("\nDid not restore PSF from file\n")
  (goodFits, goodMeds, goodSTDs, goodPSF, fwhm
   ) = inspectStars(data, catalog_psf, repfact, verbose=True)
  fwhm = goodPSF.FWHM()
  print(" fwhm = ", fwhm)
  outfile.write("\ngoodFits={}".format(goodFits))
  outfile.write("\ngoodMeds={}".format(goodMeds))
  outfile.write("\ngoodSTDs={}".format(goodSTDs))
  outfile.write("\n goodPSF = {}\n".format(goodPSF))
  outfile.write("\n fwhm = {}\n".format(fwhm))
  goodPSF.line(rate, angle, EXPTIME / 3600., pixScale=pxscale,
               useLookupTable=True)
  goodPSF.computeRoundAperCorrFromPSF(psf.extent(0.1 * fwhm, 6 * fwhm, 100),
                                      display=False,
                                      displayAperture=False,
                                      useLookupTable=True)
  goodPSF.computeLineAperCorrFromTSF(psf.extent(0.1 * fwhm, 6 * fwhm, 100),
                                     l=(EXPTIME / 3600.) * rate / pxscale,
                                     a=angle, display=False,
                                     displayAperture=False)
  starAperCorr = (goodPSF.lineAperCorr(starAperRad * fwhm) if tnotrack
                  else goodPSF.roundAperCorr(starAperRad * fwhm))
  tnoAperCorr = (goodPSF.roundAperCorr(tnoAperRad * fwhm) if tnotrack
                 else goodPSF.lineAperCorr(tnoAperRad * fwhm))
  goodPSF.psfStore(inputName + '_psf.fits')
  fwhm = goodPSF.FWHM()
  print("  fwhm = ", fwhm)
  outfile.write("\nfwhm = {}\n".format(fwhm))
  goodStarFile = open(inputName + '_goodStars.pickle', 'wb')
  dill.dump([goodFits, goodMeds, goodSTDs, goodPSF, starAperCorr, tnoAperCorr],
            goodStarFile, dill.HIGHEST_PROTOCOL)
  goodStarFile.close()
except UnboundLocalError:
  print("Data error occurred!")
  outfile.write("\nData error occured!\n")
  raise

dAperCorr = 0.01
print("tnoAperCorr, starAperCorr = ", tnoAperCorr, starAperCorr, "\n")
outfile.write("\ntnoAperCorr,starAperCorr={},{}".format(tnoAperCorr,
                                                        starAperCorr))

#print(goodStars)
catalog_phot = extractGoodStarCatalogue(catalog_psf, goodFits[:, 4],
                                        goodFits[:, 5])
#catalog_phot = catalog_psf

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
  starPhot(xcat, ycat, radius=fwhm * starAperRad, l=lstar, a=astar,
           exptime=EXPTIME,
           zpt=MAGZERO, skyRadius=4 * fwhm, width=5 * fwhm,
           enableBGSelection=verbose, display=verbose, backupMode="smart",
           trimBGHighPix=3., zscale=False)
  starPhot.SNR(gain=GAIN, useBGstd=True)
  starPhotInst = starPhot.magnitude - starAperCorr
  dstarPhotInst = (starPhot.dmagnitude ** 2 + dAperCorr ** 2) ** 0.5
  print("{0:13.8f} {1:13.8f} {2:} {3:}".format(
        xcat, ycat, starPhotInst, dstarPhotInst))
  outfile.write("\n{0:13.8f} {1:13.8f} {2:} {3:}".format(
                xcat, ycat, starPhotInst, dstarPhotInst))
  magStars.append(starPhotInst)
  dmagStars.append(dstarPhotInst)
  fluxStars.append(starPhot.sourceFlux)
  SNRStars.append(starPhot.snr)
  bgStars.append(starPhot.bg)

(xUse, yUse, centroidUsed, fitPars
 ) = chooseCentroid(data, xUse, yUse, xPred, yPred, np.median(bgStars),
                    goodPSF, NAXIS1, NAXIS2, outfile=outfile, repfact=repfact,
                    centroid=centroid, remove=remove, filename=inputFile)

print('\nPhotometry of moving object')
outfile.write("\nPhotometry of moving object\n")
TNOPhot = pill.pillPhot(data, repFact=repfact)
# Make sure to use IRAF coordinates not numpy/sextractor coordinates!

TNOPhot(xUse, yUse, radius=fwhm * tnoAperRad,
        l=ltno, a=atno, skyRadius=4 * fwhm, width=6 * fwhm,
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

TNOPhotInst = TNOPhot.magnitude - tnoAperCorr
dTNOPhotInst = (TNOPhot.dmagnitude ** 2 + dAperCorr ** 2) ** 0.5

print("\nAlmost final (non-calibrated) results!")
print("#{0:12} {1:13} {2:13} {3:13} {4:13}".format(
      '   x ', '    y ', ' magnitude ', '  dmagnitude ', ' magzero '))
print("{0:13.8f} {1:13.8f} {2:} {3:} {4:13.10f}\n".format(
      xUse, yUse, TNOPhotInst, dTNOPhotInst, MAGZERO))
outfile.write("\nFINAL (non-calibrated) RESULT!")
outfile.write("\n#{0:12} {1:13} {2:13} {3:13} {4:13}\n".format(
              '   x ', '    y ', ' magnitude ', '  dmagnitude ', ' magzero '))
outfile.write("{0:13.8f} {1:13.8f} {2:} {3:} {4:13.10f}\n".format(
              xUse, yUse, TNOPhotInst, dTNOPhotInst, MAGZERO))

# Convert star catalog's PS1 magnitudes to Instrument magnitudes
catphotPS1 = PS1_to_LBT(catalog_phot) if teles == 'LBT' \
             else PS1_to_Gemini(catalog_phot) if teles == 'Gemini' \
             else PS1_to_CFHT(catalog_phot)

# Add trippy photometry to the star catalog
finalCat = catphotPS1

magCalibration = np.zeros(len(starAperRad))
dmagCalibration = magCalibration.copy()
sigmaclip = []
for i, starAR in enumerate(starAperRad):
  apStr = str(starAR)
  finalCat = addTrippyToCat(finalCat, np.array(magStars)[:, i],
                            np.array(dmagStars)[:, i],
                            np.array(fluxStars)[:, i],
                            np.array(SNRStars)[:, i],
                            bgStars, FILTER, apStr)
# Calculate magnitude calibration factor
  (magCalibration[i], dmagCalibration[i], sc
   ) = calcCalib(finalCat, FILTER, apStr, teles)
  sigmaclip.append(sc)
sigmaclip = np.array(sigmaclip)

# Correct the TNO magnitude and zero point
finalTNOphotInst = (TNOPhotInst + magCalibration,
                    (dTNOPhotInst ** 2 + dmagCalibration ** 2) ** 0.5)
zptGood = MAGZERO + magCalibration
#finalTNOphotPS1 = CFHT_to_PS1(finalTNOphotInst[0], finalTNOphotInst[1],FILTER)
if teles == 'CFHT':
  finalTNOphotPS1 = LBT_not_to(finalTNOphotInst[0],
                               finalTNOphotInst[1], FILTER)
if teles == 'LBT':
  finalTNOphotPS1 = LBT_not_to(finalTNOphotInst[0],
                               finalTNOphotInst[1], FILTER)
if teles == 'Gemini':
  finalTNOphotPS1 = LBT_not_to(finalTNOphotInst[0],
                               finalTNOphotInst[1], FILTER)
#FIX THIS!!!! convert back

print("\nFINAL (calibrated) RESULT!")
print("#{0:12} {1:13} {2:13} {3:13} {4:13}".format(
      '   x ', '    y ', ' magnitude ', '  dmagnitude ', ' magzero '))
print("{0:13.8f} {1:13.8f} {2:} {3:} {4:}\n".format(
      xUse, yUse, finalTNOphotPS1[0], finalTNOphotPS1[1], zptGood))
outfile.write("\nFINAL (calibrated) RESULT!")
outfile.write("\n#{0:12} {1:13} {2:13} {3:13} {4:13}\n".format(
              '   x ', '    y ', ' magnitude ', '  dmagnitude ', ' magzero '))
outfile.write("{0:13.8f} {1:13.8f} {2:} {3:} {4:}\n".format(
              xUse, yUse, finalTNOphotPS1[0], finalTNOphotPS1[1], zptGood))

TNOCoords = WCS.all_pix2world(xUse, yUse, 1)
#Save TNO magnitudes neatly.
timeNow = datetime.now().strftime('%Y-%m-%d/%H:%M:%S')
saveTNOMag2(inputFile, coordsfile, MJDm, TNOCoords, xUse, yUse,
            FILTER, fwhm, finalTNOphotPS1, timeNow, __version__, extno=extno)
saveTNOMag(inputFile, coordsfile, MJD, MJDm, TNOCoords, xUse, yUse,
           MAGZERO, FILTER, fwhm, tnoAperRad, TNOPhot, tnoAperCorr,
           magCalibration, dmagCalibration, finalTNOphotInst, zptGood,
           finalTNOphotPS1, timeNow, np.array(TNOPhot.bgSamplingRegion),
           __version__, extno=extno)
saveStarMag(inputFile, finalCat, timeNow, __version__,
            MJD, sigmaclip, extno=extno)

# You could stop here.
# However, to confirm that things are working well,
# let's generate the trailed PSF and subtract the object out of the image.
#if centroid and (('e' in centroidUsed) or ('E' in centroidUsed) or
#                 ('s' in centroidUsed) or ('S' in centroidUsed)):
#  remove = True

removeTSF(data, xUse, yUse, TNOPhot.bg, goodPSF, NAXIS1, NAXIS2, header,
          inputName, outfile=outfile, repfact=repfact, remove=remove,
          fitPars=fitPars)

#Run function to save photometry in MPC format
pix2MPC(WCS, EXPTIME, MJD, np.nanmean(finalTNOphotPS1[0]),
        xUse, yUse, FILTER, extno)

print('Done with ' + inputFile + '!')
outfile.close()
# End of file.
# Nothing to see here.
