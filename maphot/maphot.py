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
from six.moves import input, zip
import numpy as np
import pylab as pyl
import mp_ephem
import astropy.io.fits as pyf
from astropy.visualization import interval
#from astropy.io import fits
from trippy import psf, pill, MCMCfit
from stsci import numdisplay  # pylint: disable=import-error
import best
from maphot_functions import (getArguments, getObservations, coordRateAngle,
                              getSExCatalog, predicted2catalog,
                              runMCMCCentroid, saveTNOmag, saveStarMag,
                              getDataHeader, addPhotToCatalog, PS1SExCat,
                              PS1_to_CFHT, inspectStars,
                              __version__)
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')

###############################################################################

useage = 'maphot -c <coordsfile> -f <fitsfile> -v False '\
         + '-. False -o False -r False -a 0.7'
(inputFile, coordsfile, verbose, centroid, overrideSEx, remove,
 aprad, repfact, pxscale, roundAperRad, SExParFile, extno
 ) = getArguments(sys.argv, useage)

print("ifile =", inputFile, ", coords =", coordsfile, ", verbose =", verbose,
      ", centroid =", centroid, ", overrideSEx =", overrideSEx,
      ", remove =", remove, ", aprad =", aprad)
if verbose:
  print(np.array([centroid]).dtype, np.array([remove]).dtype)
  if centroid or remove:
    print("Will run MCMC centroiding")

# Read in the image and get the data, header and keywords needed.
(data, header, EXPTIME, MAGZERO, MJD, GAIN, NAXIS1, NAXIS2, WCS
 ) = getDataHeader(inputFile, extno=extno)

# Set up an output file that has all sorts of information.
# Preferably, whenever something is printed to screen, save it here too.
inputName = inputFile[:-5]
outfile = open(inputName + '.trippy', 'w')
print("############################")
print("Working on ", inputFile)
print("############################")
outfile.write("\nWorking on {}.\n".format(inputFile))
print("\nMJD = ", MJD)
outfile.write("\nMJD = {}\n".format(MJD))

#Get the object coordinates and rates of motion
with open(coordsfile) as han:
    mpc = han.readlines()
observations = getObservations(mpc)
TNOorbit = mp_ephem.BKOrbit(observations)
TNOcoord, rate, angle = coordRateAngle(TNOorbit, MJD, WCS)
print('TNO predicted to be at {},\nmoving at '.format(TNOcoord) +
      '{} pix/hr inclined {} deg (to x+).'.format(rate, angle))
outfile.write('\nTNO predicted to be at {},\nmoving at '.format(TNOcoord) +
              '{} pix/hr inclined {} deg (to x+).'.format(rate, angle))

#Estimated TNO pixel location.
x0, y0 = WCS.wcs_world2pix(TNOcoord, 1)

#Get the Source Extractor catalogue
if SExParFile is None:
  SEx_params = np.array([2.0, 2.0, 27.8, 10.0, 2.0, 2.0])
else:
  SEx_params = np.genfromtxt(SExParFile)
fullSExCat = getSExCatalog(inputName, SEx_params)

#If using the Source Extractor centroid is undesirable, set overrideSEx=True
#Otherwise, the source nearest the predicted TNO location is used.
xt, yt, centroidShift = predicted2catalog(fullSExCat, TNOcoord)
if (centroidShift > 15) | overrideSEx:
  xt, yt = x0, y0  # Use predicted location.
print("xt, yt = ", xt, yt)
outfile.write("xt, yt = {}, {}\n".format(xt, yt))

# Read in catalogue of good stars
try:
  bestCat = best.unpickleCatalogue('best.cat')
  print('Success!')
except IOError:
  print('Uh oh! Unpickling unsuccesful. Does best.cat exist?')
  print('If not, run best.best([fitsList]).')
  #best.best([glob.glob('*.fits')], repfact)
  #bestCat = best.unpickleCatalogue('best.cat')
  raise IOError('best.cat missing. Run best.best')
catalog_psf = best.findSharedCatalogue([fullSExCat, bestCat], 0)
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
  outfile.write("\n fwhm = {}\n".format(fwhm))
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
bgstars = []
print('Photometry of catalog stars')
outfile.write("\n# Photometry of catalog stars\n")
outfile.write("\n#   x       y   magnitude  dmagnitude")
for xcat, ycat in np.array(list(zip(catalog_phot['XWIN_IMAGE'],
                                    catalog_phot['YWIN_IMAGE']))):
  phot = pill.pillPhot(data, repFact=repfact)
  phot(xcat, ycat, radius=fwhm * roundAperRad, l=0.0, a=0.0, exptime=EXPTIME,
       #zpt=26.0, skyRadius=4 * fwhm, width=30.,
       zpt=MAGZERO, skyRadius=4 * fwhm, width=30.,
       enableBGSelection=verbose, display=verbose, backupMode="smart",
       trimBGHighPix=3., zscale=False)
  phot.SNR(gain=GAIN, useBGstd=True)
  print("{0:13.8f} {1:13.8f} {2:13.10f} {3:13.10f}".format(
        xcat, ycat, phot.magnitude - roundAperCorr, phot.dmagnitude))
  outfile.write("\n{0:13.8f} {1:13.8f} {2:13.10f} {3:13.10f}".format(
                xcat, ycat, phot.magnitude - roundAperCorr, phot.dmagnitude))
  bgstars.append(phot.bg)

bgmedian = np.median(bgstars)
Data = (data[np.max([0, int(yt) - 200]):np.min([NAXIS2 - 1, int(yt) + 200]),
             np.max([0, int(xt) - 200]):np.min([NAXIS1 - 1, int(xt) + 200])]
        - bgmedian)
dtransy = int(yt) - np.max([0, int(yt) - 200]) - 1
dtransx = int(xt) - np.max([0, int(xt) - 200]) - 1
Zoom = (data[np.max([0, int(yt) - 15]):np.min([NAXIS2 - 1, int(yt) + 15]),
             np.max([0, int(xt) - 15]):np.min([NAXIS1 - 1, int(xt) + 15])]
        - bgmedian)
zy = int(yt) - np.max([0, int(yt) - 15]) - 1
zx = int(xt) - np.max([0, int(xt) - 15]) - 1
m_obj = np.max(data[np.max([0, int(yt) - 5]):
                    np.min([NAXIS2 - 1, int(yt) + 5]),
                    np.max([0, int(xt) - 5]):
                    np.min([NAXIS1 - 1, int(xt) + 5])])

'''
Use MCMC fitting to fit the TSF to the object, thus centroiding on it.
This is often NOT better than the SExtractor location, especially
when the object is only barely trailed or
when the sky has a gradient (near something bright).
This fit is also used to remove the object from the image, later.
fit takes time proportional to nWalkers*(2+nBurn+nStep).
'''
xt0, yt0 = xt, yt
while True:
  (z1, z2) = numdisplay.zscale.zscale(Zoom)
  normer = interval.ManualInterval(z1, z2)
  pyl.imshow(normer(Zoom), origin='lower')
  pyl.plot([zx + x0 - int(xt0)], [zy + y0 - int(yt0)], 'k*', ms=10)
  pyl.plot([zx + xt0 - int(xt0)], [zy + yt0 - int(yt0)], 'w+', ms=10, mew=2)
  if centroid or remove:
    print("Should I be doing this?")
    xcent, ycent, fitPars, fitRange = runMCMCCentroid(goodPSF, Data, x0, y0,
                                                      m_obj, bgmedian,
                                                      dtransx, dtransy,
                                                      repfact)
    pyl.plot([zx + xcent - int(xt0)],
             [zy + ycent - int(yt0)], 'gx', ms=10, mew=2)
    print("Estimated    (black)  x,y = ", x0, y0)
    print("SExtractor   (white)  x,y = ", xt, yt)
    print("MCMCcentroid (green)  x,y = ", xcent, ycent)
    pyl.show()
    yn = input('Accept MCMC centroid (m or c), '
               + 'SExtractor centroid (S), or estimate (e)? ')
    if ('e' in yn) or ('E' in yn):  # if press e/E use estimate
      xt, yt = x0, y0
      break
    elif ('m' in yn) or ('M' in yn) or ('c' in yn) or ('C' in yn):  # centroid
      xt, yt = xcent, ycent
      break
    else:
      yn = 'S'  # else do nothing, use SExtractor co-ordinates.
      break
  else:  # if not previously centroided, check whether needed
    if (x0 == xt) & (y0 == yt):  # if TNO not seen in SExtractor, run centroid
      centroid = True
    else:  # else pick between estimate, SExtractor and recentroiding
      print("Estimated    (black)  x,y = ", x0, y0)
      print("SExtractor   (white)  x,y = ", xt, yt)
      pyl.show()
      yn = input('Accept '
                 + 'SExtractor centroid (S), or estimate (e), '
                 + ' or recentroid using MCMC (m or c)? ')
      if ('e' in yn) or ('E' in yn):  # if press e/E use estimate
        xt, yt = x0, y0
        break
      elif ('m' in yn) or ('M' in yn) or ('c' in yn) or ('C' in yn):  # cntroid
        centroid = True
      else:
        yn = 'S'  # else do nothing, use SExtractor co-ordinates.
        break


print('\nPhotometry of moving object')
outfile.write("\nPhotometry of moving object\n")
phot = pill.pillPhot(data, repFact=repfact)
# Make sure to use IRAF coordinates not numpy/sextractor coordinates!
apertures = np.arange(0.7, 1.6, 0.1)
linedmag = np.zeros(len(apertures))
for i, ap in enumerate(apertures):
  phot(xt, yt, radius=fwhm * ap, l=(EXPTIME / 3600.) * rate / pxscale,
       a=angle, skyRadius=4 * fwhm, width=6 * fwhm,
       #zpt=26.0, exptime=EXPTIME, enableBGSelection=False, display=False,
       zpt=MAGZERO, exptime=EXPTIME, enableBGSelection=False, display=False,
       backupMode="smart", trimBGHighPix=3., zscale=False)
  phot.SNR(gain=GAIN, verbose=False, useBGstd=True)
  linedmag[i] = phot.dmagnitude
bestap = apertures[np.argmin(linedmag)]
if aprad > 0:
  bestap = np.arange(aprad, aprad + 1)[0]  # stupid but wouldn't work otherwise
lineAperRad = bestap
print("Aperture used= ", bestap)
outfile.write("\nBest aperture = {}".format(bestap))
lineAperCorr = goodPSF.lineAperCorr(lineAperRad * fwhm)
print("lineAperCorr, roundAperCorr = ", lineAperCorr, roundAperCorr, "\n")
outfile.write("\nlineAperCorr,roundAperCorr={},{}".format(lineAperCorr,
                                                          roundAperCorr))

phot(xt, yt, radius=fwhm * lineAperRad, l=(EXPTIME / 3600.) * rate / pxscale,
     a=angle, skyRadius=4 * fwhm, width=6 * fwhm,
     #zpt=26.0, exptime=EXPTIME, enableBGSelection=True, display=True,
     zpt=MAGZERO, exptime=EXPTIME, enableBGSelection=True, display=True,
     backupMode="smart", trimBGHighPix=3., zscale=False)
phot.SNR(gain=GAIN, verbose=True, useBGstd=True)

# Print those values
print("phot.magnitude = ", phot.magnitude)
print("phot.dmagnitude = ", phot.dmagnitude)
print("phot.sourceFlux = ", phot.sourceFlux)
print("phot.snr = ", phot.snr)
print("phot.bg = ", phot.bg)
outfile.write("\nphot.magnitude={}".format(phot.magnitude))
outfile.write("\nphot.dmagnitude={}".format(phot.dmagnitude))
outfile.write("\nphot.sourceFlux={}".format(phot.sourceFlux))
outfile.write("\nphot.snr={}".format(phot.snr))
outfile.write("\nphot.bg={}".format(phot.bg))

print("\nFINAL RESULT!")
print("#{0:12} {1:13} {2:13} {3:13} {4:13}".format(
      '   x ', '    y ', ' magnitude ', '  dmagnitude ', ' magzero '))
print("{0:13.8f} {1:13.8f} {2:13.10f} {3:13.10f} {4:13.10f}\n".format(
      xt, yt, phot.magnitude - lineAperCorr, phot.dmagnitude, MAGZERO))
outfile.write("\nFINAL RESULT!")
outfile.write("\n#{0:12} {1:13} {2:13} {3:13} {4:13}\n".format(
              '   x ', '    y ', ' magnitude ', '  dmagnitude ', ' magzero '))
outfile.write("{0:13.8f} {1:13.8f} {2:13.10f} {3:13.10f} {4:13.10f}\n".format(
              xt, yt, phot.magnitude - lineAperCorr, phot.dmagnitude, MAGZERO))

# Match phot stars to PS1 catalog
PS1PhotCat = addPhotToCatalog(goodFits[:, 4], goodFits[:, 5],
                              PS1SExCat, photCat)
# Convert star catalog's PS1 magnitudes to CFHT magnitudes
finalCat = PS1_to_CFHT(PS1PhotCat)
# Calculate magnitude calibration factor
magCalibArray = (finalCat[obsFILTER + 'MeanPSFMag_Gemini']
                 - finalCat[obsFILTER + 'MagTrippy'
                            + '{}'.format(aperMulti[faveAper])])
magCalibration = np.median(magCalibArray)
dmagCalibration = np.std(magCalibArray)

# Correct the TNO magnitude and zero point
finalTNOphotCFHT = (TNOphot.magnitude[faveAper] + magCalibration,
                    (TNOphot.dmagnitude[faveAper] ** 2
                    + dmagCalibration ** 2) ** 0.5)
zptGood = zpt + magCalibration
finalTNOphotPS1 = CFHT_to_PS1(finalTNOphotCFHT[0], finalTNOphotCHFT[1], 'r')

#Save TNO magnitudes neatly.
timeNow = datetime.now().strftime('%Y-%m-%d/%H:%M:%S')
saveTNOmag(image_fn, mpc_fn, headerMJD, obsMJD, SExTNOCoord, x_tno, y_tno, 
           zpt, obsFILTER, FWHM, aperMulti, TNOphot,
           magCalibration, dmagCalibration, finalTNOphotCFHT, zptGood,
           finalTNOphotPS1, timeNow, TNObgRegion, __version__)
saveStarMag(image_fn, timeNow, __version__, finalCat, headerMJD)

# You could stop here.
# However, to confirm that things are working well,
# let's generate the trailed PSF and subtract the object out of the image.
if centroid and remove and (('e' in yn) or ('E' in yn) or
                            ('s' in yn) or ('S' in yn)):
  Data = (data[np.max([0, int(yt) - 200]):np.min([NAXIS2 - 1, int(yt) + 200]),
               np.max([0, int(xt) - 200]):np.min([NAXIS1 - 1, int(xt) + 200])]
          - phot.bg)
  dtransy = int(yt) - np.max([0, int(yt) - 200]) - 1
  dtransx = int(xt) - np.max([0, int(xt) - 200]) - 1
  m_obj = np.max(data[np.max([0, int(yt) - 5]):
                      np.min([NAXIS2 - 1, int(yt) + 5]),
                      np.max([0, int(xt) - 5]):
                      np.min([NAXIS1 - 1, int(xt) + 5])])
  print("Should I be doing this?")
  fitter = MCMCfit.MCMCfitter(goodPSF, Data)
  fitter.fitWithModelPSF(dtransx + xt - int(xt), dtransy + yt - int(yt),
                         m_in=m_obj / repfact ** 2., fitWidth=2, nWalkers=10,
                         nBurn=10, nStep=10, bg=phot.bg, useLinePSF=True,
                         verbose=True, useErrorMap=False)
  (fitPars, fitRange) = fitter.fitResults(0.67)

if centroid or remove:
  print("\nfitPars = ", fitPars, "\n")
  print("\nfitRange = ", fitRange, "\n")
  outfile.write("\nfitPars={}".format(fitPars))
  outfile.write("\nfitRange={}".format(fitRange))
  removed = goodPSF.remove(fitPars[0], fitPars[1], fitPars[2],
                           Data, useLinePSF=True)
  (z1, z2) = numdisplay.zscale.zscale(removed)
  normer = interval.ManualInterval(z1, z2)
  modelImage = goodPSF.plant(fitPars[0], fitPars[1], fitPars[2], Data,
                             addNoise=False, useLinePSF=True, returnModel=True)
  pyl.imshow(normer(goodPSF.lookupTable), origin='lower')
  pyl.show()
  #pyl.imshow(normer(modelImage), origin='lower')
  #pyl.show()
  #pyl.imshow(normer(Data), origin='lower')
  #pyl.show()
  #pyl.imshow(normer(removed), origin='lower')
  #pyl.show()
  hdu = pyf.PrimaryHDU(modelImage, header=han[0].header)
  list = pyf.HDUList([hdu])
  list.writeto(inputName + '_modelImage.fits', overwrite=True)
  hdu = pyf.PrimaryHDU(removed, header=han[0].header)
  list = pyf.HDUList([hdu])
  list.writeto(inputName + '_removed.fits', overwrite=True)
else:
  (z1, z2) = numdisplay.zscale.zscale(Data)
  normer = interval.ManualInterval(z1, z2)
  pyl.imshow(normer(goodPSF.lookupTable), origin='lower')
  pyl.show()
  #pyl.imshow(normer(Data), origin='lower')
  #pyl.show()

hdu = pyf.PrimaryHDU(goodPSF.lookupTable, header=han[0].header)
list = pyf.HDUList([hdu])
list.writeto(inputName + '_lookupTable.fits', overwrite=True)
hdu = pyf.PrimaryHDU(Data, header=han[0].header)
list = pyf.HDUList([hdu])
list.writeto(inputName + '_Data.fits', overwrite=True)

print('Done with ' + inputFile + '!')
outfile.close()
# End of file.
# Nothing to see here.
