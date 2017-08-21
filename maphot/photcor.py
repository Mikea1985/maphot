#!/usr/bin/python
'''
Photcor callibrates the photometry relative to stars
that are vissible in all frames.

Chagelog: see git log.

Originally, this module could use several different aperture sizes at once.
This was abandoned, but not removed, so some functions still have a weird
structure as a result of this. Sorry.
A lot has been added and edited since that time, but at least the functions
with an "aperture" or "naperture" argument should still work with
multiple apertures, if that's ever of interest.
'''
from __future__ import print_function, division
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy import coordinates as coords
from astropy.table.table import Table as AstroTable
from astropy.wcs import WCS
from astropy.io import fits
from astroquery.sdss import SDSS
from astroquery.vizier import Vizier
#import uncertainties as u
from uncertainties import unumpy as unp
__author__ = ('Mike Alexandersen (@mikea1985, github: mikea1985, '
              'mike.alexandersen@alumni.ubc.ca)')

def readtrippyfile(filename):
  '''Use this function to read in trippy files.
     Returns a bunch of stuff.'''
  file = open(filename, 'r')
  maglines = [line for line in file if re.match(re.compile(
              r'....\......... ....\......... ..\...........  .\...........'
              ), line)]
  file.close()
  file = open(filename, 'r')
  MJDline = [line for line in file if re.match(re.compile('MJD = '), line)]
  xcoord, ycoord, magni, dmagni = np.genfromtxt(maglines,
                                                usecols=(0, 1, 2, 3),
                                                unpack=True)
  mjdate = np.genfromtxt(MJDline, usecols=(2), unpack=True)
  file.close()
  return (xcoord[-1], ycoord[-1], magni[-1], dmagni[-1],
          xcoord[0:-1], ycoord[0:-1], magni[0:-1], dmagni[0:-1], mjdate)


def readmagfile(filename, nobject, naperture):
  '''I don't remember what this does...
     It's not currently used.'''
  aperture = np.zeros(naperture)
  magnitude, magerror = np.zeros([2, nobject, naperture])
  for ii in np.arange(nobject):
    aperture, magnitude[ii], magerror[ii] = \
        np.genfromtxt(filename, usecols=(0, 4, 5), unpack=True,
                      skip_header=79 + ii * 23, max_rows=naperture)
  return aperture, magnitude, magerror


def averagemag(magnitude, magerror, naperture, useobject):
  '''Calculates the average magnitude of the selected stars.'''
  meanmag, meanerror = np.zeros([2, naperture])
  if naperture == 1:
    meanmag = np.mean(magnitude[useobject])
    meanerror = np.std(magerror[useobject])
  else:
    for ii in np.arange(naperture):
      meanmag[ii] = np.mean(magnitude[useobject, ii])
      meanerror[ii] = np.std(magerror[useobject, ii])
  return meanmag, meanerror


def plotapcor(allobjects, useobject, aperture, magnitude,
              magerror, meanmag, meanerror):
  '''Plot the apperture correction.
     Not currently used.'''
  fig1, ax1 = plt.subplots()
  fig2, ax2 = plt.subplots()
  fig1 = fig1  # This is just here to make pylint shut up!
  fig2 = fig2  # This is just here to make pylint shut up!
  for ii in allobjects:
    ax1.errorbar(aperture, magnitude[ii, :], magerror[ii, :],
                 lw=useobject[ii])
    ax2.plot(aperture, magerror[ii, :], lw=useobject[ii])
  ax1.errorbar(aperture, meanmag[:], meanerror[:], lw=3)
  ax2.plot(aperture, meanerror[:], lw=3)
  plt.show()


def printscatter(useobject, x, y, relmagnitude):
  '''Print the scatter of each star across exposures'''
  maxi, maxstd, numsource = 0, 0, 0
  useindex = np.arange(len(useobject))[useobject]
  for ii in useindex:
    numsource += 1
    print(" {0:3d} | {1:4.0f} {2:4.0f} |".format(ii, x[ii], y[ii]), end='')
    for jj in np.arange(len(relmagnitude[0, 0, :])):
      stdrelmag = np.std(relmagnitude[:, ii, jj])
      print("{0:3.0f}".format(stdrelmag * 1000), end='')
    if stdrelmag > maxstd:
      maxstd = stdrelmag
      maxi = ii
    print(" ")
  print("----------------------")
  print(" {0:3d} | {1:4.0f} {2:4.0f} |".format(maxi, x[maxi], y[maxi]) +
        " {0:3.0f} <- Max scatter".format(maxstd * 1000) +
        " of {} sources.".format(numsource))
  return maxi, maxstd, numsource


def plotscatter(aperture, alltimes, relmagnitude, useobject,
                magerror, x, y, verbosity=True):
  '''Plot the scatter'''
  scattererr = np.zeros([len(alltimes), len(aperture)])
  for jj in np.arange(len(aperture)):
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax4 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax5 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    useindex = np.arange(len(useobject))[useobject]
    [ax3.plot(alltimes, relmagnitude[:, ii, jj], lw=useobject[ii]
              ) for ii in useindex]
    ax3.set_xlabel(aperture[jj])
    scattermag = np.array([relmagnitude[:, ii, jj] -
                           np.mean(relmagnitude[:, ii, jj])
                           for ii in np.arange(len(useobject))]).T
    [ax4.errorbar(alltimes, scattermag[:, ii], magerror[:, ii, jj],
                  lw=useobject[ii], zorder=2) for ii in useindex]
    scattererr[:, jj] = [np.nanstd(scattermag[tt, useindex])
                         for tt in np.arange(len(alltimes))]
    ax4.errorbar(alltimes, np.zeros(len(alltimes)), scattererr[:, jj], lw=0,
                 capsize=20, elinewidth=0, color='k', zorder=1)
    ax4.set_xlabel(aperture[jj])
    [ax5.plot(x[ii], y[ii], 'o', ms=np.std(scattermag[:, ii]) * 300, alpha=0.3
              ) for ii in useindex]
    ax5.axis([0, 2048, 0, 4176])
    if verbosity:
      print("Mean |           | {0:3.0f}"
            .format(np.mean(scattererr[:]) * 1000))
      print("Max  |           | {0:3.0f}".format(np.max(scattererr[:]) * 1000))
    plt.show()
  return scattererr


def trimcatalog(xstar, ystar, magstar, dmagstar):
  '''This trims all stars out of the catalog that do not have
     measured magnitudes in every frame.'''
  nobjmax = np.sum([len(xstar[tt]) for tt in np.arange(len(xstar))])
  master = np.zeros([nobjmax, len(xstar) * 2 + 2])
  n0 = len(xstar[0])
  master[0:n0, 0:4] = np.array([xstar[0], ystar[0], magstar[0], dmagstar[0]]).T
  nobjec = n0
  for tt in np.arange(1, len(xstar)):
    for s in np.arange(len(xstar[tt])):
      d = ((master[:nobjec, 0] - xstar[tt][s]) ** 2 +
           (master[:nobjec, 1] - ystar[tt][s]) ** 2) ** 0.5
      idx = d.argmin()
      if d[idx] < 5:  # if match previous star,  add mag to its array
        master[idx, 2 + 2 * tt:4 + 2 * tt] = magstar[tt][s], dmagstar[tt][s]
      else:  # else add a new star entry
        master[nobjec, 2 + 2 * tt:4 + 2 * tt] = magstar[tt][s], dmagstar[tt][s]
        master[nobjec, 0:2] = xstar[tt][s], ystar[tt][s]
        nobjec += 1
  trimlist = []
  for s in np.arange(nobjec):
    if len(np.where(master[s] == 0)[0]) == 0:
      trimlist.append(s)
  trimmed = master[trimlist]
  return trimmed


def trimcatalog_unwrap(xstar, ystar, magstar, dmagstar):
  '''A wrapper to make pylint shut up about me having too many variables
     in a single functions.'''
  trimmed = trimcatalog(xstar, ystar, magstar, dmagstar)
  xtrim, ytrim = trimmed[:, 0], trimmed[:, 1]
  magtrim, dmagtrim = trimmed[:, 2::2], trimmed[:, 3::2]
  idx = np.argsort(np.mean(trimmed, 1))
  return xtrim[idx], ytrim[idx], magtrim[idx], dmagtrim[idx]


def sdss_check(x, y):
  """
  Check whether stars are in the SDSS catalogue.
  This function accepts either a single x and y coordinate,
  or an array of each.
  """
  w = WCS('a100.fits')
  sfilt = []
  # Check which format x and y are given in.
  if not (isinstance(x, (np.ndarray, list, float, int)) &
          isinstance(y, (np.ndarray, list, float, int)) &
          (np.shape(x) == np.shape(y))):
    print('Error: Need a set of pixel coordinates.')
    print('       X and Y must have same non-zero size.')
    raise TypeError
  x = [x] if (np.shape(x) == ()) else x
  y = [y] if (np.shape(y) == ()) else y
  lon, lat = w.all_pix2world(x, y, 1)
  pos = coords.SkyCoord(lon, lat, unit="deg")
  if len(pos) == 1:
    pos = [pos]
  table_fields = ['RA', 'Dec', 'psfMag_r', 'psfMagErr_r',
                  'psffwhm_r', 'nDetect', 'X_pixel', 'Y_pixel']
  sfilt = AstroTable(names=table_fields)
  for index, position in enumerate(pos):
    sfull = SDSS.query_region(position, radius='1arcsec', data_release=13,
                              photoobj_fields=table_fields[:-2])
    try:
      sline = (sfull[np.where((sfull['nDetect'] > 0) &
                              (sfull['psfMag_r'] > -99))[0]][0])
      slist = [sl for sl in sline]
      slist.append(x[index])
      slist.append(y[index])
      sfilt.add_row(slist)
    except (TypeError, IndexError):
      print("Star at " + str(position)[39:-1] + " not found :-(.")
      slist = np.zeros(len(table_fields))
      slist[-2:] = x[index], y[index]
      sfilt.add_row(slist)
  return sfilt


def usno_check(x, y):
  """
  Check whether stars are in the USNO catalogue.
  This function accepts either a single x and y coordinate,
  or an array of each.
  """
  w = WCS('a100.fits')
  sfilt = []
  # Check which format x and y are given in.
  if not (isinstance(x, (np.ndarray, list, float, int)) &
          isinstance(y, (np.ndarray, list, float, int)) &
          (np.shape(x) == np.shape(y))):
    print('Error: Need a set of pixel coordinates.')
    print('       X and Y must have same non-zero size.')
    raise TypeError
  x = [x] if (np.shape(x) == ()) else x
  y = [y] if (np.shape(y) == ()) else y
  lon, lat = w.all_pix2world(x, y, 1)
  pos = coords.SkyCoord(lon, lat, unit="deg")
  if len(pos) == 1:
    pos = [pos]
  table_fields = ['RAJ2000', 'DEJ2000', 'R2mag', 'Ndet', 'X_pixel', 'Y_pixel']
  sfilt = AstroTable(names=table_fields)
  vizier = Vizier(columns=table_fields, catalog="I/284")
  for index, position in enumerate(pos):
    try:
      sfull = vizier.query_region(position, radius='2s')[0][table_fields[:-2]]
      sline = (sfull[np.where((sfull['Ndet'] > 0) &
                              (sfull['R2mag'] > -99))[0]][0])
      slist = [sl for sl in sline]
      slist.append(x[index])
      slist.append(y[index])
      sfilt.add_row(slist)
    except TypeError:
      print("Star at " + str(position)[39:-1] + " not found in USNO :-(.")
      slist = np.zeros(len(table_fields))
      slist[-2:] = x[index], y[index]
      sfilt.add_row(slist)
    except IndexError:
      print("Star at " + str(position)[39:-1] + " has no USNO magnitude :-(.")
      slist = np.zeros(len(table_fields))
      slist[-2:] = x[index], y[index]
      sfilt.add_row(slist)
  return sfilt


def print_tno_file(objname, odometer, julian, corrected,
                   error, magcalerr, magzero):
  '''
  Write a file with the calibrated TNO magnitudes.
  '''
  calmagfile = objname + '.txt'
  texmagfile = objname + '.tex'
  calfile = open(calmagfile, 'w')
  texfile = open(texmagfile, 'w')
  print("#Odo          mjd              magnitude        " +
        "dmagnitude       Calibration_err  zero-point")
  calfile.write("#Odo          mjd              magnitude        " +
                "dmagnitude       calibration_err  zero-point\n")
  texfile.write("%mjd                magnitude             " +
                "dmagnitude           calibration_err    zero-point\n")
  texfile.write(r"\midrule" + "\n")
  for j, odo in enumerate(odometer):
    print("{0:13s} {1:16.10f} ".format(odo, julian[j]) +
          "{0:16.13f} {1:16.13f} ".format(corrected[j], error[j]) +
          "{0:16.13f} {1:16.13f}".format(magcalerr, magzero[j]))
    calfile.write("{0:13s} {1:16.10f} ".format(odo, julian[j]) +
                  "{0:16.13f} {1:16.13f} ".format(corrected[j], error[j]) +
                  "{0:16.13f} {1:16.13f} \n".format(magcalerr, magzero[j]))
    texfile.write("{0:16.10f} & ".format(julian[j]) +
                  r"${0:16.13f} \pm ".format(corrected[j]) +
                  r"{0:16.13f} \pm ".format(error[j]) +
                  "{0:16.13f}$ & ".format(magcalerr) +
                  "{0:16.13f} \n".format(magzero[j]))
  texfile.write(r"\midrule" + "\n")
  calfile.close()


def print_stars_file(calstarfile, useobjects,
                     xcoord, ycoord, r_magnitude, r_magerr,
                     c_magnitude, c_magerr,
                     sdss_magnitude=None, sdss_magerror=None):
  '''
  Print the file with the calibration stars used.
  '''
  calfile = open(calstarfile, 'w')
  if (np.sum(sdss_magnitude) is not None) & \
     (np.sum(sdss_magerror is not None)):
    print("#xcoo            ycoo             mag              dmag        " +
          "     calibrated_mag   calibrated_dmag  sdss_mag         sdss_dmag")
    calfile.write("#xcoo            ycoo             ccd_mag          " +
                  "ccd_dmag         calibrated_mag   calibrated_dmag  " +
                  "sdss_mag         sdss_dmag\n")
  else:
    print("#xcoo            ycoo             mag              dmag        " +
          "     calibrated_mag   calibrated_dmag")
    calfile.write("#xcoo            ycoo             ccd_mag          " +
                  "ccd_dmag         calibrated_mag   calibrated_dmag\n")
  for j in useobjects:
    r_u = np.mean(unp.uarray(r_magnitude[j], r_magerr[j])).s
    r_m = np.mean(unp.uarray(r_magnitude[j], r_magerr[j])).n
    c_u = np.mean(unp.uarray(c_magnitude[j], c_magerr[j])).s
    c_m = np.mean(unp.uarray(c_magnitude[j], c_magerr[j])).n
    if (np.sum(sdss_magnitude) is not None) & \
       (np.sum(sdss_magerror is not None)):
      print("{0:16.11f} {1:16.11f} ".format(xcoord[j], ycoord[j]) +
            "{0:16.13f} {1:16.13f} ".format(r_m, r_u) +
            "{0:16.13f} {1:16.13f} ".format(c_m, c_u) +
            "{0:16.13f} {1:16.13f}".format(sdss_magnitude[j],
                                           sdss_magerror[j]))
      calfile.write("{0:16.11f} {1:16.11f} {2:16.13f} {3:16.13f} "
                    .format(xcoord[j], ycoord[j], r_m, r_u) +
                    "{0:16.13f} {1:16.13f} ".format(c_m, c_u) +
                    "{0:16.13f} {1:16.13f}\n".format(sdss_magnitude[j],
                                                     sdss_magerror[j]))
    else:
      print("{0:16.11f} {1:16.11f} ".format(xcoord[j], ycoord[j]) +
            "{0:16.13f} {1:16.13f} ".format(r_m, r_u) +
            "{0:16.13f} {1:16.13f} ".format(c_m, c_u))
      calfile.write("{0:16.11f} {1:16.11f} {2:16.13f} {3:16.13f} "
                    .format(xcoord[j], ycoord[j], r_m, r_u) +
                    "{0:16.13f} {1:16.13f}\n".format(c_m, c_u))
  calfile.close()


def calculate_corrected_TNO_mags(reduced_star_mags, ccd_average,
                                 reduced_tno_mags, tno_mags_error,
                                 usecatalogue, catalogue_magnitudes,
                                 trippymag):
  '''
  Calculate the corrected TNO magnitudes.
  Calibrate to a catalogue, if possible, otherwise calibrate to average = 0.
  '''
  catalogue_average = np.mean(catalogue_magnitudes)
  delta_averages = catalogue_average - ccd_average
  if usecatalogue:
    mag_correction = ccd_average + delta_averages
    mag_star_mean = np.mean((reduced_star_mags + mag_correction), 1)
    mag_correction_err = np.std(mag_star_mean - catalogue_magnitudes)
  else:
    mag_correction = -np.mean(reduced_tno_mags)
    mag_star_mean = 0
    mag_correction_err = np.nan
  tno_corrected = trippymag + mag_correction
  tnoerr_corrected = tno_mags_error
  tno_corrected_err = mag_correction_err
  return tno_corrected, tnoerr_corrected, tno_corrected_err


def StarInspector(useobject, xstar, ystar, julian, magnitude, magerror,
                  autokill=False):
  '''
  Allow user to weed out any variable stars.
  autokill=True automatically kills the most variable star iteratively until
  there are less than 25 stars or the maximum stddev is <=0.025 mag.
  '''
  yn = 'Yes'
  while not ('n' in yn) | ('N' in yn):
    average = np.mean(magnitude[useobject], 0)
    reduced_mag = (magnitude - average).T
    print("Star |  x    y   | Standard deviation (milli-mags)")
    imax, stdmax, nsource = printscatter(useobject, xstar, ystar,
                                         np.array([reduced_mag.T]).T)
    if not ('n' in yn) | ('N' in yn):
      if autokill & (nsource > 25) & (stdmax > 0.025):
        killobj = imax
      else:
        scattererr = plotscatter([1], mjd, np.array([reduced_mag.T]).T,
                                 useobject, np.array([magerror]).T,
                                 xstar, ystar)
        killobj = raw_input("Which star would you like to kill? " +
                            "(# above,  n for none) ")
      try:
        kobj = int(killobj)
        useobject[kobj] = False
        magnitude[kobj, :] = np.nan
        magerror[kobj, :] = np.nan
        continue
      except ValueError:
        if ('n' in killobj) | ('N' in killobj):
          break
        else:
          print("That's not a valid number!")
          continue
  average = np.mean(magnitude[useobj], 0)
  reduced_mag = (magnitude - average).T
  print("Star |  x    y   | Standard deviation (milli-mags)")
  printscatter(useobject, xstar, ystar, np.array([reduced_mag.T]).T)
  scattererr = plotscatter([1], julian, np.array([reduced_mag.T]).T, useobject,
                           np.array([magerr]).T, xccd, yccd)
  return useobject, scattererr, average, reduced_mag


def readzeropoint(filename):
  '''
  Reads the zeropoint of a file.
  '''
  hdulist = fits.open(filename)
  magzero = hdulist[0].header['MAGZERO']
  magzeroerr = hdulist[0].header['MAGZERO_RMS']
  hdulist.close()
  return magzero, magzeroerr


###############################################################################
## Functions above here. ######################################################
###############################################################################
## Main below here. ###########################################################
###############################################################################

verbose = True
files = glob.glob("./a???.trippy")
files.sort()
ntimes = len(files)
xcoo = list(np.zeros(ntimes))
ycoo, mag, magerr, rmag = xcoo[:], xcoo[:], xcoo[:], xcoo[:]
magin, magerrin = xcoo[:], xcoo[:]
avmag = np.zeros(ntimes)
mjd = avmag.copy()
zeros = avmag.copy()
zeroserr = avmag.copy()
xobj, yobj, magobj, magerrobj, rmagobj = np.zeros([5, ntimes])
avmagobj, avmagerrobj = np.zeros([2, ntimes])
zeros_default = 26.0
usesdss = False

for t, infile in enumerate(files):
  print(infile)
  (xobj[t], yobj[t], magobj[t], magerrobj[t], xcoo[t], ycoo[t],
   magin[t], magerrin[t], mjd[t]) = readtrippyfile(infile)
  zeros[t], zeroserr[t] = readzeropoint(infile[:-7] + '.fits')

if zeros_default == 'None':
  zeros_default = zeros

'''
Fix the zeropoint (don't use the default 26.0) # needed because I was stupid
'''
dzero = zeros - zeros_default  # So real mag = measured + dzero
magobj += dzero
magin = (np.array(magin).T + dzero).T
zeroserr[np.argwhere(zeroserr < 0.01)] = 0.01

'''
Don't use any stars that don't have magnitudes in all frames.
'''
xccd, yccd, mag, magerr = trimcatalog_unwrap(xcoo, ycoo, magin, magerrin)
useobj = mag[:, 0] < 30
nobj = np.shape(mag)[0]
objects = np.arange(nobj)

'''
Check whether the stars are in the SDSS catalogue.
If enough are, stop using those that are not.
'''
if usesdss:
  sdss = sdss_check(xccd, yccd)
  sdss_mag = np.array(sdss['psfMag_r'])
  sdss_magerr = np.array(sdss['psfMagErr_r'])
  insdss = sdss['nDetect'] > 0
  nsdss = len(np.where(insdss)[0])
  if nsdss >= 10:
    usesdss = False
    useobj[np.invert(insdss)] = False
  else:
    usesdss = True
print("Using SDSS stars: " + str(usesdss))
print("Using {0:3.0f} of {1:3.0f} stars.".format(len(useobj[useobj]),
                                                 len(useobj)))

'''
Inspect the stars and get rid of any variable ones.
'''
useobj, scaterr, avmag, rmag = StarInspector(useobj, xccd, yccd, mjd,
                                             mag, magerr, True)

'''
Calculate the zero-point correction.
'''
ddzero = np.mean(avmag) - avmag
zeros_corrected = zeros_default + dzero + ddzero
magobj_corrected = magobj + ddzero
magerrobj_corrected = (magerrobj ** 2 + scaterr[:, 0] ** 2) ** 0.5
print("Zero-point calibration from relative photometry:")
print(ddzero)

'''
Now read in measured object and correct it.
'''
plt.errorbar(mjd, magobj[:], magerrobj[:], fmt='--')
plt.errorbar(mjd, avmag[:], scaterr[:, 0])
correltnomag = np.array([(magobj - avmag)[t] for t, time in enumerate(mjd)])
plt.errorbar(mjd, magobj_corrected, magerrobj_corrected,
             lw=1, capsize=10, elinewidth=2)
plt.gca().invert_yaxis()

plt.figure()
plt.errorbar(mjd, magobj_corrected, magerrobj_corrected, marker='+', lw=0,
             capsize=10, elinewidth=2, label='TRIPPy')
plt.gca().invert_yaxis()
plt.legend(loc='best')
plt.xlabel('Time (MJD)')
plt.ylabel(r'$\Delta \mathrm{mag}$ (mag-mean)')
plt.show()

magobj_done = magobj_corrected.copy()
magerrobj_done = magerrobj_corrected.copy()
systematic_err = np.mean(unp.uarray(zeros, zeroserr)).std_dev
# Systematic error is technically not from averaging the zero points,
# but this should be pretty close.
mag_done = mag + ddzero
magerr_done = (magerr ** 2 + scaterr[:, 0] ** 2) ** 0.5 + systematic_err

print_tno_file('calibratedmags', files, mjd, magobj_done,
               magerrobj_done, systematic_err, zeros_corrected)

if usesdss:
  print_stars_file('calibrationstars.txt', objects[useobj],
                   xccd, yccd, mag, magerr, mag_done, magerr_done,
                   sdss_mag, sdss_magerr)
else:
  print_stars_file('calibrationstars.txt', objects[useobj],
                   xccd, yccd, mag, magerr, mag_done, magerr_done)
#
