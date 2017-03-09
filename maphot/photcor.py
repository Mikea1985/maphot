#!/usr/bin/python
'''
Photcor callibrates the photometry relative to stars
that are vissible in all frames.
'''
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy import coordinates as coords
from astropy.table.table import Table as astroTable
from astropy.wcs import WCS
from astroquery.sdss import SDSS


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


def printscatter(useindex, x, y, relmagnitude):
  '''Print the scatter of each star across exposures'''
  maxi, maxstd, numsource = 0, 0, 0
  for ii in useindex:
    numsource += 1
    print " {0:3d} | {1:4.0f} {2:4.0f} |".format(ii, x[ii], y[ii]),
    for jj in np.arange(len(relmagnitude[0, 0, :])):
      stdrelmag = np.std(relmagnitude[:, ii, jj])
      print "{0:3.0f}".format(stdrelmag * 1000),
    if stdrelmag > maxstd:
      maxstd = stdrelmag
      maxi = ii
    print " "
  print "----------------------"
  print " {0:3d} | {1:4.0f} {2:4.0f} |".format(maxi, x[maxi], y[maxi]),
  print "{0:3.0f} <- Max scatter".format(maxstd * 1000),
  print " of {} sources.".format(numsource)
  return maxi, maxstd, numsource


def plotscatter(aperture, alltimes, relmagnitude, useobject, magerror, x, y):
  '''Plot the scatter'''
  scattererror = np.zeros([len(alltimes), len(aperture)])
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
    scattererror[:, jj] = [np.nanstd(scattermag[tt, useindex])
                           for tt in np.arange(len(alltimes))]
    ax4.errorbar(alltimes, np.zeros(len(alltimes)), scattererror[:, jj], lw=0,
                 capsize=20, elinewidth=0, color='k', zorder=1)
    ax4.set_xlabel(aperture[jj])
    [ax5.plot(x[ii], y[ii], 'o', ms=np.std(scattermag[:, ii]) * 300, alpha=0.3
              ) for ii in useindex]
    ax5.axis([0, 2048, 0, 4176])
    plt.show()
  return scattererror


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
    print 'Error: Need a set of pixel coordinates.'
    print '       X and Y must have same non-zero size.'
    raise TypeError
  x = [x] if (np.shape(x) == ()) else x
  y = [y] if (np.shape(y) == ()) else y
  lon, lat = w.all_pix2world(x, y, 1)
  pos = coords.SkyCoord(lon, lat, unit="deg")
  if len(pos) == 1:
    pos = [pos]
  table_fields = ['RA', 'Dec', 'psfMag_r', 'psfMagErr_r',
                  'psffwhm_r', 'nDetect', 'X_pixel', 'Y_pixel']
  sfilt = astroTable(names=table_fields)
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
    except TypeError:
      print "Star at " + str(position)[39:-1] + " not found :-(."
      slist = np.zeros(len(table_fields))
      slist[-2:] = x[index], y[index]
      sfilt.add_row(slist)
  return sfilt


verbose = 1
files = glob.glob("./a???.trippy")
files.sort()
ntimes = len(files)
xcoo = list(np.zeros(ntimes))
ycoo, mag, magerr, rmag = xcoo[:], xcoo[:], xcoo[:], xcoo[:]
magin, magerrin = xcoo[:], xcoo[:]
avmag = np.zeros(ntimes)
avmagerr, mjd = avmag.copy(), avmag.copy()
xobj, yobj, magobj, magerrobj, rmagobj = np.zeros([5, ntimes])
avmagobj, avmagerrobj = np.zeros([2, ntimes])

for t, infile in enumerate(files):
  print infile
  (xobj[t], yobj[t], magobj[t], magerrobj[t], xcoo[t], ycoo[t],
   magin[t], magerrin[t], mjd[t]) = readtrippyfile(infile)

'''
Stop using any objects that don't have magnitudes in all frames.
'''
xcat, ycat, mag, magerr = trimcatalog_unwrap(xcoo, ycoo, magin, magerrin)
useobj = mag[:, 0] < 30
nobj = np.shape(mag)[0]
objects = np.arange(nobj)

'''
Check whether the stars are in the SDSS catalogue.
If enough are, stop using those that are not.
'''
sdss = sdss_check(xcat, ycat)
sdss_mag = np.array(sdss['psfMag_r'])
sdss_magerr = np.array(sdss['psfMagErr_r'])
insdss = sdss['nDetect'] > 0
nsdss = len(np.where(insdss)[0])
if nsdss >= 10:
  usesdss = True
  useobj[np.invert(insdss)] = False
else:
  usesdss = False

'''
Calculate the average again,  using only the objects that we want.
'''
rmag = np.zeros(np.shape(mag)).T
for t, time in enumerate(mjd):
  avmag[t], avmagerr[t] = averagemag(mag[:, t], magerr[:, t], 1, useobj)
  '''
  Calculate the magnitude of all stars relative to the average.
  '''
  rmag[t, :] = mag[:, t] - avmag[t]


'''
Allow user to weed out any variable stars.
'''
yn = 'Yes'
autokill = True
while not ('n' in yn) | ('N' in yn):
  for t, time in enumerate(mjd):
    avmag[t], avmagerr[t] = averagemag(mag[:, t], magerr[:, t], 1, useobj)
    rmag[t, :] = mag[:, t] - avmag[t]
  print "Star |  x    y   | Standard deviation (milli-mags)"
  imax, stdmax, nsource = printscatter(objects[useobj],
                                       xcat, ycat, np.array([rmag.T]).T)
  if not ('n' in yn) | ('N' in yn):
    if autokill & (nsource > 25) & (stdmax > 0.025):
      killobj = imax
    else:
      scaterr = plotscatter([1], mjd, np.array([rmag.T]).T, useobj,
                            np.array([magerr]).T, xcat, ycat)
      print "Mean |           | {0:3.0f}".format(np.mean(scaterr[:]) * 1000)
      print "Max  |           | {0:3.0f}".format(np.max(scaterr[:]) * 1000)
      killobj = raw_input("Which star would you like to kill? " +
                          "(# above,  n for none) ")
    try:
      kobj = int(killobj)
      useobj[kobj] = False
      mag[kobj, :] = np.nan
      magerr[kobj, :] = np.nan
#      yn = raw_input("Are there more stars you would like to ignore? (Y/n) ")
      continue
    except ValueError:
      if ('n' in killobj) | ('N' in killobj):
        break
      else:
        print "That's not a valid number!"
        continue

'''
Calculate the magnitude of all stars relative to the average.
'''
for t, time in enumerate(mjd):
  avmag[t], avmagerr[t] = averagemag(mag[:, t], magerr[:, t], 1, useobj)
  rmag[t, :] = mag[:, t] - avmag[t]
'''Plot the magnitude of objects as a function of time,
   for each aperture size.
   Also print the standard deviation of the mag for each star.'''
print "Star |  x    y   | Standard deviation (milli-mags)"
printscatter(objects[useobj], xcat, ycat, np.array([rmag.T]).T)
scaterr = plotscatter([1], mjd, np.array([rmag.T]).T, useobj,
                      np.array([magerr]).T, xcat, ycat)
print "Mean |           | {0:3.0f}".format(np.mean(scaterr[:]) * 1000)
print "Max  |           | {0:3.0f}".format(np.max(scaterr[:]) * 1000)

'''
Now read in measured object and correct it.
'''
tnomag, tnomagerr = magobj.copy(), magerrobj.copy()
plt.errorbar(mjd, tnomag[:], tnomagerr[:])
plt.errorbar(mjd, avmag[:], scaterr[:, 0])
correltnomag = np.array([(tnomag - avmag)[t] for t, time in enumerate(mjd)])
plt.errorbar(mjd, np.mean(avmag) + correltnomag, tnomagerr[:] + scaterr[:, 0],
             lw=1, capsize=20, elinewidth=3)
plt.gca().invert_yaxis()

plt.figure()
trippyerr = (tnomagerr ** 2 + scaterr[:, 0] ** 2) ** 0.5
trippymag = correltnomag
plt.errorbar(mjd, trippymag - np.mean(trippymag), trippyerr, marker='+', lw=0,
             capsize=20, elinewidth=3, label='TRIPPy')
plt.gca().invert_yaxis()
plt.legend(loc='best')
plt.xlabel('Time (MJD)')
plt.ylabel(r'$\Delta \mathrm{mag}$ (mag-mean)')
plt.show()

'''
Calculate the sdss correction, if using sdss.
'''
avsdss = np.mean(sdss_mag[useobj])
delta_sdss = avsdss - avmag
if usesdss:
  mag_correction = avmag + delta_sdss
  mag_star_mean = np.mean((rmag.T + mag_correction)[useobj], 1)
  mag_correction_err = np.std(mag_star_mean - sdss_mag[useobj])
else:
  mag_correction = -np.mean(trippymag)
  mag_star_mean = 0
  mag_correction_err = 1

tnomag_corrected = trippymag + mag_correction
tnomagerr_corrected = trippyerr
tnomag_corrected_err = mag_correction_err

calmagfil = open('calibratedmags.txt', 'w')
print "#Odo          mjd              magnitude        " + \
      "dmagnitude       Calibration_err"
calmagfil.write("#Odo          mjd              magnitude        " +
                "dmagnitude       Calibration_err\n")
for t, infile in enumerate(files):
  print "{0:13s} {1:16.10f} ".format(infile, mjd[t]) + \
        "{0:16.13f} {1:16.13f} {2:16.13f}".format(tnomag_corrected[t],
                                                  tnomagerr_corrected[t],
                                                  tnomag_corrected_err)
  calmagfil.write("{0:13s} {1:16.10f} {2:16.13f} {3:16.13f} {4:16.13f}\n"
                  .format(infile, mjd[t], tnomag_corrected[t],
                          tnomagerr_corrected[t], tnomag_corrected_err))

calmagfil.close()
calstarfil = open('calibrationstars.txt', 'w')
print "#xcoo            ycoo             mag              dmag" + \
      "             sdss_mag         sdss_dmag"
calstarfil.write("#xcoo            ycoo             ccd_mag          " +
                 "ccd_dmag         sdss_mag         sdss_dmag\n")
for i, j in enumerate(objects[useobj]):
  print "{0:16.11f} {1:16.11f} ".format(xcat[j], ycat[j]) + \
        "{0:16.13f} {1:16.13f} ".format(np.mean(mag[j]),
                                        np.sum(magerr[j] ** 2) ** 0.5) + \
        "{0:16.13f} {1:16.13f}".format(sdss_mag[j], sdss_magerr[j])
  calstarfil.write("{0:16.11f} {1:16.11f} {2:16.13f} {3:16.13f} "
                   .format(xcat[j], ycat[j], np.mean(mag[j]),
                           np.sum(magerr[j] ** 2) ** 0.5) +
                   "{0:16.13f} {1:16.13f}\n".format(sdss_mag[j],
                                                    sdss_magerr[j]))

calstarfil.close()
