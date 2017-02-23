#!/usr/bin/python
'''
Photcor callibrates the photometry relative to stars
that are vissible in all frames.
'''
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

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
    [ax3.plot(alltimes, relmagnitude[:, ii, jj], lw=useobject[ii]
              ) for ii in np.arange(len(useobject))]
    ax3.set_xlabel(aperture[jj])
    scattermag = np.array([relmagnitude[:, ii, jj] -
                           np.mean(relmagnitude[:, ii, jj])
                           for ii in np.arange(len(useobject))]).T
    [ax4.errorbar(alltimes, scattermag[:, ii], magerror[:, ii, jj],
                  lw=useobject[ii]) for ii in np.arange(len(useobject))]
    scattererror[:, jj] = [np.nanstd(scattermag[tt, :])
                           for tt in np.arange(len(alltimes))]
    ax4.errorbar(alltimes, np.zeros(len(alltimes)), scattererror[:, jj], lw=0,
                 capsize=20, elinewidth=3)
    ax4.set_xlabel(aperture[jj])
    [ax5.plot(x[ii], y[ii], 'o', ms=np.std(scattermag[:, ii]) * 300, alpha=0.3
              ) for ii in np.arange(len(useobject))]
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

verbose = 1
files = glob.glob("./a???.trippy")
files.sort()
ntimes = len(files)
#nobj = 32
#nobj = 500
napr = 14
#objects = np.arange(nobj)
#useobj = np.ones(nobj, dtype = bool)
#useobj = objects<200
#mag, magerr, rmag = np.zeros([3, ntimes, nobj, napr])
#avmag, avmagerr = np.zeros([2, ntimes, napr])
#srmag = np.zeros([nobj, napr])
xcoo = list(np.zeros(ntimes))
ycoo, mag, magerr, rmag = xcoo[:], xcoo[:], xcoo[:], xcoo[:]
magin, magerrin = xcoo[:], xcoo[:]
avmag = np.zeros(ntimes)
avmagerr, mjd = avmag.copy(), avmag.copy()
xobj, yobj, magobj, magerrobj, rmagobj = np.zeros([5, ntimes])
avmagobj, avmagerrobj = np.zeros([2, ntimes])

for t, infile in enumerate(files):
  print infile
#  apr, mag[t, :, :], magerr[t, :, :] = \
#    readmagfile('a-00'+str(time)+'-013.fits.mag.3', nobj, napr)
#  xobj[t], yobj[t], magobj[t], magerrobj[t], xcoo[t], ycoo[t], magin[t], \
#    magerrin[t] = readtrippyfile('a-00'+str(time)+'-013.trippy')  # noqa E127
  xobj[t], yobj[t], magobj[t], magerrobj[t], xcoo[t], ycoo[t], \
    magin[t], magerrin[t], mjd[t] = readtrippyfile(infile)  # noqa E127
#  avmag[t, :], avmagerr[t, :] = averagemag(mag[t, :, :], magerr[t, :, :],
#                                            napr, useobj)
#  verbose = False
#  if (verbose): plotapcor(objects, useobj, apr, mag[t, :, :],
#                           magerr[t, :, :], avmag[t, :], avmagerr[t, :])

'''
Stop using any objects that don't have magnitudes in all appertures.
'''
xcat, ycat, mag, magerr = trimcatalog_unwrap(xcoo, ycoo, magin, magerrin)
useobj = mag[:, 0] < 30
nobj = np.shape(mag)[0]
objects = np.arange(nobj)
#for t, time in enumerate(times):
#  for j in np.arange(napr):
#    nouseobj = np.isnan(mag[t, :, j])
#    mag[:, nouseobj, :] = np.nan
#    magerr[:, nouseobj, :] = np.nan
#    useobj[nouseobj] = False

'''
Calculate the average again,  using only the objects that we want.
'''
rmag = np.zeros(np.shape(mag)).T
for t, time in enumerate(mjd):
  avmag[t], avmagerr[t] = averagemag(mag[:, t], magerr[:, t], 1, useobj)
#  avmag[t, :], avmagerr[t, :] = averagemag(mag[t, :, :], magerr[t, :, :],
#                                            napr, useobj)
#  verbose = False
#  if (verbose): plotapcor(objects, useobj, apr, mag[t, :, :],
#                          magerr[t, :, :], avmag[t, :], avmagerr[t, :])
  '''
  Calculate the magnitude of all stars relative to the average.
  '''
#  rmag[t, :, :] = mag[t, :, :]-avmag[t, :]
  rmag[t, :] = mag[:, t] - avmag[t]

'''
Plot the magnitude of objects as a function of time, 
for each aperture size.
Also print the standard deviation of the mag for each star.
'''
#print "Star | Standard deviation (milli-mags) at various apertures"
#printscatter(objects[useobj], rmag)
#scaterr = plotscatter(apr, times, rmag, useobj, magerr)
#print "Mean |",
#for j in np.arange(napr):
#  print "{0:3.0f}".format(np.mean(scaterr[:, j])*1000),
#print " "
#print "Star | Standard deviation (milli-mags)"
#printscatter(objects[useobj], np.array([rmag.T]).T)
#scaterr = plotscatter([1], ttimes, np.array([rmag.T]).T, useobj,
#                      np.array([magerr]).T)
#print "Mean |",
#print "{0:3.0f}".format(np.mean(scaterr[:])*1000)

'''
Allow user to weed out any variable stars.
'''
yn = 'Yes'
autokill = True
#yn = raw_input("Are there any stars you wish to " +
#               "ignore due to variability? (Y/n) ")
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
Calculate the average again,  using only the objects that we want.
'''
#for t, time in enumerate(times):
#  avmag[t, :], avmagerr[t, :] = averagemag(mag[t, :, :], magerr[t, :, :],
#                                           napr, useobj)
#  verbose = False
#  if (verbose): plotapcor(objects, useobj, apr, mag[t, :, :],
#                          magerr[t, :, :], avmag[t, :], avmagerr[t, :])
'''
Calculate the magnitude of all stars relative to the average.
'''
#  rmag[t, :, :] = mag[t, :, :]-avmag[t, :]
for t, time in enumerate(mjd):
  avmag[t], avmagerr[t] = averagemag(mag[:, t], magerr[:, t], 1, useobj)
  rmag[t, :] = mag[:, t] - avmag[t]

'''Plot the magnitude of objects as a function of time,
   for each aperture size.
   Also print the standard deviation of the mag for each star.'''
#print "Star | Standard deviation (milli-mags) at various times"
print "Star |  x    y   | Standard deviation (milli-mags)"
printscatter(objects[useobj], xcat, ycat, np.array([rmag.T]).T)
scaterr = plotscatter([1], mjd, np.array([rmag.T]).T, useobj,
                      np.array([magerr]).T, xcat, ycat)
#printscatter(objects[useobj], rmag)
#scaterr = plotscatter(apr, times, rmag, useobj, magerr)
#print "Mean |",
#for j in np.arange(napr):
#  print "{0:3.0f}".format(np.mean(scaterr[:, j])*1000),
#print " "
print "Mean |           | {0:3.0f}".format(np.mean(scaterr[:]) * 1000)
print "Max  |           | {0:3.0f}".format(np.max(scaterr[:]) * 1000)

# Now read in measured object
tnomag, tnomagerr = magobj.copy(), magerrobj.copy()
#tnomag, tnomagerr, tnormag = np.zeros([3, ntimes, napr])
#for t, time in enumerate(times):
#  print time
#  apr, tnomag[t, :], tnomagerr[t, :] = readmagfile('a-00'+str(time) +
#                                                   '-013.fits.mag.5', 1, napr)
#  fig1, ax1 = plt.subplots()
#  fig2, ax2 = plt.subplots()
#  ax1.errorbar(apr, tnomag[t, :], tnomagerr[t, :])
#  ax2.plot(apr, tnomagerr[t, :])
#  plt.show()
#[plt.errorbar(times, tnomag[:, j], tnomagerr[:, j]) for j in np.arange(napr)]
#[plt.errorbar(times, tnomag[:, j] - avmag[:, j],
#              tnomagerr[:, j] + scaterr[:, j]) for j in np.arange(napr)]
plt.errorbar(mjd, tnomag[:], tnomagerr[:])
plt.errorbar(mjd, tnomag[:] - avmag[:], tnomagerr[:] + scaterr[:, 0])

#apcor = np.array([avmag[t, :]-avmag[t, -1] for t, time in enumerate(times)])
#[plt.errorbar(times, tnomag[:, j], tnomagerr[:, j]) for j in np.arange(napr)]
#[plt.errorbar(times, avmag[:, j], scaterr[:, j]) for j in np.arange(napr)]
#[plt.errorbar(times, 12+tnomag[:, j] - avmag[:, j],
#              tnomagerr[:, j] + scaterr[:, j]) for j in np.arange(napr)]
plt.errorbar(mjd, tnomag[:], tnomagerr[:])
plt.errorbar(mjd, avmag[:], scaterr[:, 0])
plt.errorbar(mjd, 12 + tnomag[:] - avmag[:], tnomagerr[:] + scaterr[:, 0])
#correltnomag = np.array([(tnomag-avmag)[t, np.argmin(tnomagerr[t, :])]
#                         for t, time in enumerate(times)])
correltnomag = np.array([(tnomag - avmag)[t] for t, time in enumerate(mjd)])
#plt.errorbar(times, 10+correltnomag, tnomagerr[:, 0]+scaterr[:, 0], lw = 1,
#             capsize = 20, elinewidth = 3)
plt.errorbar(mjd, 10 + correltnomag, tnomagerr[:] + scaterr[:, 0], lw=1,
             capsize=20, elinewidth=3)

plt.figure()
#[plt.errorbar(times, rmag[:, i, -1], magerr[:, i, -1]+scaterr[:, 0],
#               lw=useobj[i]) for i in objects]
#plt.errorbar(times, correltnomag, tnomagerr[:, 5]+scaterr[:, 0], lw=1,
#             capsize=20, elinewidth=3)
[plt.errorbar(mjd, rmag[:, i], magerr[i, :] + scaterr[:, 0], lw=useobj[i]
              ) for i in objects[useobj]]
plt.errorbar(mjd, correltnomag, tnomagerr[:] + scaterr[:, 0], lw=1,
             capsize=20, elinewidth=3)

plt.show()

plt.figure()
trippyerr = (tnomagerr ** 2 + scaterr[:, 0] ** 2) ** 0.5
trippymag = correltnomag
#sexmag=np.array([25.0304, 24.8231, 24.8566, 24.8246, 24.8735, 24.6488,
#                 24.5450, 24.8227, 24.4284, 24.4142])
#sexerr=np.array([0.1571, 0.1159, 0.1089, 0.0812, 0.0804, 0.0753, 0.0800,
#                 0.1396, 0.0605, 0.0600])
#plt.errorbar(mjd, sexmag-np.nanmean(sexmag), sexerr, lw=1, capsize=20,
#             elinewidth=3, label='SExtractor')
plt.errorbar(mjd, trippymag - np.mean(trippymag), trippyerr, marker='+', lw=0,
             capsize=20, elinewidth=3, label='TRIPPy')
plt.gca().invert_yaxis()
plt.legend(loc='best')
plt.xlabel('Time (MJD)')
plt.ylabel(r'$\Delta \mathrm{mag}$ (mag-mean)')
plt.show()

calmagfil = open('calibratedmags.txt', 'w')
print "#Odo   mjd              magnitude        dmagnitude"
calmagfil.write("#Odo   mjd              magnitude        dmagnitude\n")
for t, infile in enumerate(files):
  print "{0:13s} {1:16.10f} ".format(infile, mjd[t]) + \
        "{0:16.13f} {1:16.13f}".format((trippymag - np.mean(trippymag))[t],
                                       trippyerr[t])
  calmagfil.write("{0:13s} {1:16.10f} {2:16.13f} {3:16.13f}\n"
                  .format(infile, mjd[t], (trippymag - np.mean(trippymag))[t],
                          trippyerr[t]))
calmagfil.close()

calstarfil = open('calibrationstars.txt', 'w')
print "#xcoo             ycoo            mag               dmag"
calstarfil.write("#xcoo             ycoo            mag               dmag\n")
for i, j in enumerate(objects[useobj]):
  print "{0:16.11f} {1:16.11f} ".format(xcat[j], ycat[j]) + \
        "{0:16.13f} {1:16.13f}".format(np.mean(mag[j]),
                                       np.sum(magerr[j] ** 2) ** 0.5)
  calstarfil.write("{0:16.11f} {1:16.11f} {2:16.13f} {3:16.13f}\n"
                   .format(xcat[j], ycat[j], np.mean(mag[j]),
                           np.sum(magerr[j] ** 2) ** 0.5))
calstarfil.close()
