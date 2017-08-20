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
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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


def print_tno_file(calmagfile, odometer, julian, corrected, error, magzero):
  '''
  Write a file with the calibrated TNO magnitudes.
  '''
  calfile = open(calmagfile, 'w')
  print "#Odo          mjd              magnitude        " + \
        "dmagnitude       zeropoint"
  calfile.write("#Odo          mjd              magnitude        " +
                "dmagnitude       zeropoint\n")
  for j, odo in enumerate(odometer):
    print "{0:13s} {1:16.10f} ".format(odo, julian[j]) + \
          "{0:16.13f} {1:16.13f} ".format(corrected[j], error[j]) + \
          "{0:8.5f}".format(magzero[j])
    calfile.write("{0:13s} {1:16.10f} {2:16.13f} {3:16.13f} {4:8.5f}\n"
                  .format(odo, julian[j], corrected[j], error[j], magzero[j]))
  calfile.close()


def readzeropoint(filename):
  '''
  Reads the zeropoint of a file.
  '''
  hdulist = fits.open(filename)
  magzero = hdulist[0].header['MAGZERO']
  hdulist.close()
  return magzero


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
xobj, yobj, magobj, magerrobj, rmagobj = np.zeros([5, ntimes])
avmagobj, avmagerrobj = np.zeros([2, ntimes])

'''
Read all the info in the .trippy files, and get the zero points
that should have been used (instead of 26) from the fits files.
'''
for t, infile in enumerate(files):
  print infile
  (xobj[t], yobj[t], magobj[t], magerrobj[t], xcoo[t], ycoo[t],
   magin[t], magerrin[t], mjd[t]) = readtrippyfile(infile)
  zeros[t] = readzeropoint(infile[:-7] + '.fits')

'''
Calculate the zero-point offset required for callibration
'''
dzero = zeros - 26  # So real mag = measured + dzero # 26 because I am stupid


'''
Now read in measured object and correct it.
'''
tnomag, tnomagerr = magobj.copy(), magerrobj.copy()
plt.errorbar(mjd, tnomag[:], tnomagerr[:])
plt.plot(mjd, zeros[:], '+')
tnomag_corrected = tnomag + dzero
plt.errorbar(mjd, tnomag_corrected, tnomagerr[:],
             lw=1, capsize=20, elinewidth=3)
plt.gca().invert_yaxis()
plt.show()

print_tno_file('calibratedmags.txt', files, mjd, tnomag_corrected,
               tnomagerr, zeros)

# The end.
