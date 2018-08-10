"""
This function reads in the wcs header of a file,
then converts an x-y pixel coordinate to a RA and Dec.
"""
import datetime
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
#import astropy.io.fits as pyf
from astropy.time import Time
from astropy import wcs

def pix2world(fitsfile, expt, mjd, mg, xc, yc, fil, extn, observatory=568): 
  """
  This function takes variables (given in maphot.py),
  reads in the wcs header of a file,
  then converts an x-y pixel coordinate to a RA and Dec.
  Also converts the time of observation to an MPC friendly format.
  Returns an MPC formatted line.
  """
  aEXPTIME = expt #EXPTIME
  aMJD = mjd #MJD
  mag = mg #finalTNOphotPS1[0]
  xcoo = xc #xUse
  ycoo = yc #yUse
  filter = fil #FILTER
  name = 'extno{0:02.0f}'.format(extn)
  #Get WCS and convert pixel coordinates to world coordinates
  hdulist = fits.open(fitsfile)
  #WCS = wcs.WCS(hdulist[0].header)
  WCS = wcs.WCS(hdulist[extn].header)
  ra, dec = WCS.all_pix2world(xcoo, ycoo, 1)
  #Convert the ra/dec (which is in degrees) to sexagesimals
  raString = Angle(str(ra) + 'degrees').to_string(unit=u.hour,
                                                  sep=' ', pad=True)
  decString = Angle(str(dec) + 'degrees').to_string(unit=u.degree, sep=' ',
                                                    pad=True, alwayssign=True)
  #Calculate time of the middle of the image & convert to year month day.dd
  mT = (Time(aMJD, format='mjd').datetime +
        datetime.timedelta(seconds=aEXPTIME / 2.))
  decimalday = (mT.day + (mT.hour + (mT.minute +
                (mT.second + mT.microsecond / 1.e6) / 60.) / 60.) / 24.)
  midTimeString = '{0:4.0f} {1:02.0f} '.format(mT.year, mT.month) +\
                  '{0:08.5f}'.format(decimalday)
  #Put it all together in an MPC line
  MPCString = ('     ' + name + '  C' + midTimeString[:16] + ' ' +
               raString[:12] + decString[:12] +
               '         {0:4.1f} '.format(mag) + filter[:1] +
               '      {0:3.0f}'.format(observatory))
  wf = open("{}.mpc".format(name), "a+")
  wf.write("{}\n".format(MPCString))
  print(MPCString)
