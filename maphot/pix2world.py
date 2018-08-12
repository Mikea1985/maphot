"""
This function converts a bunch of inputs to a nicely formatted MPC line.
"""
from __future__ import print_function, division
import datetime
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time

def pix2MPC(WCS, aEXPTIME, aMJD, mag, xcoo, ycoo, filtr, extn,
            observatory=568):
  """
  This function takes variables (given in maphot.py),
  then converts an x-y pixel coordinate to a RA and Dec.
  Also converts the time of observation to an MPC friendly format.
  Returns an MPC formatted line.
  """
  name = 'extno{0:02.0f}'.format(extn)
  #Convert pixel coordinates to world coordinates
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
               '         {0:4.1f} '.format(mag) + filtr[:1] +
               '      {0:3.0f}'.format(observatory))
  wf = open("{}.mpc".format(name), "a+")
  wf.write("{}\n".format(MPCString))
  print(MPCString)
