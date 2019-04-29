# maphot
My wrapper for using [``TRIPPy``](https://github.com/fraserw/trippy) to do photometry of the same object in many
images of the same field.

#Warnings:
- Work in progress. 
- Only ever used by three people (under my supervision), so I'm sure it's full of bugs. 
- Definitely has hard-coded (Bad Mike, bad!) parts that makes it unlikely to work correctly with telescopes/cameras other than CFHT/MegaPrime, Gemini-N/GMOS-N, Subaru/HSC.
- Probably not suitable for use by anybody that hasn't at least consulted me first. If you don't want to talk to me, I strongly urge you learn how to use ``TRIPPy`` and make your own wrapper script optimised for your own data. 

# ``best``
The ``best`` module that runs Source Extracter for all images in a set, 
identifies sources that are present in all images, lets the user inspect and
filter those sources (using ``TrIPPy``'s ``psfStarChooser`` tool) and then
returns a Source Extractor-like catalog. 
Having this "best" catalog speeds up the process of doing the photometry, as only a small catalog with known good sources are being used. 

# ``maphot_functions``
The intention of the ``maphot_functions`` module is to contain all the functions that are used by both ``best`` and ``maphot``, although I think a few that are only used by one or the other has snuck in.

# ``maphot``
The ``maphot`` module is the main body, which loops over every image in a set,
performing photometry using the ``TRIPPy`` package of both the TNO and of all
stars in the input catalog.

# ``photcor`` (DEPRECATED?)
This module is deprecated, but is left in in case parts of it may become useful in the future. 
The ``photcor`` module reads the star magnitudes measured in each image, 
identifies stars that were measured in all images, displays the magnitudes as a
function of image number to the user and aids the user to reject non-constant 
stars (stars that don't follow the median trend) and then uses the median trend
of these good stars to remove that trend from all star and TNO photometry, 
resulting in (on average) constant stars and well-calibrated TNO magnitudes.
