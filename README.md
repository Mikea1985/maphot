# maphot
My wrapper for using ``TrIPPy`` to do photometry of the same object in many
images of the same field.

#``best``
The ``best`` module that runs Source Extracter for all images in a set, 
identifies sources that are present in all images, lets the user inspect and
filter those sources (using ``TrIPPy``'s ``psfStarChooser`` tool) and then
returns a Source Extractor-like catalog. 
Having this "best" catalog speeds up the process of doing the photometry, as only a small catalog with known good sources are being used. 

``maphot``
The ``maphot`` module is the main body, which loops over every image in a set,
performing photometry using the ``TrIPPy`` package of both the TNO and of all
stars in the input catalog.

#``photcor``
The ``photcory`` module reads the star magnitudes measured in each image, 
identifies stars that were measured in all images, displays the magnitudes as a
function of image number to the user and aids the user to reject non-constant 
stars (stars that don't follow the median trend) and then uses the median trend
of these good stars to remove that trend from all star and TNO photometry, 
resulting in (on average) constant stars and well-calibrated TNO magnitudes.

#``fixzero``
I think fixzero is not relevant anymore and can probably be deleted. It
originates from a time when I had hardcoded the zeropoint to be 26 rather than
using the zeropoint in the image headers, which is of course stupid. 

#Warnings:
Work in progress. 
Only ever used by one person, so I'm sure it's full of bugs. 
This code has only been tested on images from Hyper Suprime-Cam on Subaru. 
Currently assumes that images are aligned.
Currently requires image zero-points to already be callibrated.
