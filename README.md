# px-interferometry
Third year Warwick Physics interferometry lab.

The lab script can be viewed [here](https://www.overleaf.com/read/dftrzzvfxsmb).

Analyse images with `python widget.py path/to/file.fits 2`

The extra argument (`2` above) is how much the input image is binned. 
More binning makes the code run faster, but too much binning can lose information.

Upon running, a plot window appears, interact by moving mouse to appropriate point
in image window (except rms) and hit a key.

- c - set PSF center at cursor location
- g - set PSF width as cursor distance from center
- b - set background from cursor location
- p - set image peak from cursor location (use brightest fringe)
- t - set image trough from cursor location (use first dark fringe)
- w - set fringe wavelength and angle from cursor distance/angle from center
- left/right - rotate angle by 1deg in either direction
- up/down - increase/decrease fringe wavelength
- a - set phase angle with peak at cursor
- r - set rms from box around cursor location in residual image
- m - minimise (walk downhill in chi^2)
- M - minimise with Markov-Chain Monte Carlo (slower)
- S - save parameters to numpy save file (file.fits -> file.npy)
