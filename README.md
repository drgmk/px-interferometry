# px-interferometry
Third year Warwick Physics interferometry lab.

The lab script can be viewed [here](https://www.overleaf.com/read/dftrzzvfxsmb).

Analyse images with `python widget.py [-h] [-n bins] [-b] [-f] path/to/image.fits`

The `-n` argument is how much the input image is binned. 
More binning makes the code run faster, but too much binning can lose information.
Other options are to use a Bessell function for the PSF (`-b`),
and to show panels for the Fourier transform of the image (`-f`).
To get help include `-h`.

Upon running, a plot window appears, interact by moving mouse to an appropriate point
in one of the upper image windows and hit a key.

- c - set PSF center at cursor location
- g - set PSF width (Gaussian sigma) as cursor distance from center
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

### conda install

Install a conda env to run this with

`conda create -n pxlab -c conda-forge python matplotlib scipy numpy astropy emcee tqdm jupyter`

(for an M1/2 Mac add `CONDA_SUBDIR=osx-arm64` before `conda` and things will run a lot faster)
