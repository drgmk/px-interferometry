#!python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from scipy.optimize import minimize
import scipy.ndimage
import scipy.special
from astropy.io import fits
import emcee

'''Script to fit visibilities in images.

Image has PSF and fringe components, fit both of these to derive
visibility. Minimisation doesn't work particularly well in a global
sense, so mouse/key interaction allows parameter specification for an
initial guess. Script takes file and rebin factor as arguments, e.g.

`python3 widget.py path/file.fits 2`

One plot window appears, interact by moving mouse to appropriate point
in image window (except rms) and hit a key.

c - set PSF center at cursor location
g - set PSF width as cursor distance from center
b - set background from cursor location
p - set image peak from cursor location (use brightest fringe)
t - set image trough from cursor location (use first dark fringe)
w - set fringe wavelength and angle from cursor distance/angle from center
f - set phase with peak at cursor
r - set rms from box around cursor location in residual image
m - minimise (walk downhill in chi^2)
M - minimise with Markov-Chain Monte Carlo (slower)
S - save parameters to numpy save file (file.fits -> file.npy)

The widget can also be loaded in a python terminal and run as a
function, e.g.

`import widget`
`fit_fringes('path/file.fits', sc=2)`

To run this script will likely require installing some python
modules. Assuming that you don't have the necessary permissions
to install these in the system, it will be best to use the --user
option, e.g

`pip install --user numpy scipy astropy emcee`

If you have problems with the keypress interaction, it may be
that changing the matplotlib backend helps. Try typing

`export MPLBACKEND=qtagg`

in the termnal before launching the widget, or some of the
other backends listed here
https://matplotlib.org/stable/users/explain/backends.html
If this works, you can make the change permanent by adding
a line to your matplotlibrc file.
'''

def fit_fringes(file, sc=1, fourier=False):

    # set output path, get image, and size
    if file is None:
        exit("need file name as first argument")
    path = os.path.dirname(file)
    paramfile = os.path.splitext(file)[0]+'-params.npy'
    im = fits.getdata(file)
    sz = im.shape

    # median filter out hot/cold pixels
    im[np.invert(np.isfinite(im))] == np.median(im)
    im = scipy.ndimage.median_filter(im,3)

    # rebin image down to speed things up, just chop some off if necessary
    im = im[:sc*(sz[0]//sc),:sc*(sz[1]//sc)]
    shape = np.array(im.shape)//sc
    sh = shape[0],im.shape[0]//shape[0],shape[1],im.shape[1]//shape[1]
    im = im.reshape(sh).mean(-1).mean(1)

    # account for rebinning effect on parameters
    def sc_par(par):
        x0,y0,sw,st,sp,sm,sv,gw,bg = par
        return [x0/sc,y0/sc,sw/sc,st,sp,sm,sv,gw/sc,bg]

    def unsc_par(par):
        x0,y0,sw,st,sp,sm,sv,gw,bg = par
        return [x0*sc,y0*sc,sw*sc,st,sp,sm,sv,gw*sc,bg]

    # open/set/guess pars now, will alter later
    if os.path.exists(paramfile):
        par = sc_par(np.load(paramfile))
        pt = [par[8]+par[5]*(par[6]+1),
              par[8]+par[5]*(-par[6]+1),
              1]
    else:
        bg = np.percentile(im, 5)

        peak = np.percentile(im, 99) - bg
        trough = np.percentile(im, 80) - bg

        x0 = 0.1
        y0 = 0.1
        sw = 10
        st = 0
        sp = 0
        sm = (peak+trough)/2
        sv = (peak-trough)/2 / sm
        gw = 20

        rms = 1
        pt = [peak, trough, rms]

        par = [x0,y0,sw,st,sp,sm,sv,gw,bg]

    # size/pixel arrays
    sz = im.shape
    xc = sz[1]/2 + 0.5
    yc = sz[0]/2 + 0.5
    x = np.arange(sz[1]) - xc
    y = np.arange(sz[0]) - yc

    def func(p):
        '''Return a model of the image

        Parameters are:
        x0 - x offset from image center
        y0 - y offset from image center
        sw - wavelength of fringe pattern in pixels
        st - angle from up of fringe pattern (baseline vector)
        sp - phase of fringe pattern
        sm - average flux of fringe pattern (minus background)
        sv - visibility of fringe pattern
        gw - scale of PSF (Airy or Gaussian function)
        bg - background level

        x0, y0, sw, st, sp, sm, sv, gw, bg
        0,  1,  2,  3,  4,  5,  6,  7,  8
        '''
        xx, yy = np.meshgrid(x-p[0], y-p[1])
        r = np.sqrt( xx**2 + yy**2 )
    #    psf = 1 * np.exp(-0.5 * (r/p[7])**2)
        psf = ( 2 * scipy.special.jv(1, r/p[7]) / (r/p[7]) )**2
        s2 = p[5]*( p[6]*np.cos(2*np.pi*(xx*np.sin(p[3]) + yy*np.cos(p[3]))/p[2] - p[4]) + 1 )
        return p[8] + s2 * psf

    def res(p):
        '''Residual image.'''
        return im - func(p)

    def chi2(p):
        '''Sum of squared residuals.'''
        return np.sum((res(p)/pt[2])**2)

    def lnlike(p):
        '''Log likelihood.'''
        return -0.5*chi2(p)

    # Start making the widget
    if fourier:
        fig, ax = plt.subplots(2,3, figsize=(18,9))
    else:
        fig, ax = plt.subplots(2,2, figsize=(12,9))

    def update_plot(par):
        '''Function to update plot.'''
    #    print(f'updating plot: {par}')
        for a in ax.flatten():
            a.clear()
        vmax = np.percentile(im,99.9)
        vmin = np.percentile(im,1)
        ax[0,0].imshow(im, vmin=vmin, vmax=vmax)
        ax[0,0].plot(xc+par[0], yc+par[1], '+', color='grey')
        ax[0,0].set_title('image')
        ax[0,1].imshow(func(par), vmin=vmin, vmax=vmax)
        ax[0,1].plot(xc+par[0], yc+par[1], '+', color='grey')
        ax[0,1].set_title('model')

        if fourier:
            # get model of just PSF
            par_tmp = par.copy()
            par_tmp[6] = 0
            par_tmp[8] = 0
            psf = func(par_tmp)
            
            # subtract this from data and normalise to peak
            sub = (im-par[8]) - psf
            sub = sub / np.max(sub)
            msub = func(par)-par[8]-psf
            msub = msub / np.max(msub)
            
            ft = np.fft.ifftshift(np.fft.ifft2(sub))
            aft = np.abs(ft)
            mft = np.fft.ifftshift(np.fft.ifft2(msub))
            amft = np.abs(mft)
            
#            ax[1,2].imshow(sub,
#                           vmin=np.percentile(sub,1), vmax=np.percentile(sub,99))
            ax[1,2].imshow(aft)
#            [sz[0]//2-30:sz[0]//2+30,sz[1]//2-45:sz[1]//2+45],
#                           vmin=np.percentile(aft,1), vmax=np.sort(aft.flatten())[-5])
            ax[1,2].set_title('FT$^{-1}$(data)')
            
            ax[0,2].imshow(amft)
#            [sz[0]//2-30:sz[0]//2+30,sz[1]//2-45:sz[1]//2+45],
#                           vmin=np.percentile(aft,1), vmax=np.sort(aft.flatten())[-5])
            ax[0,2].set_title('FT$^{-1}$(model)')

            mxy = np.unravel_index(np.argmax(amft), amft.shape)
            ftmax = aft[mxy]
            ax[1,2].plot(mxy[1], mxy[0], '+', label=f'peak:{ftmax:5.3f}')
            ax[1,2].legend()

        res_img = res(par)
        ax[1,0].imshow(res_img, vmin=np.percentile(res_img,5), vmax=np.percentile(res_img,95))
        ax[1,0].set_title(f'image-model, $\chi^2_r$:{chi2(par)/(sz[0]*sz[1]-len(par)-1):0.2f}')
        ang = -np.rad2deg(par[3])

        rot1 = scipy.ndimage.rotate(scipy.ndimage.shift(im,(-par[1],-par[0])), ang, reshape=False)
        rot2 = scipy.ndimage.rotate(scipy.ndimage.shift(func(par),(-par[1],-par[0])), ang, reshape=False)
        rot3 = scipy.ndimage.rotate(scipy.ndimage.shift(res(par),(-par[1],-par[0])), ang, reshape=False)
        line1 = np.median(rot1[:,int(xc)-10:int(xc)+10], axis=1)
        line2 = np.median(rot2[:,int(xc)-10:int(xc)+10], axis=1)
        line3 = np.median(rot3[:,int(xc)-10:int(xc)+10], axis=1)
        ax[1,1].plot(line1)
        ax[1,1].plot(line2)
        ax[1,1].plot(line3)
        ax[1,1].hlines(par[8], *ax[1,1].get_xlim(), color='grey', alpha=0.5)
        ax[1,1].text(0,par[8], 'background', color='grey', alpha=0.5)
        ax[1,1].hlines(par[8]+par[5]*(par[6]+1), *ax[1,1].get_xlim(), color='grey', alpha=0.5)
        ax[1,1].text(0,par[8]+par[5]*(par[6]+1), 'max', color='grey', alpha=0.5)
        ax[1,1].hlines(par[8]+par[5]*(-par[6]+1), *ax[1,1].get_xlim(), color='grey', alpha=0.5)
        ax[1,1].text(0,par[8]+par[5]*(-par[6]+1), 'min', color='grey', alpha=0.5)
        ax[1,1].set_xlim(0,len(line1))
        ax[1,1].text(len(line1)/2, par[8]/2, f'V={par[6]:0.2f}',
                     horizontalalignment='center')
        ax[1,1].text(len(line1)/2, par[8]/2-12, '$\\theta$={:0.0f}'.format(np.rad2deg(par[3])),
                     horizontalalignment='center')
        ax[1,1].set_title('line cut')
        ax[1,1].set_xlabel('pixels along baseline direction')
        ax[1,1].set_ylabel('counts')

        for a in ax.flatten()[:3]:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)

        fig.suptitle(file)
        fig.canvas.draw()

    def keypress(event):
        '''Deal with keyboard input.'''
        if event.key == 'c':
            par[0:2] = event.xdata-xc, event.ydata-yc

        if event.key == 'g':
            par[7] = np.sqrt( (event.xdata-xc)**2 + (event.ydata-yc)**2 )

        if event.key == 'p':
            data = im[int(event.ydata)-1:int(event.ydata)+2,
                      int(event.xdata)-1:int(event.xdata)+2]
            pt[0] = np.median(data) - par[8]
            par[5] = (pt[0]+pt[1])/2
            par[6] = (pt[0]-pt[1])/par[5]/2

        if event.key == 't':
            data = im[int(event.ydata)-1:int(event.ydata)+2,
                      int(event.xdata)-1:int(event.xdata)+2]
            pt[1] = np.median(data) - par[8]
            par[5] = (pt[0]+pt[1])/2
            par[6] = (pt[0]-pt[1])/par[5]/2

        if event.key == 'b':
            data = im[int(event.ydata)-1:int(event.ydata)+2,
                      int(event.xdata)-1:int(event.xdata)+2]
            par[8] = np.median(data)

        if event.key == 'r':
            resid = res(par)
            data = resid[int(event.ydata)-20:int(event.ydata)+21,
                         int(event.xdata)-20:int(event.xdata)+21]
            pt[2] = np.std(data)

        if event.key == 'w':
            x1 = par[0]+xc + 1j*(par[1]+yc)
            x = event.xdata + 1j*event.ydata
            par[3] = -np.angle(x - x1) + np.pi/2
            par[2] = np.abs(x - x1)

        if event.key == 'f':
            x1 = par[0]+xc + 1j*(par[1]+yc)
            x = event.xdata + 1j*event.ydata
            par[4] = 2 * np.pi * np.abs(x - x1) / par[2]

        if event.key == 'm':
            print(f'chi2: {chi2(par)}')
            r = minimize(chi2, par, method='Nelder-Mead',
                         options={'maxiter':10})
            par[:] = r['x']
            print(f'chi2: {chi2(par)}')

        if event.key == 'M':
            print(f'chi2: {chi2(par)}')
            nwalk = len(par)*2
            nstep = 50
            sampler = emcee.EnsembleSampler(18, len(par), lnlike)
            p0 = np.array(par)
            pos = [p0 + p0 * 0.01 * np.random.randn(len(par)) for i in range(nwalk)]
            pos, prob, state = sampler.run_mcmc(pos, nstep, progress=True)
            par[:] = sampler.flatchain[np.argmax(sampler.flatlnprobability), :]
            print(f'chi2: {chi2(par)}')

        if event.key == 'S':
            np.save(paramfile, unsc_par(par))

        if event.key == 'h':
            print("""
    c - set PSF center at cursor location
    g - set PSF width as cursor distance from center
    b - set background from cursor location
    p - set image peak from cursor location (use brightest fringe)
    t - set image trough from cursor location (use first dark fringe)
    w - set fringe wavelength and angle from cursor distance/angle from center
    f - set phase with peak at cursor
    r - set rms from box around cursor location in residual image
    m - minimise (walk downhill in chi^2)
    M - minimise with Markov-Chain Monte Carlo (slower)
    S - save parameters to numpy save file (file.fits -> file.npy)
    """)

        update_plot(par)

    fig.canvas.mpl_connect('key_press_event', keypress)

    update_plot(par)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.95, bottom=0.075,
                        hspace=0.15, wspace=0.2)
    plt.show()


if __name__ == "__main__":

    # open a file
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        exit('\nGive file path as first argument, '
             'rebin factor as optional second (default=1)\n'
             'e.g.> python widget.py dir/file.fits 2\n')

    sc = 1
    if len(sys.argv) > 2:
        sc = int(sys.argv[2])

    fit_fringes(file, sc=sc, fourier=True)
