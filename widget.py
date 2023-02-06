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
left/right - rotate angle by 1deg in either direction
up/down - increase/decrease fringe wavelength
a - set phase angle with peak at cursor
r - set rms from box around cursor location in residual image
m - minimise (walk downhill in chi^2)
M - minimise with Markov-Chain Monte Carlo (slower)
S - save parameters to numpy save file (file.fits -> file.npy)

The widget can also be loaded in a python terminal and run as a
function, where the fourier option is also available, e.g.

`import widget`
`fit_fringes('path/file.fits', sc=2, fourier=True)`

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

vis_save = 0 # to be filled later

def centroid(im, x0=0, y0=0):
    '''Return intensity weighted centroid of an image.'''
    yy,xx = np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[0]))
    xc = np.sum(xx * im) / np.sum(im)
    yc = np.sum(yy * im) / np.sum(im)
    cen = round(xc+x0), round(yc+y0)
    return cen


def centroid_sum(im, xy, dxy=1, r=2):
    '''Return centroided position and sum +/-dxy.'''
    cut = im[xy[0]-r:xy[0]+r+1, xy[1]-r:xy[1]+r+1]
    cen = centroid(cut, x0=xy[0]-r, y0=xy[1]-r)
    tot = np.sum(im[cen[0]-dxy:cen[0]+dxy+1, cen[1]-dxy:cen[1]+dxy+1])
    return cen, tot


def vis_ft(im):
    '''Return visibility, wavelength, and orientation from image.'''
    sz = im.shape
    ft = np.fft.fftshift(np.fft.fft2(im))
    aft = np.abs(ft)
    aft /= np.max(aft)
    c = 3
    aft[sz[0]//2-c:sz[0]//2+c+1, sz[1]//2-c:sz[1]//2+c+1] = 0
    mxy = np.unravel_index(np.argmax(aft), aft.shape)
    mxy, vis = centroid_sum(aft, mxy, dxy=0)
    vis = 2 * vis # 2 is n_holes, see Tuthill thesis
    x = mxy[1] - (aft.shape[1]-1) / 2
    y = mxy[0] - (aft.shape[0]-1) / 2
    ang = np.arctan2(y, x) + np.pi/2
    wav = sz[0] / np.sqrt(x*x + y*y)
    print(mxy,vis,ang,wav)
    return mxy, vis, ang, wav


def estimate_par(im):
    '''Automatic parameter estimation.'''
    sz = im.shape
    xc = sz[1]/2 + 0.5
    yc = sz[0]/2 + 0.5

    # peak smoothed pixel as center
    sm = scipy.ndimage.gaussian_filter(im, sz[0]//10)
    pk = np.argmax(sm)
    x0 = np.unravel_index(pk, im.shape)[1] - xc - 0.0001
    y0 = np.unravel_index(pk, im.shape)[0] - yc - 0.0001
    
    # find orientation and wavelength
    mxy, vis, st, sw = vis_ft(im)

    bg = np.percentile(im, 5)
    peak = np.percentile(im, 99.9) - bg
    trough = np.percentile(im, 95) - bg

    sp = 0
    sm = (peak+trough)/2
    sv = (peak-trough)/2 / sm
    gw = 110

    rms = 1
    pt = [peak, trough, rms]

    par = [x0,y0,sw,st,sp,sm,sv,gw,bg]
    return par, pt


def zero_pad(im, n):
    '''Zero pad an array by twice an amount in each dimension.'''
    if n == 0:
        return im
    sz = im.shape
    new = np.zeros((sz[0]+2*n, sz[1]+2*n))
    new[n+1:n+1+sz[0], n+1:n+1+sz[1]] = im
    return new
    

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

    # size/pixel arrays
    sz = im.shape
    xc = sz[1]/2 + 0.5
    yc = sz[0]/2 + 0.5
    x = np.arange(sz[1]) - xc
    y = np.arange(sz[0]) - yc

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
        par, pt = estimate_par(im)
        par[7] /= sc


    def fringes(p, xx, yy, dw=0, nw=5):
        '''Function to make fringes, optional bandwidth.'''
        if dw == 0:
            return p[5]*( p[6]*np.cos(2*np.pi*(xx*np.sin(-p[3]) + \
                                               yy*np.cos(-p[3]))/p[2] - p[4]) + 1 )

        else:
            im = np.zeros_like(xx)
            ds = np.linspace(1-dw, 1+dw, nw)
            for d in ds:
                im += p[5]*( p[6]*np.cos(2*np.pi*(xx*np.sin(-p[3]) + \
                                               yy*np.cos(-p[3]))/p[2]/d - p[4]) + 1 )

            return im/nw

    def func(p):
        '''Return a model of the image

        Parameters are:
        x0 - x offset from image center
        y0 - y offset from image center
        sw - wavelength of fringe pattern in pixels
        st - acw angle from up of fringe pattern (baseline vector)
        sp - phase of fringe pattern
        sm - average flux of fringe pattern (minus background)
        sv - visibility of fringe pattern
        gw - scale of PSF (Airy or Gaussian function)
        bg - background level

        x0, y0, sw, st, sp, sm, sv, gw, bg
        0,  1,  2,  3,  4,  5,  6,  7,  8
        
        Numpy exp for Gaussian PSF is ~10x faster than scipy Bessell.
        '''
        xx, yy = np.meshgrid(x-p[0], y-p[1])
        r = np.sqrt( xx**2 + yy**2 )
        
        psf = 1 * np.exp(-0.5 * (r/p[7]/1.3)**2) # factor 1.3 to make equiv. to Bessell
#        psf = ( 2 * scipy.special.jv(1, r/p[7]) / (r/p[7]) )**2
        
        s2 = fringes(p, xx, yy, dw=0.05) # 0.05 is an estimate, could be a parameter
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
        ax[0,0].imshow(im, origin='lower', vmin=vmin, vmax=vmax)
        ax[0,0].plot(xc+par[0], yc+par[1], '+', color='grey')
        ax[0,0].set_title('image')
        ax[0,0].set_xlabel('pixel')
        ax[0,0].set_ylabel('pixel')
        ax[0,1].imshow(func(par), origin='lower', vmin=vmin, vmax=vmax)
        ax[0,1].plot(xc+par[0], yc+par[1], '+', color='grey')
        ax[0,1].set_title('model')
        ax[0,1].set_xlabel('pixel')
        ax[0,1].set_ylabel('pixel')

        if fourier:
            # get model without PSF or background
            par_tmp = par.copy()
            par_tmp[7] = 1e9
            par_tmp[8] = 0
            mod = func(par_tmp)
            mod -= np.mean(mod)
            mod2 = func(par) - par[8]
            
            # subtract background from data
            sub = im - par[8]

            # zero pad images, this will change the FT a bit
            pad = 0
            sub = zero_pad(sub, pad)
            mod = zero_pad(mod, pad)
            mod2 = zero_pad(mod2, pad)

            # compute FTs and normalise to peak
            ft = np.fft.fftshift(np.fft.fft2(sub))
            aft =  np.abs(ft)
            aft /= np.max(aft)
            mft = np.fft.fftshift(np.fft.fft2(mod))
            amft = np.abs(mft)
            amft /= np.max(amft)
            mft2 = np.fft.fftshift(np.fft.fft2(mod2))
            amft2 = np.abs(mft2)
            amft2 /= np.max(amft2)

            # show FT of the data,
            # colour scale ignores the brightest pixel
            ax[1,2].imshow(aft, origin='lower',
                           vmin=np.percentile(aft,1), vmax=np.sort(aft.flatten())[-2])
            ax[1,2].set_title('FT(data)')
            ax[1,2].set_xlabel('pixel')
            ax[1,2].set_ylabel('pixel')

            ax[0,2].imshow(sub, origin='lower',
                           vmin=np.percentile(sub,1), vmax=np.percentile(sub,99.5))
            ax[0,2].set_title('image for FT')
            ax[0,2].set_xlabel('pixel')
            ax[0,2].set_ylabel('pixel')

            # get peak in FT from model and get visibility
            mxy = np.unravel_index(np.argmax(amft), amft.shape)
            mxy = mxy[0], mxy[1]
            vis_mod = amft2[mxy] * 2
            mxy, tot = centroid_sum(aft, mxy, dxy=0)
            vis = tot * 2
            global vis_save
            vis_save = vis.copy()
#            mxy, vis, ang, wav = vis_ft(im)
            ax[1,2].plot(mxy[1], mxy[0], '+', label=f'vis:{vis:5.3f}\n(model:{vis_mod:5.3f})')
            ax[1,2].legend()
            
            # zoom the FT a bit
            cx = (aft.shape[1]-1) / 2
            cy = (aft.shape[0]-1) / 2
            n = 30 #np.max([np.abs(mxy[0]-cy), np.abs(mxy[1]-cx)])
            ax[1,2].set_xlim(cx-2*n,cx+2*n)
            ax[1,2].set_ylim(cy-2*n,cy+2*n)

        res_img = res(par)
        ax[1,0].imshow(res_img, origin='lower',
                       vmin=np.percentile(res_img,5), vmax=np.percentile(res_img,95))
        ax[1,0].set_title(f'image-model, $\chi^2_r$:{chi2(par)/(sz[0]*sz[1]-len(par)-1):0.2f}')
        ax[1,0].set_xlabel('pixel')
        ax[1,0].set_ylabel('pixel')

        ang = np.rad2deg(par[3])

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
        ax[1,1].text(len(line1)/2, par[8]/2-12, '$\\phi$={:0.0f}'.format(np.rad2deg(par[3])),
                     horizontalalignment='center')
        ax[1,1].set_title('line cut')
        ax[1,1].set_xlabel('pixels along baseline direction')
        ax[1,1].set_ylabel('counts')

        fig.suptitle(file)
        fig.canvas.draw()

    def keypress(event):
        '''Deal with keyboard input.'''
        if event.key == 'A':
            # intended for automatic parameter estimation

            # peak smoothed pixel as center
            sm = scipy.ndimage.gaussian_filter(im, 40)
            pk = np.argmax(sm)
            par[0] = np.unravel_index(pk, im.shape)[1] - xc - 0.0001
            par[1] = np.unravel_index(pk, im.shape)[0] - yc - 0.0001
            
            # find orientation and wavelength
            mxy, vis, par[3], par[2] = vis_ft(im)

        if event.key == 'c':
            par[0:2] = event.xdata-xc, event.ydata-yc

        if event.key == 'g':
            par[7] = np.sqrt( (event.xdata-par[0]-xc)**2 + (event.ydata-par[1]-yc)**2 )

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
            par[3] = np.angle(x - x1) + np.pi/2
            par[2] = np.abs(x - x1)

        if event.key == 'left':
            par[3] = par[3] + np.deg2rad(1)

        if event.key == 'right':
            par[3] = par[3] - np.deg2rad(1)

        if event.key == 'up':
            par[2] = par[2] + 1

        if event.key == 'down':
            par[2] = par[2] - 1

        if event.key == 'a':
            x1 = par[0]+xc + 1j*(par[1]+yc)
            x = event.xdata + 1j*event.ydata
            par[4] = 2 * np.pi * np.abs(x - x1) / par[2]

        if event.key == 'm':
            print(f'chi2: {chi2(par)}')
            r = minimize(chi2, par, method='Nelder-Mead',
                         options={'maxiter':20})
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
            if fourier:
                global vis_save
                np.save(paramfile.replace('-params','-FTparams'), vis_save)

        if event.key == 'h':
            print("""
    c - set PSF center at cursor location
    g - set PSF width as cursor distance from center
    b - set background from cursor location
    p - set image peak from cursor location (use brightest fringe)
    t - set image trough from cursor location (use first dark fringe)
    w - set fringe wavelength and angle from cursor distance/angle from center
    left/right - rotate angle by 1deg in either direction
    up/down - increase/decrease fringe wavelength
    a - set phase with peak at cursor
    r - set rms from box around cursor location in residual image
    m - minimise (walk downhill in chi^2)
    M - minimise with Markov-Chain Monte Carlo (slower)
    S - save parameters to numpy save file (file.fits -> file.npy)
    """)

        update_plot(par)

    fig.canvas.mpl_connect('key_press_event', keypress)

    update_plot(par)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.925, bottom=0.075,
                        hspace=0.2, wspace=0.15)
    plt.show()


if __name__ == "__main__":

    # open a file
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        exit('\nGive file path as first argument, '
             'rebin factor as optional second (default=1)\n'
             'e.g.> python widget.py dir/file.fits 2\n'
             'any third argument will turn Fourier on.\n')

    sc = 1
    if len(sys.argv) > 2:
        sc = int(sys.argv[2])

    fourier = False
    if len(sys.argv) > 3:
        fourier = True

    fit_fringes(file, sc=sc, fourier=fourier)
