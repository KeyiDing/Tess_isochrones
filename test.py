from astropy.io.fits import getdata
from glob import glob
from astropy.io import ascii as at
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table as tb
import os
import sys
from scipy.optimize import curve_fit as cf
from scipy.stats import gaussian_kde
from scipy import stats
import csv
import logging
import math
import multiprocessing
from multiprocessing import Pool
import pandas as pd
from scipy.interpolate import LinearNDInterpolator as interp

def run_isochrones():

    import isochrones
    from isochrones.mist import MIST_Isochrone
    from isochrones.mist.bc import MISTBolometricCorrectionGrid
    from isochrones import get_ichrone, SingleStarModel
    from isochrones.priors import FlatPrior, PowerLawPrior

    #I am attaching two support files that include all the data necessary to run the isochrones fitting
    #for a single star.
    file = at.read("./Join/lamost-isochrones.csv",delimiter=",",header_start=0)
    file.fill_value = -999

    #just a for loop to run it for all the stars in my file list
    length = len(file['source_id'])
    for i in range(0,15):
        """
        isochrones divides the parameters in two: One with details that are known about the star, such
        as temperature, gravity, or distance (the parallax), and extinction
        (tells you how much light of the star was absorbed or scattered before it got to earth).
        These info are below at the dictionary I am calling params_iso.
        """
        params_iso = {'parallax':(file['parallax'][i], file['parallax_error'][i])}

        #parallax cannot be negative
        if file['parallax'][i]<0:
            continue

        """
        The second set of data are the actual stellar magnitudes (or brightness) in
        each pass band. This is done in two ways. First you tell isochrones which
        bands it should look for, what is being done in the bands array, and then
        you actually tell isochrones the values and uncertainties of each pass band. This
        is done through a dictionary, which I am calling mags_iso below.
        """
        bands = ['Gaia_G_DR2Rev','2MASS_J','2MASS_H','2MASS_Ks','WISE_W1','WISE_W2']
        mags_iso = {'Gaia_G_DR2Rev':(file['phot_g_mean_mag'][i],0.3),\
          '2MASS_J':(file['j_m'][i],file['j_cmsig'][i]),'2MASS_H':(file['h_m'][i],file['h_cmsig'][i]),\
          '2MASS_Ks':(file['k_m'][i],file['k_cmsig'][i]),'WISE_W1':(file['w1mpro'][i],file['w1sigmpro'][i]),\
          'WISE_W2':(file['w2mpro'][i],file['w2sigmpro'][i])}
        """
        In this example all the stars have Gaia G, 2MASS J, H, and K, and Wise W1, W2, and W3
        pass band data. However some stars data for some other passbands. The if(s) below
        append the bands and dictionary with data for other pass bands when available. See the
        data in the file appended.
        """

        if file['nuvmag'][i]!=-999:
            bands.append('GALEX_NUV')
            mags_iso['GALEX_NUV'] = (file['nuvmag'][i],file['e_nuvmag'][i])

        # a flag variable that checks if SkyMapper U band exists
        u_flag = 0
        if file['u_psf'][i]!=-999:
            u_flag = 1
            bands.append('SkyMapper_u')
            mags_iso['SkyMapper_u'] = (file['u_psf'][i], file['e_u_psf'][i])
        else:
            u_flag = 0

        if u_flag == 1:
            if file['v_psf'][i]!=-999:
                bands.append('SkyMapper_v')
                mags_iso['SkyMapper_v'] = (file['v_psf'][i], file['e_v_psf'][i])
            if file['g_psf'][i]!=-999:
                bands.append('SkyMapper_g')
                mags_iso['SkyMapper_g'] = (file['g_psf'][i], file['e_g_psf'][i])
            if file['r_psf'][i]!=-999:
                bands.append('SkyMapper_r')
                mags_iso['SkyMapper_r'] = (file['r_psf'][i], file['e_r_psf'][i])
            if file['i_psf'][i]!=-999:
                bands.append('SkyMapper_i')
                mags_iso['SkyMapper_i'] = (file['i_psf'][i], file['e_i_psf'][i])
            if file['z_psf'][i]!=-999:
                bands.append('SkyMapper_z')
                mags_iso['SkyMapper_z'] = (file['z_psf'][i], file['e_z_psf'][i])

        elif u_flag == 0:
            if file['psfmag_u'][i]!=-999:
                bands.append('SDSS_u')
                mags_iso['SDSS_u'] = (file['psfmag_u'][i], file['psfmagerr_u'][i])
            # if file['psfmag_g'][i]!=-999:
            #     bands.append('SDSS_g')
            #     mags_iso['SDSS_g'] = (file['psfmag_g'][i], file['psfmagerr_g'][i])
            # if file['psfmag_r'][i]!=-999:
            #     bands.append('SDSS_r')
            #     mags_iso['SDSS_r'] = (file['psfmag_r'][i], file['psfmagerr_r'][i])
            # if file['psfmag_u'][i]!=-999:
            #     bands.append('SDSS_i')
            #     mags_iso['SDSS_i'] = (file['psfmag_i'][i], file['psfmagerr_i'][i])
            # if file['psfmag_z'][i]!=-999:
            #     bands.append('SDSS_z')
            #     mags_iso['SDSS_z'] = (file['psfmag_z'][i], file['psfmagerr_z'][i])

        print(params_iso)
        print(mags_iso)

        """
        get_ichrone and SingleStarModel commands gets the grid of models from the
        available grids with the specific pass bands you requested (get_ichrone)
        and SingleStarModel create an initial stellar model based on your parameters
        and the grid of isochrones from get_ichrone.
        """
        mist = get_ichrone('mist', basic=False, bands=bands)
        model1 = SingleStarModel(mist, **params_iso, **mags_iso)
        """
        Below we give the code the necessary information about these stars that it
        needs to run the Bayesian statistics.
        """
        #You set the information about possible composition.
        model1.set_prior(feh=FlatPrior((-2,0)), AV=PowerLawPrior(alpha=-2., bounds=(0.0001,1.0)))

        # model1.set_prior(feh=FlatPrior((-2, 0)))

        """
        you bound your grid to certain distances. The lower and upper limits of
        distances. Do not worry about the names of the columns (r_lo_photogeo)
        because it is just telling how the distance was calculated in the
        particular study they came from.
        """
        # model1._bounds['distance'] = (mags['r_lo_photogeo'][i]-20,mags['r_hi_photogeo'][i]+20)
        """
        We could bound the age and mass, but we will not do that in this example.
        """
        #model1._bounds['age'] = (np.log10(1.0e9),np.log10(13.721e9))
        #model1._bounds['mass'] = (0.1,2)
        """
        below are limits to the extinction and composition. We only bound the
        extinction, for which we have data.
        """
        #model1._bounds['AV'] = (0,.3) #
        #model1._bounds['feh'] = (params['feh'][idx][0] - params['err_feh'][idx][0], params['feh'][idx][0] + params['err_feh'][idx][0])
        #Runs and saves the results.
        model1.fit(refit=True,n_live_points=1000,evidence_tolerance=0.5)
        model1.derived_samples.to_csv("./isochrones/%s_take2.csv"%int(file['source_id'][i]), index_label='index')
        plot1 = model1.corner_observed()
        # plt.savefig("final_corner/corner_{}.png".format(int(file['source_id'][i])))
        plt.savefig("corner_{}.png".format(int(file['source_id'][i])))
        plot2 = model1.corner_physical()
        # plt.savefig("final_physical/physical_{}.png".format(int(file['source_id'][i])))
        plt.savefig("physical_{}.png".format(int(file['source_id'][i])))



def read_isochrones():
    """
    Open the files with the isochrones results and saves the median values,
    the percentiles (16th and 84th) for all information of interest.
    """
    params = at.read("./Join/lamost-isochrones.csv",delimiter=",",header_start=0)
    stars = params['source_id']
    folder = './isochrones/'
    f = open("iso_params_2.csv", "w")
    f.write("id,teff_16,teff,teff_84,e_teff,logg_16,logg,logg_84,e_logg,feh_16,feh,feh_84,e_feh,age_16,age,age_84,mass_16,mass,mass_84,radius_16,radius,radius_84,distance_16,distance,distance_84,luminosity_16,luminosity,luminosity_84,AV_16,AV,AV_84,nu_max_16,nu_max,nu_max_84,delta_nu_16,delta_nu,delta_nu_84\n")
    length = len(stars)
    for s in range(0,15):
        try:
            i=int(stars[s])
            data = pd.read_csv('./isochrones/%s_take2.csv'%(i))
            print(i)
            f.write("%s,%i,%i,%i,%i,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n"\
              %(i,np.quantile(data['Teff'],[0.16])-np.quantile(data['Teff'],[0.5]),np.quantile(data['Teff'],[0.5]),np.quantile(data['Teff'],[0.84])-np.quantile(data['Teff'],[0.5]),\
              np.sqrt((np.quantile(data['Teff'],[0.5]) - np.quantile(data['Teff'],[0.16]))**2. + (np.quantile(data['Teff'],[0.5]) - np.quantile(data['Teff'],[0.84]))**2.),\
              np.quantile(data['logg'],[0.16])-np.quantile(data['logg'],[0.5]),np.quantile(data['logg'],[0.5]),np.quantile(data['logg'],[0.84])-np.quantile(data['logg'],[0.5]),\
              np.sqrt((np.quantile(data['logg'],[0.5]) - np.quantile(data['logg'],[0.16]))**2. + (np.quantile(data['logg'],[0.5]) - np.quantile(data['logg'],[0.84]))**2.),\
              np.quantile(data['feh'],[0.16]),np.quantile(data['feh'],[0.5]),np.quantile(data['feh'],[0.84]),\
              np.sqrt((np.quantile(data['feh'],[0.5]) - np.quantile(data['feh'],[0.16]))**2. + (np.quantile(data['feh'],[0.5]) - np.quantile(data['feh'],[0.84]))**2.),\
              (10**np.quantile(data['age'],[0.16]))/10**9 - (10**np.quantile(data['age'],[0.5]))/10**9,(10**np.quantile(data['age'],[0.5]))/10**9,(10**np.quantile(data['age'],[0.84]))/10**9 - (10**np.quantile(data['age'],[0.5]))/10**9,\
              np.quantile(data['mass'],[0.16])-np.quantile(data['mass'],[0.5]),np.quantile(data['mass'],[0.5]),np.quantile(data['mass'],[0.84])-np.quantile(data['mass'],[0.5]),\
              np.quantile(data['radius'],[0.16])-np.quantile(data['radius'],[0.5]),np.quantile(data['radius'],[0.5]),np.quantile(data['radius'],[0.84])-np.quantile(data['radius'],[0.5]),\
              np.quantile(data['distance'],[0.16])-np.quantile(data['distance'],[0.5]),np.quantile(data['distance'],[0.5]),np.quantile(data['distance'],[0.84])-np.quantile(data['distance'],[0.5]),\
              10**np.quantile(data['logL'],[0.16])-10**np.quantile(data['logL'],[0.5]),10**np.quantile(data['logL'],[0.5]),10**np.quantile(data['logL'],[0.84])-10**np.quantile(data['logL'],[0.5]),\
              np.quantile(data['AV'],[0.16])-np.quantile(data['AV'],[0.5]),np.quantile(data['AV'],[0.5]),np.quantile(data['AV'],[0.84])-np.quantile(data['AV'],[0.5]), \
              np.quantile(data['nu_max'],[0.16])-np.quantile(data['nu_max'],[0.5]),np.quantile(data['nu_max'],[0.5]),np.quantile(data['nu_max'],[0.84])-np.quantile(data['nu_max'],[0.5]), \
              np.quantile(data['delta_nu'],[0.16])-np.quantile(data['delta_nu'],[0.5]),np.quantile(data['delta_nu'],[0.5]),np.quantile(data['delta_nu'],[0.84])-np.quantile(data['delta_nu'],[0.5])))
        except OSError:
            continue
    f.close()

def triangle_plots():
    plt.close('all')
    """
    A code to create the corner density plots.
    """
    ##### CORNER plots
    lista = glob("./isochrones/*_take2.csv")
    
    count = 0
    for k in lista:
        count += 1
        if count > 50:
            break
        name = k
        name = name.replace("_take2.csv","")
        name = name.replace("./isochrones/","")
        data = at.read(k, delimiter=",",header_start=0)
        data['distance'] = data['distance']/1000.
        data['age'] = 10**data['age']/(10**9)
        data['logL'] = 10**data['logL']
        #figure, axis = plt.subplots(7,7, gridspec_kw={'hspace': 0.1}, figsize=(15,15))
        figure, axis = plt.subplots(7,7, gridspec_kw={'hspace': 0.1}, figsize=(15,15))
        # Compute a histogram of the sample
        stuff = ['Teff','logg','distance','AV','mass','radius','age']
        label = [r'T$_{\rm{eff}}$ (K)',r'log$g$','Distance (kpc)',r'A$_V$ (mag)','Mass (M$_{\odot}$)',r'Radius (R$_{\odot}$)','Age (Gyr)']
        """
        stuff = ['Teff','logg','logL','mass','radius','age']
        label = [r'T$_{\rm{eff}}$ (K)',r'log$g$','L (L$_{\odot}$)','Mass (M$_{\odot}$)',r'Radius (R$_{\odot}$)','Age (Gyr)']
        """
        nbins = 120
        for i in range(0,len(stuff)):
            for j in range(0,i+1):
                if i==j:
                    bins = np.linspace(min(data['%s'%stuff[j]])-1, max(data['%s'%stuff[j]])+1, 100)
                    histogram, bins = np.histogram(data['%s'%stuff[j]], bins=bins, density=True)
                    histogram = histogram/max(histogram)
                    bin_centers = 0.5*(bins[1:] + bins[:-1])
                    axis[i,j].plot(bin_centers, histogram, color='red')
                    #axis[i,j].set_xticks([])
                    #axis[i,j].set_yticks([])
                else:
                    xy = np.vstack([data['%s'%stuff[j]],data['%s'%stuff[i]]])
                    k = gaussian_kde(xy)
                    #Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
                    xi, yi = np.mgrid[data['%s'%stuff[j]].min():data['%s'%stuff[j]].max():nbins*1j, data['%s'%stuff[i]].min():data['%s'%stuff[i]].max():nbins*1j]
                    zi = k(np.vstack([xi.flatten(),yi.flatten()]))
                    z_min, z_max = -np.abs(xy).max(), np.abs(xy).max()
                    axis[i,j].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='magma', rasterized=True)
                    axis[j,i].set_visible(False)
                if j==0:
                    axis[i,j].set_ylabel('%s'%label[i], fontsize=20)
                else:
                    axis[i,j].set_yticks([])
                if i==6:
                    axis[i,j].set_xlabel('%s'%label[j], fontsize=20)
                else:
                    axis[i,j].set_xticks([])
                print(stuff[i],stuff[j])
        plt.savefig("./plots/%s.pdf"%name)

