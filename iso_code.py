from astropy.table import Table as tb
import numpy as np
import pandas as pd
import isochrones
from isochrones.mist import MIST_Isochrone
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from isochrones import get_ichrone, SingleStarModel
from isochrones.priors import FlatPrior, PowerLawPrior
import matplotlib.pyplot as plt
from astropy.io import ascii as at
import random
import math
import os


def run_isochrones(row,name):
    #row is now the name of the row we are working on
    """
    here there are two alternatives:
    Keep working with the same data structure you had before (i. e. an Astropy Table)
    or use the pandas data structure. I will exemplify both

    First keeping the Tables structure:To do this you must transform the pandas
    structure into an astropy Table: (see packages imported above)
    """
    # row = tb.from_pandas(row)
    """
    If you choose this your code will keep the same structure but all the [0]]
    should be changed to 0:
    """
    # params_iso = {'parallax':(row['parallax'][0], row['parallax_error'][0])}

    """
    the second alternative is to keep the pandas dataframe structue. In this
    option be aware that getting a single value from a pandas dataframe is a bit different.
    (notice that I am using the .values[0] command because I also know there is
    only one row in my data!)
    example:
    """
    params_iso = {'parallax':(row['parallax'].values[0], row['parallax_unc'].values[0])}

    """
    Please change the remaining of the code accoring to your choice. Memory wise
    pandas is generally better, but because we are dealing in this section with a single row it
    should not matter
    """
    #parallax cannot be negative; if so, just stop
    if row['parallax'].values[0]<0:
        return

    """
    The second set of data are the actual stellar magnitudes (or brightness) in
    each pass band. This is done in two ways. First you tell isochrones which
    bands it should look for, what is being done in the bands array, and then
    you actually tell isochrones the values and uncertainties of each pass band. This
    is done through a dictionary, which I am calling mags_iso below.
    """
    # count how many Photometry are available(this is a parameter that helps me with the logic I use
    # to decide which photometry to use)
    count = 0

    bands = ['Gaia_G_DR2Rev']
    mags_iso = {'Gaia_G_DR2Rev':(row['gaia_g'].values[0],row['gaia_g_unc'].values[0])}

    #Change the following section based on your need. In my implementation, I check the availability of
    #each photometry, and append them based on the logic of choosing photometry
    #To add a photometry, append bands array with the band name, and add the data to mags_iso dictionary

    # if GALEX FUV/NUV are available, then always include them
    if row['galex_fuv'].isna().values[0]==False:
        count += 1
        bands.append('GALEX_FUV')
        mags_iso['GALEX_FUV'] = (row['galex_fuv'].values[0],row['galex_fuv_unc'].values[0])

    if row['galex_nuv'].isna().values[0]==False:
        count += 1
        bands.append('GALEX_NUV')
        mags_iso['GALEX_NUV'] = (row['galex_nuv'].values[0],row['galex_nuv'].values[0])

    # a flag variable that checks if SkyMapper U band exists
    #if only SkyMapper u-band data available, then use SkyMapper
    skymapper_flag = 0
    if row['skymapper_u'].isna().values[0]==False:
        skymapper_flag = 1
        bands.append('SkyMapper_u')
        mags_iso['SkyMapper_u'] = (row['skymapper_u'].values[0], row['skymapper_u_unc'].values[0])

    if skymapper_flag == 1:
        count += 1
        if row['skymapper_v'].isna().values[0]==False:
            bands.append('SkyMapper_v')
            mags_iso['SkyMapper_v'] = (row['skymapper_v'].values[0], row['skymapper_v_unc'].values[0])
        if row['skymapper_g'].isna().values[0]==False:
            bands.append('SkyMapper_g')
            mags_iso['SkyMapper_g'] = (row['skymapper_g'].values[0], row['skymapper_g_unc'].values[0])
        if row['skymapper_r'].isna().values[0]==False:
            bands.append('SkyMapper_r')
            mags_iso['SkyMapper_r'] = (row['skymapper_r'].values[0], row['skymapper_r_unc'].values[0])
        if row['skymapper_i'].isna().values[0]==False:
            bands.append('SkyMapper_i')
            mags_iso['SkyMapper_i'] = (row['skymapper_i'].values[0], row['skymapper_i_unc'].values[0])
        if row['skymapper_z'].isna().values[0]==False:
            bands.append('SkyMapper_z')
            mags_iso['SkyMapper_z'] = (row['skymapper_z'].values[0], row['skymapper_z_unc'].values[0])

    sdss_flag = 0
    #  if only SDSS u-band data available, then use SDSS
    if row['sdss_u'].isna().values[0]==False:
        sdss_flag = 1
        bands.append('SDSS_u')
        mags_iso['SDSS_u'] = (row['sdss_u'].values[0], row['sdss_u_unc'].values[0])

    if sdss_flag == 1:
        count += 1
        if row['sdss_g'].isna().values[0]==False:
            bands.append('SDSS_g')
            mags_iso['SDSS_g'] = (row['sdss_g'].values[0], row['sdss_g_unc'].values[0])
        if row['sdss_r'].isna().values[0]==False:
            bands.append('SDSS_r')
            mags_iso['SDSS_r'] = (row['sdss_r'].values[0], row['sdss_r_unc'].values[0])
        if row['sdss_i'].isna().values[0]==False:
            bands.append('SDSS_i')
            mags_iso['SDSS_i'] = (row['sdss_i'].values[0], row['sdss_i_unc'].values[0])
        if row['sdss_z'].isna().values[0]==False:
            bands.append('SDSS_z')
            mags_iso['SDSS_z'] = (row['sdss_z'].values[0], row['sdss_z_unc'].values[0])

    # if neither SkyMapper or SDSS u-band data is available, then use the survey with the most available bands
    if sdss_flag == 0 and skymapper_flag == 0:
        if row['skymapper_v'].isna().values[0]==False:
            count += 1
            bands.append('SkyMapper_v')
            mags_iso['SkyMapper_v'] = (row['skymapper_v'].values[0], row['skymapper_v_unc'].values[0])
        if row['skymapper_g'].isna().values[0]==False:
            count += 1
            bands.append('SkyMapper_g')
            mags_iso['SkyMapper_g'] = (row['skymapper_g'].values[0], row['skymapper_g_unc'].values[0])
        if row['skymapper_r'].isna().values[0]==False:
            count += 1
            bands.append('SkyMapper_r')
            mags_iso['SkyMapper_r'] = (row['skymapper_r'].values[0], row['skymapper_r_unc'].values[0])
        if row['skymapper_i'].isna().values[0]==False:
            count += 1
            bands.append('SkyMapper_i')
            mags_iso['SkyMapper_i'] = (row['skymapper_i'].values[0], row['skymapper_i_unc'].values[0])
        if row['skymapper_z'].isna().values[0]==False:
            count += 1
            bands.append('SkyMapper_z')
            mags_iso['SkyMapper_z'] = (row['skymapper_z'].values[0], row['skymapper_z_unc'].values[0])

        if row['sdss_g'].isna().values[0] == False:
            count += 1
            bands.append('SDSS_g')
            mags_iso['SDSS_g'] = (row['sdss_g'].values[0], row['sdss_g_unc'].values[0])
        if row['sdss_r'].isna().values[0] == False:
            count += 1
            bands.append('SDSS_r')
            mags_iso['SDSS_r'] = (row['sdss_r'].values[0], row['sdss_r_unc'].values[0])
        if row['sdss_i'].isna().values[0] == False:
            count += 1
            bands.append('SDSS_i')
            mags_iso['SDSS_i'] = (row['sdss_i'].values[0], row['sdss_i_unc'].values[0])
        if row['sdss_z'].isna().values[0] == False:
            count += 1
            bands.append('SDSS_z')
            mags_iso['SDSS_z'] = (row['sdss_z'].values[0], row['sdss_z_unc'].values[0])

    if row['tmass_j'].isna().values[0] ==  False:
        count += 1
        bands.append('2MASS_J')
        mags_iso['2MASS_J'] = (row['tmass_j'].values[0],row['tmass_j_unc'].values[0])

    if row['tmass_h'].isna().values[0] ==  False:
        count += 1
        bands.append('2MASS_H')
        mags_iso['2MASS_H'] = (row['tmass_h'].values[0],row['tmass_h_unc'].values[0])

    if row['tmass_k'].isna().values[0] ==  False:
        count += 1
        bands.append('2MASS_Ks')
        mags_iso['2MASS_Ks'] = (row['tmass_k'].values[0],row['tmass_k_unc'].values[0])

    if row['wise_w1'].isna().values[0] ==  False:
        count += 1
        bands.append('WISE_W1')
        mags_iso['WISE_W1'] = (row['wise_w1'].values[0],row['wise_w1_unc'].values[0])
    if row['wise_w2'].isna().values[0] ==  False:
        count += 1
        bands.append('WISE_W2')
        mags_iso['WISE_W2'] = (row['wise_w2'].values[0],row['wise_w2_unc'].values[0])


    # if nothing other than Gaia DR2 G-band data are available by this step,
    # then add Gaia DR2 G_BP and G_RP data with appropriate uncertainties
    if count == 0:
        if row['gaia_bp'].isna().values[0] == False:
            bands.append('Gaia_BP_DR2Rev')
            mags_iso['Gaia_BP_DR2Rev'] = (row['gaia_bp'].values[0],row['gaia_bp_unc'].values[0])
        if row['gaia_rp'].isna().values[0] == False:
            bands.append('Gaia_RP_DR2Rev')
            mags_iso['Gaia_RP_DR2Rev'] = (row['gaia_rp'].values[0],row['gaia_rp_unc'].values[0])

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
    # model1._bounds['distance'] = (mags['r_lo_photogeo'].values[0]]-20,mags['r_hi_photogeo'].values[0]]+20)
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
    #model1._bounds['feh'] = (params['feh'][idx].values[0] - params['err_feh'][idx].values[0], params['feh'][idx].values[0] + params['err_feh'][idx].values[0])

    #I create a directory based on the Gaia_dr2_source_id, and save both physical plot and observed plot to that directory
    if os.path.isdir("./{n}_plots/{id}".format(n=name,id=int(row['dr3_source_id'].values[0]))) == False:
        os.mkdir("./{n}_plots/{id}".format(n=name,id=int(row['dr3_source_id'].values[0])))
    #fit the model, max_iter is set to prevent multinest from starting from a "wrong" point so that it takes a long time and 
    #huge amount of memory to fit the parameters
    model1.fit(refit=True,n_live_points=1000,evidence_tolerance=0.5, max_iter=75000)
    
    #a "protection mechanism" that ensures the derived samples is large enough to produce the plots once max_iter is set
    #unlikely to reach this if statement
    if len(model1.derived_samples)<8 or len(model1.derived_samples)<len(bands):
        model1 = SingleStarModel(mist, **params_iso, **mags_iso)
        model1.set_prior(feh=FlatPrior((-2, 0)), AV=PowerLawPrior(alpha=-2., bounds=(0.0001, 1.0)))
        model1.fit(refit=True, n_live_points=1000, evidence_tolerance=0.5, max_iter=150000)
        if len(model1.derived_samples)<8 or len(model1.derived_samples)<len(bands):
            model1.fit(refit=True, n_live_points=1000, evidence_tolerance=0.5, max_iter=225000)
            model1.set_prior(feh=FlatPrior((-2, 0)), AV=PowerLawPrior(alpha=-2., bounds=(0.0001, 1.0)))
            model1.fit(refit=True, n_live_points=1000, evidence_tolerance=0.5, max_iter=150000)
            if len(model1.derived_samples) < 8 or len(model1.derived_samples)<len(bands):
                return

    #save the derived sample to a csv file
    model1.derived_samples.to_csv("{f}_isochrones/{id}_take2.csv".format(f=name,id=int(row['dr3_source_id'].values[0])), index_label='index')
    #save the plots (the file name I'm using is Gaia_edr3 source_id, change this based on your need)
    plot1 = model1.corner_observed()
    plt.savefig("{f}_plots/{id1}/corner_{id2}.png".format(f=name,id1=int(row['dr3_source_id'].values[0]),id2=int(row['dr3_source_id'].values[0])))
    plot2 = model1.corner_physical()
    plt.savefig("{f}_plots/{id1}/physical_{id2}.png".format(f=name,id1=int(row['dr3_source_id'].values[0]),id2=int(row['dr3_source_id'].values[0])))
    """
    Here you might have one of the causes of your memory leak. matplotlib
    uses a lot of memory and you should close all open instances every time
    to avoid problems. This is true regardless of the code oyu are writing.
    """
    plt.clf()# cleans your plot1
    plt.close('all') #closes everything
    #Now when it leaves this function all local variables will be deleted!



def read_isochrones(name):
    """
    Open the files with the isochrones results and saves the median values,
    the percentiles (16th and 84th) for all information of interest.
    """
    params = at.read("./isochrones_input/{}_isochrones.csv".format(name),delimiter=",",header_start=0)
    stars = params['dr3_source_id']
    folder = './{}_isochrones/'.format(name)
    f = open("{}_iso_params_2.csv".format(name), "w")
    f.write("id,teff_16,teff,teff_84,e_teff,logg_16,logg,logg_84,e_logg,feh_16,feh,feh_84,e_feh,age_16,age,age_84,mass_16,mass,mass_84,radius_16,radius,radius_84,distance_16,distance,distance_84,luminosity_16,luminosity,luminosity_84,AV_16,AV,AV_84,nu_max_16,nu_max,nu_max_84,delta_nu_16,delta_nu,delta_nu_84\n")
    length = len(stars)
    for s in range(0,1200):
        try:
            i=int(stars[s])
            data = pd.read_csv('./{f}_isochrones/{id}_take2.csv'.format(f=name,id=i))
            print(i)
            f.write("%s,%i,%i,%i,%i,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n"\
              %(i,np.quantile(data['Teff'],[0.16])-np.quantile(data['Teff'],[0.5]),np.quantile(data['Teff'],[0.5]),np.quantile(data['Teff'],[0.84])-np.quantile(data['Teff'],[0.5]),\
              np.sqrt((np.quantile(data['Teff'],[0.5]) - np.quantile(data['Teff'],[0.16]))**2. + (np.quantile(data['Teff'],[0.5]) - np.quantile(data['Teff'],[0.84]))**2.),\
              np.quantile(data['logg'],[0.16])-np.quantile(data['logg'],[0.5]),np.quantile(data['logg'],[0.5]),np.quantile(data['logg'],[0.84])-np.quantile(data['logg'],[0.5]),\
              np.sqrt((np.quantile(data['logg'],[0.5]) - np.quantile(data['logg'],[0.16]))**2. + (np.quantile(data['logg'],[0.5]) - np.quantile(data['logg'],[0.84]))**2.),\
              np.quantile(data['feh'],[0.16])-np.quantile(data['feh'],[0.5]),np.quantile(data['feh'],[0.5]),np.quantile(data['feh'],[0.84])-np.quantile(data['feh'],[0.5]),\
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

