import astropy.units as u
from astropy.coordinates import SkyCoord

from astropy.io import ascii
from astropy.table import Table
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

t_ext = Table.read("extinction_corr_MS.csv")
print(t_ext)
# Generic function to apply the reddening/extinction correction
# and plot the "corrected" G_0 versus (G-Ks)_0 and the polynomial
# fits. The routine fits a 2nd order polynomial to the whole
# dataset and then subselects objects fainter than this fit. Then
# a 4th order polynomial is made to this single set which is used
# to define the empirical isochrone.



def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
    """
    Calculate the corrected flux excess factor for the input Gaia EDR3 data.
    
    Parameters
    ----------
    
    bp_rp: float, numpy.ndarray
        The (BP-RP) colour listed in the Gaia EDR3 archive.
    phot_bp_rp_excess_factor: float, numpy.ndarray
        The flux excess factor listed in the Gaia EDR3 archive.
        
    Returns
    -------
    
    The corrected value for the flux excess factor, which is zero for "normal" stars.
    
    Example
    -------
    
    phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, phot_bp_rp_flux_excess_factor)
    """
    if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
        bp_rp = np.float64(bp_rp)
        phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
    
    if bp_rp.shape != phot_bp_rp_excess_factor.shape:
        raise ValueError('Function parameters must be of the same shape!')
        
    do_not_correct = np.isnan(bp_rp)
    bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
    greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
    redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)
    
    correction = np.zeros_like(bp_rp)
    correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange], 2)
    correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange], 2) \
        - 0.005879*np.power(bp_rp[greenrange], 3)
    correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]
    
    return phot_bp_rp_excess_factor - correction

def photclean(data,nsig):
    phot_excess = correct_flux_excess_factor(data['BPRP'],data['BPRP_excess'])
    condphot = np.abs(phot_excess) < nsig*(0.0059898+8.817481e-12*data['Gmag']**(7.618399))
    data["BPRPcorr"] = phot_excess
    dataclean = data[condphot]                                             
    return dataclean






def quad_sum(*args):
    return np.sqrt(np.sum([i**2 for i in args]))

def get_ext(A, c, m, p):
    p = np.array(list(p[0]))[:10].astype(float)
    return (p[0] + p[1]*c   + p[2]*c**2   + p[3]*c**3 +
                   p[4]*A   + p[5]*A**2   + p[6]*A**3 +
                   p[7]*A*c + p[8]*A*c**2 + p[9]*A**2*c)

def make_fit(n_order, poly, data):
    fit = 0
    for i in range(n_order+1):
        print(i, n_order-i)
        fit = fit + data**(n_order-i)*poly[i]
    return fit

fig, ax = plt.subplots(figsize=(8,8))
ax.grid()

def prepare_cmd(CMDtype, d_mod, EBV, absM, col, plotcol1, plotcol2, magshift, clus_name, makeplot):
# Make reddening/extinction corrections using table provided in Gaia EDR3
    col_buff = 0.3
    if CMDtype == 'GKs':
        p1 = t_ext[(t_ext['Xname'] == 'GK') & (t_ext['Kname'] == 'kG')]
        p2 = t_ext[(t_ext['Xname'] == 'GK') & (t_ext['Kname'] == 'kK')]
        limcol = [1.5-col_buff, 4.2+col_buff]
        limmag = [-99, +99]
        plt.xlabel(r"$(G-K_{\rm s})_{0}$")

    if CMDtype == 'GBPRP':
        p1 = t_ext[(t_ext['Xname'] == 'BPRP') & (t_ext['Kname'] == 'kG')]
        p2 = t_ext[(t_ext['Xname'] == 'BPRP') & (t_ext['Kname'] == 'kG')]
        limcol = [0.9-col_buff,3.4+col_buff]
        limmag = [0.0,10.2]
        plt.xlabel(r"BP-RP")
    plt.ylabel(r"$G_{0}$")
    plt.xlim(limcol)

    A1 = np.array([get_ext(3.09*EBV, col[i], absM[i], p1) for i in range(len(absM))])
    A2 = np.array([get_ext(3.09*EBV, col[i], absM[i], p2) for i in range(len(absM))])
    col0 = np.array(col)-(A1-A2)
    mag0 = np.array(absM) - d_mod - A1

# Only want G6V -- M7V range for the CMD fit. This easily covers the
# colour range for the 4MOST selection, but not too large that we have
# to worry about MSTO or extremely (intrinsically) faint stars/BDs.
#    cr = np.where[(col0 > limcol[0]) & (col0 < limcol[1])]
    colR = col0[(col0 > limcol[0]) & (col0 < limcol[1]) & (mag0 > limmag[0]) & (mag0 < limmag[1])]
    magR = mag0[(col0 > limcol[0]) & (col0 < limcol[1]) & (mag0 > limmag[0]) & (mag0 < limmag[1])]
# Sort colour/magnitude points in "increasing" colour ("decreasing" Teff).
    ind = np.argsort(colR)
    colR, magR = colR[ind], magR[ind]
    p = np.polyfit(colR, magR-0.35, 2)
    g = [magR >= np.polyval(p, colR)-magshift]
    p2= np.polyfit(colR[tuple(g)], magR[tuple(g)], 4)
    print(p2)
    chi_squared = np.sum((np.polyval(p2, colR[g]) - magR[g]) ** 2)/(len(magR[g])-1)
    if makeplot == 1:
# Plot data-points
        ax.scatter(colR, magR, s=10, color=plotcol1, label=clus_name)

# second-order poly (and plot this)
        ax.plot(colR,np.polyval(p, colR)-magshift,color=plotcol2)

# sometimes it's good to shift this up and down by a few tenths of a
# magnitude to check we're getting enough stars in the "single"
# population and not having gaps in the calibrator points at any
# given colour (this is just done by eye).
        ax.plot(colR[g],np.polyval(p2, colR[g]), '--', color=plotcol2)
        return p2, colR, magR, chi_squared
    else:
        return p2, colR, magR, chi_squared

GES       = ascii.read("./clusters/table3_idr6_final_eDR3.csv")
GES["BPRP"] = GES["BPmag"]-GES["RPmag"]
GES = photclean(GES, 5.)
Pleiades = ascii.read("./clusters/Pleiades_eDR3_JHKs_final.dat")
Pleiades["BPRP"] = Pleiades["BPmag"]-Pleiades["RPmag"]
Pleiades = photclean(Pleiades, 5.)
NGC2547  = ascii.read("./clusters/NGC2547_eDR3_JHKs_final.dat")
NGC2547["BPRP"] = NGC2547["BPmag"]-NGC2547["RPmag"]
NGC2547 = photclean(NGC2547, 5.)
Hyades  = ascii.read("./clusters/Hyades_Oh20_EDR3_JHKs_final.dat")
Hyades["BPRP"] = Hyades["BPmag"]-Hyades["RPmag"]
Hyades = photclean(Hyades, 5.)


# Minor correction to put the 2MASS data on the correct photometric scale as VHS.
# Page 11 (eqn 4.3.1) http://www.eso.org/rm/api/v1/public/releaseDescriptions/144
NGC2547["Ksmag"]  = NGC2547["Ksmag"]  + 0.01*(NGC2547["Jmag"]-NGC2547["Ksmag"])


# JUST DEAL WITH THE OBJECTS IN COMMON FOR NGC2547. Pleiades in the north so no
# VHS counterparts.
Ks_TM, Ks_VH = [], []
Ks_fin, eKs_fin, rKs_fin = [], [], []
for Vf, Tf in enumerate(NGC2547["Ksmag"]):
    Ks_T, eKs_T = NGC2547[Vf]["Ksmag"], NGC2547[Vf]["e_Ksmag"]
    Ks_V, eKs_V = NGC2547[Vf]["Ksap3"], NGC2547[Vf]["e_Ksap3"]
    if Ks_V < 12.5:
        Ks_TM.append(Ks_T)
        Ks_VH.append(Ks_V)
        Ks_fin.append(Ks_T)
        eKs_fin.append(eKs_T)
        rKs_fin.append("TB")
    if Ks_V >= 12.5:
        if eKs_V > eKs_T:
            Ks_TM.append(Ks_T)
            Ks_VH.append(Ks_V)
            Ks_fin.append(Ks_T)
            eKs_fin.append(eKs_T)
            rKs_fin.append("TF")
        else:
            Ks_TM.append(Ks_T)
            Ks_VH.append(Ks_V)
            Ks_fin.append(Ks_V)
            eKs_fin.append(eKs_V)
            rKs_fin.append("VF")

plt.gca().invert_yaxis()


# PLOTS
NGC2547["Ks_fin"], NGC2547["eKs_fin"] = Ks_fin, eKs_fin
NGC2547 = NGC2547[NGC2547["ruwe"]<1.4]
NGC2547.remove_columns(['Ksmag', 'e_Ksmag'])

Pleiades = Pleiades[(Pleiades["ruwe"]<1.4) & (Pleiades["pc"] >= 0.99)]
gamVel = GES[(GES["CLUSTER"]=="gamma2_Vel") & (GES["MEM3D"] >= 0.95)]
NGC6530 = GES[(GES["CLUSTER"]=="NGC6530") & (GES["MEM3D"] >= 0.95)]
NGC2516 = GES[(GES["CLUSTER"]=="NGC2516") & (GES["MEM3D"] >= 0.95)]

#dmod reference
#Hyades: Using individual parallaxes rather than a cluster distance.
#NGC6530: d_mod = 10.60 ± 0.02 ± 0.09, Table1-Column9-Jackson+22.
#Pleiades:  d(pc) 135.15 +/- 0.45, Lodieu+19.
#gamVel: d_mod = 7.73 ± 0.01 ± 0.02, Table1-Column9-Jackson+22.
#NGC2547: 7.93 ± 0.01 ± 0.03, Table1-Column9-Jackson+22.
#NGC2516: 8.07 ± 0.01 ± 0.03, Table1-Column9-Jackson+22.

def set_up_plot(tab_in, m1, m2, m3, phot_corr):
    for m in m1:
        for col in tab_in.columns:
            if m == col:
                mag_out = tab_in[m]
    for m in m2:
        for col in tab_in.columns:
            if m == col:
                m1_out = tab_in[m]
    for m in m3:
        for col in tab_in.columns:
            if m == col:
                m2_out = tab_in[m]
    if phot_corr is not None:
        for m in phot_corr:
            for col in tab_in.columns:
                if m == col:
                    cor_out = np.array(tab_in[m], dtype=float)
        col_out = m1_out - m2_out + cor_out
    else:
        col_out = m1_out - m2_out
    return mag_out, col_out

x = input("Which CMD do you want? \n(1) = G vs G-Ks, (2) = G vs BP-RP")
choose_col = None
if int(x) == 1:
    choose_col = "GKs"
    m1_t = ['phot_g_mean_mag', 'Gmag', 'GMAG']
    m2_t = ['phot_g_mean_mag', 'Gmag', 'GMAG']
    m3_t = ['Kmag','Ks_fin', 'Ksmag', 'KMAGP']
    phot_corr = None
if int(x) == 2:
    choose_col = "GBPRP"    
    m1_t = ['phot_g_mean_mag','Gmag', 'GMAG']
    m2_t = ['BPmag', 'phot_bp_mean_mag']
    m3_t = ['RPmag', 'phot_rp_mean_mag']
    phot_corr = ["BPRPcorr"]
    
NGC6530_p2,  NGC6530_col,  NGC6530_mag,  NGC6530_chi2  = prepare_cmd(choose_col, 10.6, quad_sum(0.02, 0.09),
                                                                     set_up_plot(NGC6530, m1_t, m2_t, m3_t, phot_corr)[0],
                                                                     set_up_plot(NGC6530, m1_t, m2_t, m3_t, phot_corr)[1], 
                                                                     'gray', 'darkgray', 0.0, 'NGC 6530 ($1-2\,$Myr)', 0)
gamVel_p2,   gamVel_col,   gamVel_mag,   gamVel_chi2   = prepare_cmd(choose_col, 7.73, quad_sum(0.01, 0.02),
                                                                     set_up_plot(gamVel, m1_t, m2_t, m3_t, phot_corr)[0], 
                                                                     set_up_plot(gamVel, m1_t, m2_t, m3_t, phot_corr)[1], 
                                                                     'limegreen', 'darkgreen', 0.0, 'gam Vel ($15-25\,$Myr)', 0)
Pleiades_p2, Pleiades_col, Pleiades_mag, Pleiades_chi2 = prepare_cmd(choose_col, 5.65, 0.03,
                                                                     set_up_plot(Pleiades, m1_t, m2_t, m3_t, phot_corr)[0], 
                                                                     set_up_plot(Pleiades, m1_t, m2_t, m3_t, phot_corr)[1], 
                                                                     'pink', 'red', 0.0, 'Pleiades ($125\pm10\,$Myr)', 1)
NGC2547_p2,  NGC2547_col,  NGC2547_mag,  NGC2547_chi2  = prepare_cmd(choose_col, 7.93, quad_sum(0.01, 0.03),
                                                                     set_up_plot(NGC2547, m1_t, m2_t, m3_t, phot_corr)[0], 
                                                                     set_up_plot(NGC2547, m1_t, m2_t, m3_t, phot_corr)[1],  
                                                                     'skyblue', 'blue', 0.0, 'NGC 2547 ($38-41\,$Myr)', 1)
NGC2516_p2,  NGC2516_col,  NGC2516_mag,  NGC2516_chi2  = prepare_cmd(choose_col, 8.07, quad_sum(0.01, 0.03),
                                                                     set_up_plot(NGC2516, m1_t, m2_t, m3_t, phot_corr)[0], 
                                                                     set_up_plot(NGC2516, m1_t, m2_t, m3_t, phot_corr)[1], 
                                                                     'brown', 'brown', 0.0, 'NGC 2516 ($100-150\,$Myr)', 0)
Hyades_p2,   Hyades_col,   Hyades_mag,   Hyades_chi2   = prepare_cmd(choose_col, 0.0, 0.010,
                                                                     set_up_plot(Hyades, m1_t, m2_t, m3_t, phot_corr)[0]-5.0*np.log10(100./Hyades["parallax"]),
                                                                     set_up_plot(Hyades, m1_t, m2_t, m3_t, phot_corr)[1], 
                                                                     'orange', 'peru', 0.0, 'Hyades ($625\pm50\,$Myr)', 1)


#for i in range(len(Hyades)):
#    print(Hyades["phot_g_mean_mag"][i], Hyades["parallax"][i], Hyades["phot_g_mean_mag"][i]-5.0*np.log10(100./Hyades["parallax"][i]))

NGC2516_fit = make_fit(4, Pleiades_p2, NGC2516_col)

IsoDiff = NGC2516_mag - NGC2516_fit  > 0.0
n_miss = sum(bool(x) for x in IsoDiff)/len(IsoDiff)
j=0
while n_miss > 0.05:
    IsoDiff = NGC2516_mag - NGC2516_fit  > 0.0
    n_miss = sum(bool(x) for x in IsoDiff)/len(IsoDiff)
    print("n_miss = %4.2f, j = %4.2f" % (n_miss, j))
    NGC2516_mag = NGC2516_mag - 0.01
    j = j - 0.01

#handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
#order = [1,2,4,3,0]

#add legend to plot
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

ax.legend()
fig.savefig(choose_col+"_CMD.jpg")
