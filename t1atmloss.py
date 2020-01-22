#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constrain the evolution of the atmospheres of the TRAPPIST-1 Planets. Mostly
built of David Fleming's modeling of the stellar evolution.

Rory Barnes
rory@astro.washington.edu
22 Jan 2020

"""

import os
from approxposterior import approx, utility, gpUtils
import emcee
import numpy as np
import george

"""

Constraints:
1. Stellar luminosity
2. L_bol/L_XUV
3. Orbital period of planet e
4. Mass of planet e
5. Radius of planet e

"""

"""

Model parameters:
1. Saturation fraction, f_sat
2. Exponential decay term, beta_XUV
3. Age
4. Stellar mass
5. Stellar radius
6. Planet's iron fraction
7. Planet's silicate fraction
8. Planet's water fraction
9. Planet's hydrogen fraction

"""

"""

Free parameters:
1. Efficiency of hydrogen escape, epsilon_H
2. Efficiecny of water photolysis and H escape, epsilon_H2O
3. Radius of XUV absorption, R_XUV

"""

def SamplePrior(size=1, **kwargs):
    """
    Sample dMass, dSatXUVFrac, dSatXUVTime, dStopTime, and dXUVBeta from their
    prior distributions.
    """

    ret = []
    for ii in range(size):
        while True:
            guess = [np.random.uniform(low=0.07, high=0.11),
                     norm.rvs(loc=fsatTrappist1, scale=fsatTrappist1Sig, size=1)[0],
                     np.random.uniform(low=0.1, high=12),
                     norm.rvs(loc=ageTrappist1, scale=ageTrappist1Sig, size=1)[0],
                     norm.rvs(loc=betaTrappist1, scale=betaTrappist1Sig, size=1)[0]]
            if not np.isinf(LnPriorTRAPPIST1(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]
# end function

def LnPrior(x, **kwargs):
    """
    log prior
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x

    # Uniform prior for stellar mass [Msun]
    if (dMass < dPriorStarMassMin) or (dMass > dPriorStarMassMax):
        return -np.inf

    # Uniform prior on saturation timescale [100 Myr - 12 Gyr]
    if (dSatXUVTime < dPriorSatTimeMin) or (dSatXUVTime > dPriorSatTimeMax):
        return -np.inf

    # Large bound for age of system [Gyr] informed by Burgasser et al. (2017)
    if (dStopTime < dPriorAgeMin) or (dStopTime > dPriorAgeMax):
        return -np.inf

    # Hard bounds on XUVBeta to bracket realistic values
    if (dXUVBeta < dPriorBetaMin) or (dXUVBeta > dPriorBetaMax):
        return -np.inf

    # Hard bound on log10 saturation fraction (log10)
    if (dSatXUVFrac < dPriorSatFracMin) or (dSatXUVFrac > dPriorSatFracMax):
        return -np.inf

    # Age prior
    dLnPrior = norm.logpdf(dStopTime, dAge, dAgeSig)

    # Beta prior
    dLnPrior += norm.logpdf(dXUVBeta, dBeta, dBetaSig)

    # fsat prior
    dLnPrior += norm.logpdf(dSatXUVFrac, dFSat, dFSat)

    return dLnPrior
# end function

### Loglikelihood and MCMC functions ###

def LnLike(x, **kwargs):
    """
    loglikelihood function: runs VPLanet simulation
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
    dSatXUVFrac = 10 ** dSatXUVFrac # Unlog
    dStopTime *= 1.e9 # Convert from Gyr -> yr
    dOutputTime = dStopTime # Output only at the end of the simulation

    # Get the prior probability to ignore unphysical state vectors
    # Do this to prevent errors stemming from VPLanet not finishing
    lnprior = kwargs["LnPrior"](x, **kwargs)
    if np.isinf(lnprior):
        blobs = np.array([np.nan, np.nan, np.nan])
        return -np.inf, blobs

    # Get strings containing VPLanet input files (they must be provided!)
    try:
        star_in = kwargs.get("STARIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: Must supply STARIN and VPLIN.")
        raise

    # Get PATH
    try:
        PATH = kwargs.get("PATH")
    except KeyError as err:
        print("ERROR: Must supply PATH.")
        raise

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    starName = 'st%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    starFile = starName + '.in'
    logfile = sysName + '.log'
    starFwFile = '%s.star.forward' % sysName

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dSatXUVFrac), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dSatXUVTime), star_in)
    star_in = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dXUVBeta), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Populate the system input file

    # Populate list of planets
    saBodyFiles = str(starFile) + " #"
    saBodyFiles = saBodyFiles.strip()

    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s #' % saBodyFiles, vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        os.remove(os.path.join(PATH, "output", starFwFile))
        os.remove(os.path.join(PATH, "output", logfile))
    except FileNotFoundError:
        # Run failed!
        blobs = np.array([np.nan, np.nan, np.nan])
        return -np.inf, blobs

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age / utils.YEARSEC >= dStopTime:
        blobs = np.array([np.nan, np.nan, np.nan])
        return -np.inf, blobs

    # Get stellar properties
    dLum = float(output.log.final.star.Luminosity)
    dLumXUV = float(output.log.final.star.LXUVStellar)
    dRad = float(output.log.final.star.Radius)

    # Compute ratio of XUV to bolometric luminosity
    dLumXUVRatio = dLumXUV / dLum

    # Extract constraints
    # Must at least have luminosity, err for star
    lum = kwargs.get("LUM")
    lumSig = kwargs.get("LUMSIG")
    try:
        lumXUVRatio = kwargs.get("LUMXUVRATIO")
        lumXUVRatioSig = kwargs.get("LUMXUVRATIOSIG")
    except KeyError:
        lumXUVRatio = None
        lumXUVRatioSig = None

    # Compute the likelihood using provided constraints, assuming we have
    # luminosity constraints for host star
    lnlike = ((dLum - lum) / lumSig) ** 2
    if lumXUVRatio is not None:
        lnlike += ((dLumXUVRatio - lumXUVRatio) / lumXUVRatioSig) ** 2
    lnlike = -0.5 * lnlike

    # Return likelihood and blobs
    blobs = np.array([dLum, dLumXUV, dRad])
    return lnlike, blobs
#end function

# Dictionary to hold all constraints
kwargs = {"PATH" : ".",                          # Path to all files
          "LnPrior" : LnPriorTRAPPIST1,          # Function for priors
          "PriorSample" : samplePriorTRAPPIST1,  # Function to sample priors
          "LUM" : lumTrappist1,                  # Best fit luminosity constraint
          "LUMSIG" : lumTrappist1Sig,            # Luminosity uncertainty (Gaussian)
          "LUMXUVRATIO" : LRatioTrappist1,       # L_bol/L_XUV best fit
          "LUMXUVRATIOSIG" : LRatioTrappist1Sig, # L_bol/L_XUV uncertainty (Gaussian)
          "PER_E" : ePer,                        # Best fit orbital period for planet e
          "PER_ESIG" : ePerSig,
          "MASS_E" : eMass,
          "MASS_ESIG" : eMassSig,
          "RAD_E" : eRad,
          "RAD_ESIG" : eRadSig
          }

# Stellar properties: Trappist1 in nearly solar metallicity, so the Baraffe+2015 tracks will be good
dLum = 0.000522               # Van Grootel et al. (2018) [Lsun]
dLumSig = 0.000019            # Van Grootel et al. (2018) [Lsun]

dRadius = 0.121               # Van Grootel et al. (2018) [Rsun]
dRadiusSig = 0.003            # Van Grootel et al. (2018) [Rsun]

dLogLXUV = -6.4               # Wheatley et al. (2017), Van Grootel et al. (2018)
dLogLXUVSig = 0.05            # Wheatley et al. (2017), Van Grootel et al. (2018)

dLXUV = 3.9e-7                # Wheatley et al. (2017), Van Grootel et al. (2018)
dLXUVSig = 0.5e-7             # Wheatley et al. (2017), Van Grootel et al. (2018)

dLRatio = 7.5e-4              # Wheatley et al. (2017)
dLRatioSig = 1.5e-4           # Wheatley et al. (2017)

dBeta = -1.18                 # Jackson et al. (2012)
dBetaSig = 0.31               # Jackson et al. (2012)

dAge = 7.6                    # Burgasser et al. (2017) [Gyr]
dAgeSig = 2.2                 # Burgasser et al. (2017) [Gyr]

dFSat = -2.92                 # Wright et al. (2011) and Chadney et al. (2015)
dFSatSig = 0.26               # Wright et al. (2011) and Chadney et al. (2015)


# Define algorithm parameters

iDim = 5                         # Dimensionality of the problem
iTrainInit = 250                         # Initial size of training set
iNewPoints = 100                          # Number of new points to find each iteration
iMaxIter = 10                        # Maximum number of iterations
iSeed = 90                        # RNG seed
iGPRestarts = 25                 # Number of times to restart GP hyperparameter optimizations
iMinObjRestarts = 10             # Number of times to restart objective fn minimization
iGPOptInterval = 25                 # Optimize GP hyperparameters even this many iterations

# Define priors
dPriorStarMassMin = 0.07
dPriorStarMassMax = 0.11
dPriorSatTimeMin = 0.1
dPriorSatTimeMax = 12
dPriorAgeMin = 0.1
dPriorAgeMax = 12
dPriorBetaMin = -2
dPriorBetaMax = 0
dPriorSatFracMin = -5
dPriorSatFracMax = -1

# Prior bounds
daBounds = ((dPriorStarMassMin, dPriorStarMassMax),
          (dPriorSatFracMin, dPriorSatFracMax),
          (dPriorSatTimeMin, dPriorSatTimeMax),
          (dPriorAgeMin, dPriorAgeMax),
          (dPriorBetaMin, dPriorBetaMax))

sAlgorithm = "BAPE"              # Kandasamy et al. (2015) formalism

# Set RNG seed
np.random.seed(iSeed)

# emcee.EnsembleSampler, emcee.EnsembleSampler.run_mcmc and GMM parameters
samplerKwargs = {"nwalkers" : 100}
mcmcKwargs = {"iterations" : int(1.0e4)}

# Loglikelihood function setup required to run VPLanet simulations
#kwargs = trappist1.kwargsTRAPPIST1
#PATH = os.path.dirname(os.path.abspath(__file__))
#kwargs["PATH"] = PATH

# Get the input files, save them as strings
with open(os.path.join(PATH, "star.in"), 'r') as f:
    star_in = f.read()
    kwargs["STARIN"] = star_in
with open(os.path.join(PATH, "vpl.in"), 'r') as f:
    vpl_in = f.read()
    kwargs["VPLIN"] = vpl_in
with open(os.path.join(PATH, "e.in"), 'r') as f:
    e_in = f.read()
    kwargs["EIN"] = e_in

# Generate initial training set using latin hypercube sampling over parameter bounds
# Evaluate forward model log likelihood + lnprior for each theta
if not os.path.exists("apRunAPFModelCache.npz"):
    y = np.zeros(iTrainInit)
    #theta = utility.latinHypercubeSampling(m0, bounds, criterion="maximin")
    for ii in range(iTrainInit):
        theta[ii,:] = SamplePrior()
        y[ii] = LnLike(theta[ii], **kwargs)[0] + LnPrior(theta[ii], **kwargs)
    np.savez("apRunAPFModelCache.npz", theta=theta, y=y)

else:
    print("Loading in cached simulations...")
    sims = np.load("apRunAPFModelCache.npz")
    theta = sims["theta"]
    y = sims["y"]

### Initialize GP ###

# Use ExpSquared kernel, the approxposterior default option
gp = gpUtils.defaultGP(theta, y, order=None, white_noise=-6)

# Initialize approxposterior
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=LnPrior,
                            lnlike=LnLike,
                            priorSample=SamplePrior,
                            bounds=daBounds,
                            algorithm=sAlgorithm)

# Run!
ap.run(m=iNewPoints, nmax=iMaxIter, estBurnin=True, mcmcKwargs=mcmcKwargs,
       thinChains=True,samplerKwargs=samplerKwargs, verbose=True,
       nGPRestarts=iGPRestarts,nMinObjRestarts=iMinObjRestarts, gpCv=5,
       optGPEveryN=iGPOptInterval,seed=iSeed, cache=True, **kwargs)
# Done!
