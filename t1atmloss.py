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
from scipy.stats import norm
import emcee
import numpy as np
import george
import random
import re
import subprocess
import vplot as vpl

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

# Constants
LSUN = 3.846e26 # Solar luminosity in ergs/s
YEARSEC = 3.15576e7 # seconds per year
BIGG = 6.67428e-11 # Universal Gravitational Constant in cgs
DAYSEC = 86400.0 # seconds per day
MSUN = 1.988416e30 # mass of sun in g
AUCM = 1.49598e11 # cm per AU
MTO = 1.4e24 # Mass of all of Earth's water in g
MEarth = 5.972e27 # Mass of Earth in g

def SamplePrior(size=1, **kwargs):
    """
    Sample dMass, dSatXUVFrac, dSatXUVTime, dStopTime, and dXUVBeta from their
    prior distributions.
    """

    ret = []
    for ii in range(size):
        while True:
            guess = [np.random.uniform(low=dPriorStarMassMin, high=dPriorStarMassMax),
                     norm.rvs(loc=dFSat, scale=dFSatSig, size=1)[0],
                     np.random.uniform(low=dPriorSatTimeMin, high=dPriorSatTimeMax),
                     norm.rvs(loc=dAge, scale=dAgeSig, size=1)[0],
                     norm.rvs(loc=dBeta, scale=dBetaSig, size=1)[0]]
            if not np.isinf(LnPrior(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]
# end function

def LnPrior(daStateVector, **kwargs):
    """
    log prior
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = daStateVector

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

def LnLike(daStateVector, **kwargs):
    """
    loglikelihood function: runs VPLanet simulation
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = daStateVector
    dSatXUVFrac = 10 ** dSatXUVFrac # Unlog
    dStopTime *= 1.e9 # Convert from Gyr -> yr
    dOutputTime = dStopTime # Output only at the end of the simulation

    # Get the prior probability to ignore unphysical state vectors
    # Do this to prevent errors stemming from VPLanet not finishing
    dLnPrior = kwargs["LnPrior"](daStateVector, **kwargs)
    if np.isinf(dLnPrior):
        blobs = np.array([np.nan, np.nan, np.nan])
        return -np.inf, blobs

    # Get strings containing VPLanet input files (they must be provided!)
    try:
        sStarFileIn = kwargs.get("STARIN")
        sPrimaryFileIn = kwargs.get("VPLIN")
#        sPlanetEFileIn = kwargs.get("EIN")
    except KeyError as err:
        print("ERROR: Must supply VPLIN, STARIN, and EIN in LnLike.")
        raise

    # Get PATH
    try:
        PATH = kwargs.get("PATH")
    except KeyError as err:
        print("ERROR: Must supply PATH.")
        raise

    # Randomize file names to prevent overwrites
    sVPLName = 'vpl%012x' % random.randrange(16**12)
    sStarName = 'st%012x' % random.randrange(16**12)
#    sEName = 'e%012x' % random.randrange(16**12)
    sPrimaryFile = sVPLName + '.in'
    sStarFile = sStarName + '.in'
#    sEFile = sEName + '.in'
    sLogFile = sVPLName + '.log'
    sStarFwFile = '%s.star.forward' % sVPLName
#    sEFwFile = '%s.e.forward' % sEName


# mkdir output!

    # Populate the star input file
    sStarFileIn = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), sStarFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dSatXUVFrac), sStarFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dSatXUVTime), sStarFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dXUVBeta), sStarFileIn)
    with open(os.path.join(PATH, "output", sStarFile), 'w') as f:
        print(sStarFileIn, file = f)

    # Populate the system input file

    # Populate list of planets
    saBodyFiles = str(sStarFile) + " #"
    saBodyFiles = saBodyFiles.strip()

    sPrimaryFileIn = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), sPrimaryFileIn)
    sPrimaryFileIn = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), sPrimaryFileIn)
    sPrimaryFileIn = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sVPLName, sPrimaryFileIn)
    sPrimaryFileIn = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s #' % saBodyFiles, sPrimaryFileIn)
    with open(os.path.join(PATH, "output", sPrimaryFile), 'w') as f:
        print(sPrimaryFileIn, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", '-q', sPrimaryFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = sLogFile)

    try:
        os.remove(os.path.join(PATH, "output", sStarFile))
        os.remove(os.path.join(PATH, "output", sPrimaryFile))
        os.remove(os.path.join(PATH, "output", sStarFwFile))
        os.remove(os.path.join(PATH, "output", sLogFile))
    except FileNotFoundError:
        # Run failed!
        daParams = np.array([np.nan, np.nan, np.nan])
        return -np.inf, blobs

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age / YEARSEC >= dStopTime:
        daParams = np.array([np.nan, np.nan, np.nan])
        return -np.inf, blobs

    # Get stellar properties
    dLumTrial = float(output.log.final.star.Luminosity)
    dLumXUVTrial = float(output.log.final.star.LXUVStellar)
    dRadiusTrial = float(output.log.final.star.Radius)

    # Compute ratio of XUV to bolometric luminosity
    dLumXUVRatioTrial = dLumXUVTrial / dLumTrial

    # Extract constraints
    # Must at least have luminosity, err for star
    dLum = kwargs.get("LUM")
    dLumSig = kwargs.get("LUMSIG")
    try:
        dLumXUVRatio = kwargs.get("LUMXUVRATIO")
        dLumXUVRatioSig = kwargs.get("LUMXUVRATIOSIG")
    except KeyError:
        dLumXUVRatio = None
        dLumXUVRatioSig = None

    # Compute the likelihood using provided constraints, assuming we have
    # luminosity constraints for host star
    dLnLike = ((dLum - dLumTrial) / dLumSig) ** 2
    if dLumXUVRatio is not None:
        dLnLike += ((dLumXUVRatio - dLumXUVRatioTrial) / dLumXUVRatioSig) ** 2
    dLnLike = -0.5 * dLnLike

    # Return likelihood and diognostic parameters
    daParams = np.array([dLumTrial, dLumXUVTrial, dRadiusTrial])
    return dLnLike, daParams
#end function


######## Main code begins here ###########

# Stellar properties: Trappist1 is nearly solar metallicity, so the Baraffe+2015 tracks will be good
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

# Dictionary to hold all constraints
kwargs = {"PATH" : ".",                          # Path to all files
          "LnPrior" : LnPrior,          # Function for priors
          "PriorSample" : SamplePrior,  # Function to sample priors
          "LUM" : dLum,                  # Best fit luminosity constraint
          "LUMSIG" : dLumSig,            # Luminosity uncertainty (Gaussian)
          "LUMXUVRATIO" : dLRatio,       # L_bol/L_XUV best fit
          "LUMXUVRATIOSIG" : dLRatioSig, # L_bol/L_XUV uncertainty (Gaussian)
#          "PER_E" : ePer,                        # Best fit orbital period for planet e
#          "PER_ESIG" : ePerSig,
#          "MASS_E" : eMass,
#          "MASS_ESIG" : eMassSig,
#          "RAD_E" : eRad,
#          "RAD_ESIG" : eRadSig
          }
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
daBounds = (
          (dPriorStarMassMin, dPriorStarMassMax),
          (dPriorSatFracMin, dPriorSatFracMax),
          (dPriorSatTimeMin, dPriorSatTimeMax),
          (dPriorAgeMin, dPriorAgeMax),
          (dPriorBetaMin, dPriorBetaMax)
          )

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

# Extract path
PATH = kwargs["PATH"]

# Get the input files, save them as strings
with open(os.path.join(PATH, "star.in"), 'r') as f:
    sStarFile = f.read()
    kwargs["STARIN"] = sStarFile
with open(os.path.join(PATH, "vpl.in"), 'r') as f:
    sPrimaryFile = f.read()
    kwargs["VPLIN"] = sPrimaryFile
#with open(os.path.join(PATH, "e.in"), 'r') as f:
#    sEFile = f.read()
#    kwargs["EIN"] = sEFile

# Generate initial training set using latin hypercube sampling over parameter bounds
# Evaluate forward model log likelihood + lnprior for each theta
if not os.path.exists("apRunAPFModelCache.npz"):
    y = np.zeros(iTrainInit)
    theta = np.zeros((iTrainInit,iDim))
    for ii in range(iTrainInit):
        print("Training simulation: %d" % ii)
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
