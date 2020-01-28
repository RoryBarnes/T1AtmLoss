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

Model parameters (daStateVector) [prior-type, (prior-properties), (bounds), ref]:
1. Stellar mass (dStarMass) [uniform, (0.07,0.11), (0.07,0.11), Fleming20]
2. Log saturation fraction (dFSat) [normal, (-2.92, 0.26), (-5,-1), Wright11,Chadney15]
3. Saturation time (dTSat) [uniform, (0.1,12), (0.1,12), Fleming20]
4. Age (dAge) [normal, (7.6,2.2), (0.1,12), Burgasser17]
5. XUV Exponential decay term (dBeta) [normal, (-1.18,0.31), (-2,0), Jackson12]
6. Planet's iron fraction (dFIron) [uniform, (0.05,0.9), (0.05,0.9)]
7. Planet's rock fraction (dFRock) [uniform, (0.05,0.9), (0.05,0.9)]
8. Planet's initial water fraction (dFWater) [uniform, (0.05,0.9), (0.05,0.9)]
9. Planet's initial hydrogen fraction (dFHyd) [uniform, (0,10), (0,10)]
10. Efficiency of hydrogen escape (dEscCoeffH) [uniform, (0.1,0.5), (0.1,0.5)]
11. Efficiecny of water photolysis and H escape (dEscCoeffH20) [uniform, (0.06,0.13), (0.06,0.13), Bolmont17]
12. Pressure of XUV absorption (dPressXUV) [uniform, (0.1,10), (0.1,10), LC17]
13. Planetary Albedo (dAlbedo) [uniform, (0.05,0.8), (0.05,0.8)]
14. Initial planetary mass (dPlanetMass) [uniform, (0.5,10), (0.5,10)]

Observational constraints [type, (limits), ref]:
1. Stellar luminosity (dLum) [normal, (5.22e-4,1.9e-5), vanGrootel18]
2. L_bol/L_XUV (dLRatio) [normal, (7.5e-4,1.5e-4), Wheatley17]
3. Orbital period of planet e (dPerE) [normal, (6.099043,1.5e-5), Delrez17]
4. Stellar Effective Temperature (dTEff)[normal, (2511,37), Delrez17]
5. Current mass of planet e (dPlanetMass) [normal, (0.772,0.077), Grimm18]
6. Radius of planet e (dPlanetRad) [normal, (0.91,0.027), Grimm18]

Mathematical constraints
1. Sum of "fractions" must equal 1

Additional Parameters (daParams)
1. L_XUV (dLXUV)
2. Stellar radius (dStarRad)
3. Planetary equilibrium temp. (dEqTemp)
4. Surface pressure (dSurfPress)

"""

# Constants
LSUN = 3.846e26         # Solar luminosity in ergs/s
YEARSEC = 3.15576e7     # seconds per year
BIGG = 6.67428e-11      # Universal Gravitational Constant in cgs
DAYSEC = 86400.0        # seconds per day
MSUN = 1.988416e30      # mass of sun in g
AUCM = 1.49598e11       # cm per AU
MTO = 1.4e24            # Mass of all of Earth's water in g
MEarth = 5.972e27       # Mass of Earth in g

def SampleStateVector(iSize=1, **kwargs):

    daStateVector = []
    for iSample in range(iSize):
        while True:
            daGuess = [np.random.uniform(low=dStarMassMin, high=dStarMassMax),
                     norm.rvs(loc=dFSatMean, scale=dFSatSig, size=1)[0],
                     np.random.uniform(low=dTSatMin, high=dTSatMax),
                     norm.rvs(loc=dAgeMean, scale=dAgeSig, size=1)[0],
                     norm.rvs(loc=dBetaMean, scale=dBetaSig, size=1)[0],
                     np.random.uniform(low=dFIronMin, high=dFIronMax),
                     np.random.uniform(low=dFRockMin, high=dFRockMax),
                     np.random.uniform(low=dFWaterMin, high=dFWaterMax),
                     np.random.uniform(low=dFHydMin, high=dFHydMax),
                     np.random.uniform(low=dEscCoeffHMin, high=dEscCoeffMax),
                     np.random.uniform(low=dEscCoeffH2OMin, high=dEscCoeffH2OMax),
                     np.random.uniform(low=dPressXUVMin, high=dPressXUVMax),
                     np.random.uniform(low=dAlbedoMin, high=dAlbedoMax),
                     np.random.uniform(low=dPlanetMassMin, high=dPlanetMassMax)
                     ]
            if not np.isinf(LnPrior(daGuess, **kwargs)):
                daStateVector.append(daGuess)
                break

    if iSize > 1:
        return daStateVector
    else:
        return daStateVector[0]
# end function

def LnPrior(daStateVector, **kwargs):
    """
    log prior
    """

    dLnPrior = 0

    # Get the current vector
    dMass,dFSat,dTSat,dAge,dBeta,dFIron,dFRock,dFWater,dFHyd,dEscCoeffH,dEscCoeffH20,dPressXUV,dAlbedo,dPlanetMass = daStateVector

    # Uniform priors need no scaling, but must be in the limits
    if (dMass < dStarMassMin) or (dMass > dStarMassMax):
        return -np.inf
    if (dFSat < dFSatMin) or (dFSat > dFSatMax):
        return -np.inf
    if (dTSat < dTSatMin) or (dTSat > dTSatMax):
        return -np.inf
    if (dAge < dAgeMin) or (dAge > dAgeMax):
        return -np.inf
    if (dBeta < dBetaMin) or (dBeta > dBetaMax):
        return -np.inf
    if (dFIron< dFIronMin) or (dFIron > dFIronMax):
        return -np.inf
    if (dFRock < dFRockMin) or (dFRock > dFRockMax):
        return -np.inf
    if (dFWater < dFWaterMin) or (dFWater > dFWaterMax):
        return -np.inf
    if (dHyd < dFHydMin) or (dFHyd > dFHydMax):
        return -np.inf
    if (dEscCoeffH < dEscCoeffHMin) or (dEscCoeffH > dEscCoeffHMax):
        return -np.inf
    if (dEscCoeffH2O < dEscCoeffH2OMin) or (dEscCoeffH2O > dEscCeoffH2OMax):
        return -np.inf
    if (dPressXUV < dPressXUVMin) or (dPressXUV > dPressXUVMax):
        return -np.inf
    if (dAlbedo < dAlbedoMin) or (dAlbedo > dAlbedoMax):
        return -np.inf
    if (dPlanetMass < dPlanetMin) or (dPlanetMass > dPlanetMassMax):
        return -np.inf

    # Scale the normally distributed priors
    dLnPrior += norm.logpdf(dAge, dAgeMean, dAgeSig)
    dLnPrior += norm.logpdf(dBeta, dBetaMean, dBetaSig)
    dLnPrior += norm.logpdf(dFSat, dFSatMean, dFSatSig)

    return dLnPrior
# end function

### Loglikelihood and MCMC functions ###

def LnLike(daStateVector, **kwargs):
    """
    loglikelihood function: runs VPLanet simulation
    """

    # Get the current state vector
    dMass,dFSat,dTSat,dAge,dBeta,dFIron,dFRock,dFWater,dFHyd,dEscCoeffH,dEscCoeffH20,dPressXUV,dAlbedo,dPlanetMass = daStateVector

    # Convert to VPLanet input
    dSatXUVFrac = 10 ** dSatXUVFrac # Unlog
    dStopTime dAge*1.e9 # Convert from Gyr -> yr
    dOutputTime = dStopTime # Output only at the end of the simulation

    # Get the prior probability to ignore unphysical state vectors
    # Do this to prevent errors stemming from VPLanet not finishing
    dLnPrior = kwargs["LnPrior"](daStateVector, **kwargs)
    if np.isinf(dLnPrior):
        daParams = np.array([np.nan, np.nan, np.nan, np.nan])
        return -np.inf, daParams

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

    # Populate the input files
    sStarFileIn = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), sStarFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dFSat), sStarFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dTSat), sStarFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dBeta), sStarFileIn)

    sStarFileIn = re.sub("%s(.*?)#" % "dFracIron", "%s %.6e #" % ("dFracIron", dFIron), sPlanetFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dFracRock", "%s %.6e #" % ("dFracRock", dFRock), sPlanetFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dFracIce", "%s %.6e #" % ("dFracIce", dFWater), sPlanetFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dEnvelopeMass", "%s %.6e #" % ("dEnvelopeMass", -(dFracHyd*dPlanetMass), sPlanetFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dAtmXAbsEffH", "%s %.6e #" % ("dAtmXAbsEffH", dEscCoeffH), sPlanetFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dAtmXAbsEffH2O", "%s %.6e #" % ("dAtmXAbsEffH2O", dEscCoeffH2O), sPlanetFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dPresXUV", "%s %.6e #" % ("dPresXUV", dPressXUV), sPlanetFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dAlbedoGlobal", "%s %.6e #" % ("dAlbedoGlobal", dAlbedo), sPlanetFileIn)
    sStarFileIn = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dPlanetMass), sPlanetFileIn)

    with open(os.path.join(PATH, "output", sStarFile), 'w') as f:
        print(sStarFileIn, file = f)

    with open(os.path.join(PATH, "output", sPlanetFile), 'w') as f:
        print(sPlanetFileIn, file = f)

    # Populate the primary input file
    # Populate list of planets
    saBodyFiles = str(sStarFile) + str(sPlanetFile) +" #"

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
        os.remove(os.path.join(PATH, "output", sPlanetFile))
        os.remove(os.path.join(PATH, "output", sPrimaryFile))
        os.remove(os.path.join(PATH, "output", sStarFwFile))
        #print(sStarFile)
        os.remove(os.path.join(PATH, "output", sLogFile))
    except FileNotFoundError:
        # Run failed!
        daParams = np.array([np.nan, np.nan, np.nan, np.nan])
        return -np.inf, daParams

    # Ensure we ran for as long as we set out to
    # XXX Why not == pr < dEpsilon?
    if not output.log.final.system.Age / YEARSEC >= dStopTime:
        daParams = np.array([np.nan, np.nan, np.nan, np.nan])
        return -np.inf, daParams

    # Get final values of observed parameters
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

    print("dLnLike: ",dLnLike)
    # Return likelihood and diognostic parameters
    daParams = np.array([dLumTrial, dLumXUVTrial, dRadiusTrial])
    return dLnLike, daParams
#end function


######## Main code begins here ###########

# Observational constraints
iNumObs = 6

dLumMean = 0.000522               # Van Grootel et al. (2018) [Lsun]
dLumSig = 0.000019            # Van Grootel et al. (2018) [Lsun]

dTEffMean = 2511
dTEffSig = 37

dLRatio = 7.5e-4              # Wheatley et al. (2017)
dLRatioSig = 1.5e-4           # Wheatley et al. (2017)

dMassE = 0.772
dMassESig = 0.077

dRadE = 0.91
dRadESig = 0.027

dPerE = 6.099043
dPerESig = 1.5e-5

# Old parameters?
#dRadius = 0.121               # Van Grootel et al. (2018) [Rsun]
#dRadiusSig = 0.003            # Van Grootel et al. (2018) [Rsun]

#dLogLXUV = -6.4               # Wheatley et al. (2017), Van Grootel et al. (2018)
#dLogLXUVSig = 0.05            # Wheatley et al. (2017), Van Grootel et al. (2018)

#dLXUV = 3.9e-7                # Wheatley et al. (2017), Van Grootel et al. (2018)
#dLXUVSig = 0.5e-7             # Wheatley et al. (2017), Van Grootel et al. (2018)

# Model parameters with normally distributed priors
iNumModelPrms = 14

dBeta = -1.18                 # Jackson et al. (2012)
dBetaSig = 0.31               # Jackson et al. (2012)

dAge = 7.6                    # Burgasser et al. (2017) [Gyr]
dAgeSig = 2.2                 # Burgasser et al. (2017) [Gyr]

dFSat = -2.92                 # Wright et al. (2011) and Chadney et al. (2015)
dFSatSig = 0.26               # Wright et al. (2011) and Chadney et al. (2015)

# Define bounds of the sampled parameter space for the model parameters
dStarMassMin = 0.07
dStarMassMax = 0.11

dTSatMin = 0.1
dTSatMax = 12

dAgeMin = 0.1
dAgeMax = 12

dBetaMin = -2
dBetaMax = 0

dTSatMin = -5
dTSatMax = -1

dFIronMin = 0.05
dFIronMax = 0.9

dFRockMin = 0.05
dFRockMax = 0.9

dFWaterMin = 0.05
dFWaterMax = 0.9

dFHydMin = 0.0
dFHydMax = 10

dEscCoeffHMin = 0.1
dEscCoeffMax = 0.5

dEscCoeffH2OMin = 0.06
dEscCoeffH2OMax = 0.13

PressXUVMin = 0.1
PressXUVMax = 10

dAlbedoMin = 0.05
dAlbedoMax = 0.9

dPlanetMassMin = 0.5
dPlanetMassMax = 10

# Dictionary to hold all constraints
kwargs = {"PATH" : ".",                          # Path to all files
          "LnPrior" : LnPrior,          # Function for priors
          "PriorSample" : SampleStateVector,  # Function to sample priors
          "LUM" : dLum,                  # Best fit luminosity constraint
          "LUMSIG" : dLumSig,            # Luminosity uncertainty (Gaussian)
          "LUMXUVRATIO" : dLRatio,       # L_bol/L_XUV best fit
          "LUMXUVRATIOSIG" : dLRatioSig, # L_bol/L_XUV uncertainty (Gaussian)
          "TEFF" : dTEffMean,
          "PER_E" : PerE,                        # Best fit orbital period for planet e
          "PER_ESIG" : PerESig,
          "MASS_E" : MassE,
          "MASS_ESIG" : MassESig,
          "RAD_E" : RadE,
          "RAD_ESIG" : RadESig
          }
# Define approxposterior parameters
iTrainInit = 10                         # Initial size of training set
iNewPoints = 10                          # Number of new points to find each iteration
iMaxIter = 10                        # Maximum number of iterations
iSeed = 90                        # RNG seed
iGPRestarts = 10                 # Number of times to restart GP hyperparameter optimizations
iMinObjRestarts = 10             # Number of times to restart objective fn minimization
iGPOptInterval = 25                 # Optimize GP hyperparameters even this many iterations
dConvergenceLimit = 0.1
iNumConverged = 3

# Prior bounds
daBounds = (
          (dStarMassMin, dStarMassMax),
          (dSatFracMin, dSatFracMax),
          (dSatTimeMin, dSatTimeMax),
          (dAgeMin, dAgeMax),
          (dBetaMin, dBetaMax)
          (dFIronMin, dFIronMax),
          (dFRockMin, dFRockMax),
          (dFWaterMin, dFWaterMax),
          (dFHydMin, dFHydMax),
          (dEscCoeffHMin, dEscCoeffMax),
          (dEscCoeffH2OMin, dEscCoeffH2OMax),
          (dPressXUVMin, dPressXUVMax),
          (dAlbedoMin, dAlbedoMax),
          (dPlanetMassMin, dPlanetMassMax)
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
with open(os.path.join(PATH, "planet.in"), 'r') as f:
    sPlanetFile = f.read()
    kwargs["STARIN"] = sStarFile
with open(os.path.join(PATH, "vpl.in"), 'r') as f:
    sPrimaryFile = f.read()
    kwargs["VPLIN"] = sPrimaryFile
#with open(os.path.join(PATH, "e.in"), 'r') as f:
#    sEFile = f.read()
#    kwargs["EIN"] = sEFile

# Make sure directory output/ exists, if not make it!
if not os.path.exists("output"):
    subprocess.call(["mkdir","output"])

# Generate initial training set using latin hypercube sampling over parameter bounds
# Evaluate forward model log likelihood + lnprior for each theta
if not os.path.exists("apRunAPFModelCache.npz"):
    y = np.zeros(iTrainInit)
    theta = np.zeros((iTrainInit,iNumModelPrms))
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
gp = gpUtils.defaultGP(theta, y, order=None, white_noise=-12)

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
       nGPRestarts=iGPRestarts,nMinObjRestarts=iMinObjRestarts,
       optGPEveryN=iGPOptInterval,seed=iSeed, cache=True, convergenceCheck=True,
       eps=dConvergenceLimit,kmax=iNumConverged,**kwargs)
# Done!
