cimport numpy as np
import math
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from PyGasMix.Gasmix cimport Gasmix

cdef double drand48(double dummy)

cdef class Boltz:
    cpdef GetSimFunctions(self, BFieldMag, BFieldAngle, EnableThermalMotion)
    cpdef Start(self)

    cdef public:
        bint Crossed_SST
        '''This is flag to mark if the steady state threshold has been crossed.'''
        double EFieldOverBField
        '''This is a constant that is equal to the electric field / magentic field * 1e-9.'''
        double AngularSpeedOfRotation
        '''This is the angular speed of rotation see cycltron frequency.'''
        double BField_Angle
        '''This is the angle between the magnetic field and the electric field.'''
        double BField_Mag
        '''This is the magnitude of the magentic field.'''
        double Max_Electron_Energy
        '''This is the upper limit for the electron energy integration.'''
        double ElectronEnergyStep
        '''PyBoltz does the electron energy integration in 4000 steps this variable has the difference in energy between each step.'''
        double ThermalEnergy
        '''This indicates the amount of energy in the gas (it is equal to the Boltzman constant * absolute tempreture).'''
        double RhydbergConst
        '''This is Rydberg constant times hc in eV.'''
        double TemperatureCentigrade
        '''This is the tempreture in degrees Centigrade.'''
        double Pressure_Torr
        '''This is the pressure in Torr.'''
        double MaxCollisionTime
        '''Maximum collision time. Default is set to 100.'''
        double SmallNumber
        '''This constant is equal to 1e-20. Used to be a small constant.'''
        double InitialElectronEnergy
        '''The lower limit of the electron energy integration.'''
        double AngleFromZ
        '''Collision scattering angle from the Z plane.'''
        double AngleFromX
        '''Angle used to calculate initial direction cosines in the XY plane. Angle from the X axis.'''
        double EField
        '''Electric field [V/cm].'''
        double MaxNumberOfCollisions
        '''Number of the simulated collisions.'''
        double IonisationRate
        '''Ionisation rate.'''
        double VelocityX
        '''Drift velocity on the x axis.'''
        double VelocityY
        '''Drift velocity on the y axis.'''
        double VelocityZ
        '''Drift velocity on the z axis.'''
        double VelocityErrorX
        '''Percentage error on the x drift velocity.'''
        double VelocityErrorY
        '''Percentage error on the y drift velocity.'''
        double VelocityErrorZ
        '''Percentage error on the z drift velocity.'''
        double AttachmentRate
        '''Attachment Rate.'''
        double IonisationRateError
        '''Percentage Error for IonisationRate'''
        double AttachmentRateError
        '''Percentage Error for AttachmentRate'''
        double LongitudinalDiffusion
        '''Longitudinal diffusion.'''
        double TransverseDiffusion
        '''Transverse diffusion.'''
        double DiffusionX
        '''Diffusion on the x plane.'''
        double DiffusionY
        '''Diffusion on the y plane.'''
        double DiffusionZ
        '''Diffusion on the z plane.'''
        double DiffusionYZ
        '''Diffusion on the yz plane.'''
        double DiffusionXY
        '''Diffusion on the xy plane.'''
        double DiffusionXZ
        '''Diffusion on the xz plane.'''
        double ErrorDiffusionX
        '''Percentage error for DIFXX.'''
        double ErrorDiffusionY
        '''Percentage error for DIFYY.'''
        double ErrorDiffusionZ
        '''Percentage error for DIFZZ.'''
        double ErrorDiffusionYZ
        '''Percentage error for DIFYZ.'''
        double ErrorDiffusionXY
        '''Percentage error for DIFXY.'''
        double ErrorDiffusionXZ
        '''Percentage error for DIFXZ.'''
        double MaximumCollisionTime
        '''This variable is used to store the maximum collision time in the simulation.'''
        double MeanElectronEnergyError
        '''Percentage error for AVE.'''
        double MeanElectronEnergy
        '''Mean electron energy.'''
        double X
        '''Variable used to represent the x position in the simulation.'''
        double Y
        '''Variable used to represent the y position in the simulation.'''
        double Z
        '''Variable used to represent the z position in the simulation.'''
        double LongitudinalDiffusionError
        '''Percentage for DIFLN.'''
        double TransverseDiffusionError
        '''Percentage for DIFTR.'''
        double TemperatureKelvin
        '''Absolute tempreture in Kelvin.'''
        double ReducedIonization
        '''Variable used to represent the ionisation rate.'''
        double ReducedIonizationErr
        '''Variable used to represent the ionisation rate error.'''
        double ReducedAttachment
        '''Variable used to represent the attachement rate.'''
        double ReducedAttachmentErr
        '''Variable used to represent the attachment rate error.'''
        double CONST1
        '''Constant that is equal to (electron mass charge ratio) / 2 * 1e-9.'''
        double CONST2
        '''Constant that is equal to CONST1 * 1e-2.'''
        double CONST3
        '''Constant that is equal to sqrt(0.2 * (electron mass charge ratio)) * 1e-9.'''
        double MaxCollisionFreqTotal
        '''Sum of the maximum collision frequency of each gas.'''
        double Enable_Thermal_Motion
        '''Variable used to indicate wethier to include thermal motion or not.'''
        double PresTempCor
        '''Variable used to calculate the correlation constant between the pressure and tempreture. PresTempCor=ABZERO*Pressure/(ATMOS*(ABZERO+TemperatureC)*100.0D0).'''
        double AN
        '''Variable that is equal to 100 * PresTempCor * ALOSCH. Which is the number of molecules/cm^3.'''
        double VAN
        '''Variable that is equal to 100 * PresTempCor * CONST4 * 1e15'''
        double TimeSum
        '''Variable used to calculate the sum of the collision times in the simulation.'''
        long long Random_Seed
        '''Random number generator seed. Not used at the moment.'''
        long long NumberOfGases
        '''Number of gases in the mixture.'''
        long long EnergySteps
        '''Steps for the electron energy integration.'''
        long long Which_Angular_Model
        '''Variable used to indicate the type of the elastic angular distribtions.'''
        long long Enable_Penning
        '''Variable used to indicate the inclusion of penning effects. '''
        long long AnisotropicDetected
        '''Anisotropic flag used if anisotropic scattering data is detected.'''
        long long Decor_Colls
        '''Number of collisions for decorrelation - (Decorrelation length).'''
        long long Decor_Step
        '''Decorrelation steps.'''
        long long Decor_Lookbacks
        '''Number of decorrelation lookbacks.'''
        long long Num_Samples
        '''Number of electron samples'''
        long long FakeIonizations
        '''Fake ionisation counter.'''
        double CollisionEnergies[4000]
        '''Number of collisions in the Ith energy step.'''
        double CollisionTimes[300]
        '''Number of collisions in the Ith time step.'''
        double CollisionsPerGasPerType[6][5]
        '''Number of collisions for Ith gas at the Jth type.'''
        double ICOLNN[6][10]
        '''Null scatter sum for the Ith gas and the Jth location.'''
        double ICOLN[6][290]
        double AMGAS[6]
        double EC0[6][290]
        double VTMB[6]
        '''Maxwell Boltzman velocity factor for each gas component.'''
        double MaxCollisionFreqTotalG[6]
        '''Fraction of the maximum collision frequency of each gas over the sum of the maximum collision frequencies of all the gases.'''
        double GasIDs[6]
        '''Array used to store the number of the 6 gases in the mixture.'''
        double GasFractions[6]
        '''Array used to store the percentage of each gas in the mixture.'''
        double MoleculesPerCm3PerGas[6]
        '''Array used to calculate the number of molecules/cm^3 for each gas.'''
        double VMoleculesPerCm3PerGas[6]
        '''Array used to calculate the VAN for each gas.'''
        double CrossSectionSum[4000], ElasticCrossSection[4000], AttachmentSectionSum[4000]
        double E[4000]
        '''Energy ar each energy step.'''
        double SqrtEnergy[4000]
        '''The square root of each energy step.'''
        double TotalCrossSection[4000], RelativeIonMinusAttachCrossSection[4000], InelasticCrossSection[4000], FakeIonizationsT[8], FakeIonizationsD[9], LAST[6]
        double N_Inelastic[6]
        '''Number of momentum cross section data points for types other than attachment and ionisation.'''
        double NumMomCrossSectionPoints[6]
        '''Number of momentum cross section data points for each gas.'''
        double ISIZE[6]
        double MaxCollisionFreq[6]
        '''Maximum value of collision frequency for each gas.'''
        double NumMomCrossSectionPointsNull[6]
        '''Number of momentum cross section data points for null collisions.'''
        double AngularModel[6][290], NC0[6][290], EnergyLevels[6][290], InteractionType[6][290], RGas[6][290], ElectronNumChange[6][290],  IonizationCrossSection[6][4000], InelasticCrossSectionPerGas[6][250][4000], LIN[6][250], ALIN[6][250]
        double CollisionFrequency[6][4000][290]
        '''Collision frequency for each gas at every energy step for every data point.'''
        double TotalCollisionFrequency[6][4000]
        '''Total collision frequency for each gas at every energy step.'''
        double PenningFraction[6][3][290]
        '''PenningFraction of each gas'''
        double NullCollisionFreq[6][4000][10]
        '''Null collision frequency for each gas at every energy step for every data point.'''
        double TotalCollisionFrequencyNull[6][4000]
        '''Total null collision frequency for each gas at every energy step.'''
        double ScaleNull[6][10], ScatteringParameter[6][4000][290], AngleCut[6][4000][290]
        double TransverseDiffusion1
        '''Transverse diffusion in microns/cm^0.5.'''
        double TransverseDiffusion1Error
        '''Percentage error for DT1'''
        double LongitudinalDiffusion1
        '''Longitudinal diffusion in microns/cm^0.5'''
        double LongitudinalDiffusion1Error
        '''Percentage error for DL1'''
        double NumMomCrossSectionPointsNT,ISIZENT
        double PIR2
        int Console_Output_Flag
        '''Flag used to stop console printouts'''
        double MeanCollisionTime
        '''Mean collision time. Calculated using a moving average filter where it is equal to 0.9 * MeanCollisionTime + 0.1 * NewTime'''
        # Variables and arrays used when the thermal motion is not included.
        double CollisionFrequencyNT[4000][960]
        '''Collision frequency for each energy step, at each cross section point. This is used for when EnableThermalMotion = 0'''
        double EnergyLevelsNT[960]
        double TotalCollisionFrequencyNT[4000]
        '''Total collision frequency at each energy step. This is used for when EnableThermalMotion = 0'''
        double InteractionTypeNT[960],RGasNT[960],ElectronNumChangeNT[960],PenningFractionNT[3][960],MaxCollisionFreqNT[8]
        double NullCollisionFreqNT[4000][60],TotalCollisionFrequencyNullNT[4000],ScaleNullNT[60],ScatteringParameterNT[4000][960],AngleCutNT[4000][960],AngularModelNT[960],NC0NT[960],EC0NT[960]
        double CollisionsPerGasPerTypeNT[30],ICOLNNT[960],WPLNT[960],WPL[6][290]
        double ICOLNNNT[60]
        '''Null scatter sum for each gas at each sample. This is used for when EnableThermalMotion = 0'''
        double NumMomCrossSectionPointsNullNT
        '''Used to store the number of cross section points for null cross section points. This is used for when EnableThermalMotion = 0'''
        double RandomMaxBoltzArray[9]
        '''Stores randomly drawn max boltz numbers'''
        char BFieldMode
        '''Which integration mode to use'''
        Gasmix MixObject
        '''Gas mixer object'''
        double Steady_State_Threshold
        '''If this threshold is crossed by the difference of the reduced attachment and ionisation rate, the steady state simulation is done.'''
        double FakeIonisationsEstimate
        double TimeStep
        '''Time step between the planes in the pulsed townsend simulation that is used in the alpha calculation function'''
        double MaxTime
        '''Maximum amount of time for the alpha calculation function (time at the final plane), it is equal to the number of steps * Time step'''
        int NumberOfTimeSteps
        '''Number of time planes in the pulsed townsend  simulation'''
        int NumberOfSpaceSteps
        '''Number of space planes in the steady state simulation'''
        double TotalTimePrimary
        '''Total time of collisions for all the primaries in a simulation'''
        double TotalTimeSecondary
        '''Total time of collisions for all the secondaries in a simulation'''
        int IPrimary
        '''Number of primaries simulated in each alpha calculation simulation.'''
        double TotalSpaceZPrimary
        '''Total amount of space travelled by all the primaries in the simulation. In the Z direction'''
        double TotalSpaceZSecondary
        '''Total amount of space travelled by all the secondary in the simulation. In the Z direction'''
        double SpaceStepZ
        '''Space step between the space planes in the steady state simulation. In the Z direction'''
        double MaxSpaceZ
        '''Maximum amount of space for the alpha calculation function (space at the final plane), it is equal to the number of steps * space step. In the Z direction'''
        # Space planes output
        double SXPlanes[9]
        '''Sum of time in x movement at space plane I. This value is equal to the sum of the x values over the velocity in the z direction at space plane I.'''
        double SYPlanes[9]
        '''Sum of time in y movement at space plane I. This value is equal to the sum of the y values over the velocity in the z direction at space plane I.'''
        double SZPlanes[9]
        '''Sum of time in z movement at space plane I. This value is equal to the sum of the z values over the velocity in the z direction at space plane I.'''
        double STPlanes[9]
        '''Sum of the recipical of distances at space plane I.'''
        double SEPlanes[9]
        '''Sum of energy normalised over time at space plane I.'''
        double SX2Planes[9]
        '''Sum of time in x^2 movement at space plane I. This value is equal to the sum of the x^2 values over the velocity in the z direction at space plane I.'''
        double SY2Planes[9]
        '''Sum of time in y^2 movement at space plane I. This value is equal to the sum of the x^2 values over the velocity in the z direction at space plane I.'''
        double SZ2Planes[9]
        '''Sum of time in z^2 movement at space plane I. This value is equal to the sum of the x^2 values over the velocity in the z direction at space plane I.'''
        double SVZPlanes[9]
        '''Number of electrons with a velocity bigger than 0 in the z direction at space plane I.'''
        double STSPlanes[9]
        '''Sum of the recipocal of the velocity in the z direction at space plane I.'''
        double STS2Planes[9]
        '''Sum of the square of the recipocal of the velocity in the z direction at space plane I. '''
        double AttachmentOverIonisation
        '''Number of attached electrons over the number of ionising electrons.'''
        double AttachmentOverIonisationErr
        '''Error of the ratio of attached electrons over the nubmer of ionised electrons.'''
        double AttachmentErr
        '''Error of the number of attached electrons.'''
        double NumberOfElectronSST[11]
        '''Number of electrons at plane I. This doesn't include attached electrons.'''
        double TMSPL[9]
        double TTMSPL[9]
        double SRPlanes[9]
        '''The radial distance in the x,y plane at space plane I.'''
        double SR2Planes[9]
        '''The radial distance squared in the x,y plane at space plane I.'''
        # Steady State output
        double VelocitySvzSST
        '''Velocity output in the Z direction. This is calculated by multiplying SVZPlanes by STSPlanes at space plane I.'''
        double VelocitySvzSSTErr
        '''Velocity output error in the Z direction.'''
        double WSOUT
        double WSERR
        double LongitudinalDiffSST
        '''Longitudinal diffusion output for the Steady State simulation.'''
        double LongitudinalDiffSSTErr
        '''Error for the Longitudinal diffusion output of the Steady State simulation.'''
        double TransverseDiffSST
        '''Transverse diffusion output for the Steady State simulation'''
        double TransverseDiffSSTErr
        '''Error for the Transverse diffusion output for the Steady State simulation.'''
        double AlphaSST
        '''Alpha output from the Steady State simulation.'''
        double AlphaSSTErr
        '''Error for the Alpha output from the Steady State simulation.'''
        double AttachmentSST
        '''Attachment output from the Steady State simulation.'''
        double AttachmentSSTErr
        '''Error for the Sttachment output from the Steady State simulation.'''
        #Time planes output
        double TXPlanes[9]
        '''Sum of the X position of each electron in the pulsed townsend simulation.'''
        double TYPlanes[9]
        '''Sum of the Y position of each electron in the pulsed townsend simulation.'''
        double TZPlanes[9]
        '''Sum of the Z position of each electron in the pulsed townsend simulation.'''
        double TTPlanes[9]
        '''Sum of the total time of each electron in the pulsed townsend simulation.'''
        double TEPlanes[9]
        '''Sum of the Z position of each electron in the pulsed townsend simulation.'''
        double TX2Planes[9]
        '''Sum of the X^2 position of each electron in the pulsed townsend simulation.'''
        double TY2Planes[9]
        '''Sum of the Y^2 position of each electron in the pulsed townsend simulation.'''
        double TZ2Planes[9]
        '''Sum of the Z^2 position of each electron in the pulsed townsend simulation.'''
        double TVZPlanes[9]
        '''Sum of the Vz position of each electron in the pulsed townsend simulation.'''
        double NumberOfElectronsPlanes[9]
        '''Number of electrons accounted for in each time planes.'''
        double AttachmentOverIonisationPT
        '''Number of attached electrons over the number of ionising electrons.'''
        double AttachmentErrPT
        '''Error of the number of attached electrons.'''
        double AttachmentOverIonisationErrPT
        '''Error of the ratio of attached electrons over the nubmer of ionised electrons.'''


        # Pulsed Townsend
        double RealIonisation[9]
        '''Average number of real Ionisations at time plane I.'''
        double EnergyPT[9]
        '''Average energy at time plane I.'''
        double VelocityZPT[9]
        '''Average velocity at time plane I.'''
        double AverageTimePT[9]
        '''Average Time at time plane I.'''

        # Time of flight
        double ReducedAlphaTOF
        '''Time of flight reduced alpha calculation.'''
        double ReducedAlphaTOFErr
        '''Time of flight reduced alpha calculation err.'''
        double TOFEnergy
        '''Time of flight energy. '''
        double TOFEnergyErr
        '''Time of flight energy error.'''
        double VelocityTOFPT
        '''Time of flight velocity from the pulsed townsend calculation.'''
        double VelocityTOFPTErr
        '''Time of flight velocity from the pulsed townsend calculation error.'''
        double LongitudinalDiffusionTOF
        '''Time of flight longitudinal diffusion.'''
        double LongitudinalDiffusionTOFErr
        '''Time of flight longitudinal diffusion err.'''
        double TransverseDiffusionTOF
        '''Time of flight transverse diffusion.'''
        double TransverseDiffusionTOFErr
        '''Time of flight transverse diffusion err.'''
        double VelocityTOF
        '''Time of flight velocity taken from the Time of flight calculation.'''
        double VelocityTOFErr
        '''Time of flight velocity taken from the Time of flight calculation err.'''
        double ReducedAttachmentTOF
        '''Time of flight reduced attachment coeffiecent.'''
        double ReducedAttachmentTOFErr
        '''Time of flight reduced attachment coeffiecent err.'''

        # Alpha, and ATt
        double ALPHA,ATT,ALPER,ATTER,ALPTEST

        int ElasticColli[6]
        '''Number of elastic collisions that occured in the simulation.'''
        int InelasticColli[6]
        '''Number of inelastic collisions that occured in the simulation.'''
        int AttachmentColli[6]
        '''Number of attachment collisions that occured in the simulation.'''
        int IonisationColli[6]
        '''Number of ionisation collisions that occured in the simulation.'''
        int SuperElasticColli[6]
        '''Number of super elastic collisions that occured in the simulation.'''

        # FCION and FCATT
        double IonCollisionFreq[4000]
        '''Array that adds the ionisation collision frequencies at energy step I. Used in the Friedland estimation of Alpha.'''
        double AttCollisionFreq[4000]
        '''Array that adds the attachment collision frequencies at energy step I. Used in the Friedland estimation of Alpha.'''
