import math

from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from libc.string cimport memset
from PyBoltz import Setups
from PyBoltz import Mixers
from PyBoltz import EnergyLimits
import PyBoltz.MonteFuncs
import PyBoltz.Townsend
cimport PyBoltz.MonteFuncs
cimport PyBoltz.Townsend
from PyBoltz.MonteFuncs cimport MONTE,MONTET,MONTEB,MONTEBT,MONTEC,MONTECT
from PyBoltz.Townsend cimport ALPCALCT
from PyGasMix.Gasmix cimport Gasmix



cdef extern from "C/RM48.h":
    double DRAND48(double dummy)
    void RM48(double lenv)
    void RM48IN(int IJL, int NTOT, intNTOT2)


cdef double drand48(double dummy):
    return DRAND48(dummy)

cdef void setSeed(int seed):
    RM48IN(seed, 0, 0)
    return
cdef class Boltz:
    """
    This is the main object used to start the simulation, as well as store the information of the simulation.
    It has most of the needed arrays, and variables.
    More about Magboltz:
    `Magboltz_Documentation <http://cyclo.mit.edu/drift/www/aboutMagboltz.html/>`_
    .. note::
        If the variable has a "NT" at the end, that variable has the same function as its counterpart without a "NT" at the end.
    """

    def __init__(self):
        '''
        Fill all the variables needed with zeros.This function uses memset as it is fast.
        '''
        memset(self.CollisionFrequencyNT, 0, 4000 * 960 * sizeof(double))
        memset(self.EnergyLevelsNT, 0, 960 * sizeof(double))
        memset(self.TotalCollisionFrequencyNT, 0, 4000 * sizeof(double))
        memset(self.InteractionTypeNT, 0, 960 * sizeof(double))
        memset(self.RGasNT, 0, 960 * sizeof(double))
        memset(self.ElectronNumChangeNT, 0, 960 * sizeof(double))
        memset(self.PenningFractionNT, 0, 3 * 960 * sizeof(double))
        memset(self.MaxCollisionFreqNT, 0, 8 * sizeof(double))
        memset(self.NullCollisionFreqNT, 0, 4000 * 60 * sizeof(double))
        memset(self.TotalCollisionFrequencyNullNT, 0, 4000 * sizeof(double))
        memset(self.ScaleNullNT, 0, 60 * sizeof(double))
        memset(self.ScatteringParameterNT, 0, 4000 * 960 * sizeof(double))
        memset(self.AngleCutNT, 0, 4000 * 960 * sizeof(double))
        memset(self.AngularModelNT, 0, 960 * sizeof(double))
        memset(self.NC0NT, 0, 960 * sizeof(double))
        memset(self.CollisionEnergies, 0, 4000 * sizeof(double))
        memset(self.CollisionTimes, 0, 300 * sizeof(double))
        memset(self.CollisionsPerGasPerType, 0, 6 * 5 * sizeof(double))
        memset(self.CollisionsPerGasPerTypeNT, 0, 30 * sizeof(double))
        memset(self.ICOLNN, 0, 6 * 10 * sizeof(double))
        memset(self.ICOLN, 0, 6 * 290 * sizeof(double))
        memset(self.ICOLNNNT, 0, 60 * sizeof(double))
        memset(self.ICOLNNT, 0, 960 * sizeof(double))
        memset(self.AMGAS, 0, 6 * sizeof(double))
        memset(self.VTMB, 0, 6 * sizeof(double))
        memset(self.MaxCollisionFreqTotalG, 0, 6 * sizeof(double))
        memset(self.GasIDs, 0, 6 * sizeof(double))
        memset(self.GasFractions, 0, 6 * sizeof(double))
        memset(self.MoleculesPerCm3PerGas, 0, 6 * sizeof(double))
        memset(self.VMoleculesPerCm3PerGas, 0, 6 * sizeof(double))
        memset(self.CrossSectionSum, 0, 4000 * sizeof(double))
        memset(self.IonizationCrossSection, 0, 6 * 4000 * sizeof(double))
        memset(self.InelasticCrossSectionPerGas, 0, 6 * 250 * 4000 * sizeof(double))
        memset(self.E, 0, 4000 * sizeof(double))
        memset(self.SqrtEnergy, 0, 4000 * sizeof(double))
        memset(self.TotalCrossSection, 0, 4000 * sizeof(double))
        memset(self.RelativeIonMinusAttachCrossSection, 0, 4000 * sizeof(double))
        memset(self.InelasticCrossSection, 0, 4000 * sizeof(double))
        memset(self.N_Inelastic, 0, 6 * sizeof(double))
        memset(self.CollisionFrequency, 0, 6 * 290 * 4000 * sizeof(double))
        memset(self.TotalCollisionFrequency, 0, 6 * 4000 * sizeof(double))
        memset(self.EnergyLevels, 0, 6 * 290 * sizeof(double))
        memset(self.InteractionType, 0, 6 * 290 * sizeof(double))
        memset(self.RGas, 0, 6 * 290 * sizeof(double))
        memset(self.ElectronNumChange, 0, 6 * 290 * sizeof(double))
        memset(self.NumMomCrossSectionPoints, 0, 6 * sizeof(double))
        memset(self.ISIZE, 0, 6 * sizeof(double))
        memset(self.PenningFraction, 0, 6 * 290 * 3 * sizeof(double))
        memset(self.MaxCollisionFreq, 0, 6 * sizeof(double))
        memset(self.NullCollisionFreq, 0, 6 * 10 * 4000 * sizeof(double))
        memset(self.TotalCollisionFrequencyNull, 0, 6 * 4000 * sizeof(double))
        memset(self.ScaleNull, 0, 6 * 10 * sizeof(double))
        memset(self.NumMomCrossSectionPointsNull, 0, 6 * sizeof(double))
        memset(self.ScatteringParameter, 0, 6 * 290 * 4000 * sizeof(double))
        memset(self.AngleCut, 0, 6 * 290 * 4000 * sizeof(double))
        memset(self.AngularModel, 0, 6 * 290 * sizeof(double))
        memset(self.NC0, 0, 6 * 290 * sizeof(double))
        memset(self.ElasticCrossSection, 0, 4000 * sizeof(double))
        memset(self.AttachmentSectionSum, 0, 4000 * sizeof(double))
        memset(self.ElasticColli, 0, 6 * sizeof(int))
        memset(self.InelasticColli, 0, 6 * sizeof(int))
        memset(self.AttachmentColli, 0, 6 * sizeof(int))
        memset(self.IonisationColli, 0, 6 * sizeof(int))
        memset(self.SuperElasticColli, 0, 6 * sizeof(int))

        # Input parameters / settings
        self.Enable_Thermal_Motion = 0.0
        self.MaxNumberOfCollisions = 0.0
        self.BField_Angle = 0.0
        self.BField_Mag = 0.0
        self.NumberOfGases = 0
        self.Which_Angular_Model = 2
        self.TemperatureCentigrade = 0.0
        self.Pressure_Torr = 0.0
        self.Enable_Penning = 0
        self.EField = 0.0
        self.Num_Samples = 10
        self.Decor_Colls = 0
        self.Decor_Step = 0
        self.Decor_Lookbacks = 0

        # Calculated Constants
        self.CONST1 = 0.0
        self.CONST2 = 0.0
        self.CONST3 = 0.0
        self.PIR2 = 0.0
        self.RhydbergConst = 0.0
        self.EFieldOverBField = 0.0
        self.AngularSpeedOfRotation = 0.0
        self.ThermalEnergy = 0.0
        self.MaxCollisionTime = 100.0
        self.SmallNumber = 1e-20
        self.AngleFromX = 0.0
        self.PresTempCor = 0.0

        # Dynamically set
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.EnergySteps = 0
        self.AnisotropicDetected = 0
        self.Max_Electron_Energy = 0.0
        self.ElectronEnergyStep = 0
        self.InitialElectronEnergy = 0.0
        self.AngleFromZ = 0.0
        self.TimeSum = 0.0
        self.MaxCollisionFreqTotal = 0.0

        # Outputs
        self.MeanElectronEnergyError = 0.0
        self.MeanElectronEnergy = 0.0
        self.VelocityX = 0.0
        self.VelocityY = 0.0
        self.VelocityZ = 0.0
        self.VelocityErrorX = 0.0
        self.VelocityErrorY = 0.0
        self.VelocityErrorZ = 0.0
        self.AttachmentRate = 0.0
        self.IonisationRate = 0.0
        self.IonisationRateError = 0.0
        self.AttachmentRateError = 0.0
        self.LongitudinalDiffusion = 0.0
        self.LongitudinalDiffusionError = 0.0
        self.TransverseDiffusion = 0.0
        self.TransverseDiffusionError = 0.0
        self.LongitudinalDiffusion1 = 0.0
        self.LongitudinalDiffusion1Error = 0.0
        self.TransverseDiffusion1 = 0.0
        self.TransverseDiffusion1Error = 0.0
        self.DiffusionX = 0.0
        self.DiffusionY = 0.0
        self.DiffusionYZ = 0.0
        self.DiffusionYZ = 0.0
        self.DiffusionXY = 0.0
        self.DiffusionXZ = 0.0
        self.ErrorDiffusionX = 0.0
        self.ErrorDiffusionY = 0.0
        self.ErrorDiffusionZ = 0.0
        self.ErrorDiffusionYZ = 0.0
        self.ErrorDiffusionXY = 0.0
        self.ErrorDiffusionXZ = 0.0
        self.FakeIonizations = 0
        self.Random_Seed = 54217137
        self.Console_Output_Flag = 1
        self.MeanCollisionTime = 0.0
        self.AlphaSST = 0.0
        self.AlphaSSTErr = 0.0
        self.AttachmentSST = 0.0
        self.AttachmentSSTErr = 0.0
        self.ReducedIonization = 0.0
        self.ReducedAttachment = 0.0
        self.ReducedIonizationErr = 0.0
        self.ReducedAttachmentErr = 0.0
        self.Steady_State_Threshold = 40.0
        self.MixObject = Gasmix()

    def reset(self):
        cdef int I, J

        self.TimeSum = 0.0
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0

        self.TotalTimePrimary = 0.0
        self.TotalTimeSecondary = 0.0
        self.TotalSpaceZPrimary = 0.0
        self.TotalSpaceZSecondary = 0.0

        for I in range(6):
            for J in range(5):
                self.CollisionsPerGasPerType[I][J] = 0.0
            for J in range(290):
                self.ICOLN[I][J] = 0.0
            for J in range(10):
                self.ICOLNN[I][J] = 0.0

        for I in range(4000):
            self.CollisionEnergies[I] = 0.0
        for I in range(299):
            self.CollisionTimes[I]=0.0


    def end(self):
        """
        This function is used to convert some of the output values into different units.
        """

        # Calculate temprature in kelvin
        self.TemperatureKelvin = self.TemperatureCentigrade + 273.15

        cdef double DUM[6]
        if self.VelocityZ != 0:
            self.TransverseDiffusion1 = sqrt(2.0 * self.TransverseDiffusion / self.VelocityZ) * 10000.0
            self.TransverseDiffusion1Error = math.sqrt(
                self.TransverseDiffusionError ** 2 + self.VelocityErrorZ ** 2) / 2.0

            self.LongitudinalDiffusion1 = sqrt(2.0 * self.LongitudinalDiffusion / self.VelocityZ) * 10000.0
            self.LongitudinalDiffusion1Error = math.sqrt(
                self.LongitudinalDiffusionError ** 2 + self.VelocityErrorZ ** 2) / 2.0

            self.VelocityZ *= 1e-5
            self.VelocityY *= 1e-5
            self.VelocityX *= 1e-5

            self.ReducedIonization = self.IonisationRate /( 760 * self.TemperatureKelvin / (self.Pressure_Torr * 293.15))
            self.ReducedIonizationErr = self.IonisationRateError /( 760 * self.TemperatureKelvin / (self.Pressure_Torr * 293.15))

            self.ReducedAttachment = self.AttachmentRate /( 760 * self.TemperatureKelvin / (self.Pressure_Torr * 293.15))
            self.ReducedAttachmentErr = self.AttachmentRateError /( 760 * self.TemperatureKelvin / (self.Pressure_Torr * 293.15))


        # The different counters for collision types.
        if self.Enable_Thermal_Motion ==1:
            for I in range(6):
                self.ElasticColli[I] = <int>self.CollisionsPerGasPerType[I][0]
                self.InelasticColli[I] = <int>self.CollisionsPerGasPerType[I][3]
                self.AttachmentColli[I] = <int>self.CollisionsPerGasPerType[I][2]
                self.IonisationColli[I] = <int>self.CollisionsPerGasPerType[I][1]
                self.SuperElasticColli[I] = <int>self.CollisionsPerGasPerType[I][4]
        else:
            for I in range(6):
                self.ElasticColli[I] = <int>self.CollisionsPerGasPerTypeNT[0]
                self.InelasticColli[I] = <int>self.CollisionsPerGasPerTypeNT[3]
                self.AttachmentColli[I] = <int>self.CollisionsPerGasPerTypeNT[2]
                self.IonisationColli[I] = <int>self.CollisionsPerGasPerTypeNT[1]
                self.SuperElasticColli[I] = <int>self.CollisionsPerGasPerTypeNT[4]

    cpdef GetSimFunctions(self, BFieldMag, BFieldAngle, EnableThermalMotion):
        """
        This function picks which sim functions to use, depending on applied fields and thermal motion.
        """
        ELimFunc = EnergyLimits.EnergyLimitT
        MonteCarloFunc = PyBoltz.MonteFuncs.MONTET
        TownsendFunc = PyBoltz.Townsend.ALPCALCT
        if (self.Enable_Thermal_Motion != 0):
            MixerFunc = Mixers.MixerT
            if BFieldMag == 0:
                self.BFieldMode = 1
                ELimFunc = EnergyLimits.EnergyLimitT
                MonteCarloFunc = PyBoltz.MonteFuncs.MONTET
            elif BFieldAngle == 0 or BFieldAngle == 180:
                self.BFieldMode = 2
                ELimFunc = EnergyLimits.EnergyLimitT
                MonteCarloFunc = PyBoltz.MonteFuncs.MONTET
            elif BFieldAngle == 90:
                ELimFunc = EnergyLimits.EnergyLimitBT
                MonteCarloFunc = PyBoltz.MonteFuncs.MONTEBT
            else:
                ELimFunc = EnergyLimits.EnergyLimitCT
                MonteCarloFunc = PyBoltz.MonteFuncs.MONTECT
        else:
            MixerFunc = Mixers.Mixer
            if BFieldMag == 0:
                self.BFieldMode = 1
                ELimFunc = EnergyLimits.EnergyLimit
                MonteCarloFunc = PyBoltz.MonteFuncs.MONTE
            elif BFieldAngle == 0 or BFieldAngle == 180:
                self.BFieldMode = 2
                ELimFunc = EnergyLimits.EnergyLimit
                MonteCarloFunc = PyBoltz.MonteFuncs.MONTE
            elif BFieldAngle == 90:
                ELimFunc = EnergyLimits.EnergyLimitB
                MonteCarloFunc = PyBoltz.MonteFuncs.MONTEB
            else:
                ELimFunc = EnergyLimits.EnergyLimitC
                MonteCarloFunc = PyBoltz.MonteFuncs.MONTEC
        return [MixerFunc, ELimFunc, MonteCarloFunc, TownsendFunc]

    def SetExtraParameters(self, params):
        self.MixObject.ExtraParameters = params
    cpdef Start(self):
        """
        This is the main function that starts the magboltz simulation/calculation.
        The simulation starts by calculating the momentum cross section values for the requested gas mixture. If FinalElectronEnergy
        is equal to 0.0 it will then keep calling the EnergyLimit functions and the Mixer functions to find the electron
        Integration limit.
        Finally Boltz calls the Monte carlo functions, which is where the main simulation happens. The outputs are stored
        in the the parent object of this function.
        For more info on the main output variables check the git repository readme:
        `Boltz repository <https://github.com/UTA-REST/MAGBOLTZ-py/>`_
        """
        setSeed(self.Random_Seed)
        ELimNotYetFixed = 1

        cdef double ReducedField
        cdef int i = 0

        # Get the appropriate set of simulation functions given configuration keys
        MixerFunc, ELimFunc, MonteCarloFunc, TownsendFunc = self.GetSimFunctions(self.BField_Mag, self.BField_Angle,
                                                                                 self.Enable_Thermal_Motion)

        # Set up the simulation
        Setups.Setup(self)

        # Find the electron upper energy limit
        if self.Max_Electron_Energy == 0.0:
            # Given no specified upper energy limit, find it iteratively
            self.Max_Electron_Energy = 0.5
            ReducedField = self.EField * (self.TemperatureCentigrade + 273.15) / (self.Pressure_Torr * 293.15)
            if ReducedField > 15:
                self.Max_Electron_Energy = 8.0
            self.InitialElectronEnergy = self.Max_Electron_Energy / 50.0
            while ELimNotYetFixed == 1:
                MixerFunc(self)
                ELimNotYetFixed = ELimFunc(self)
                if ELimNotYetFixed == 1:
                    self.Max_Electron_Energy = self.Max_Electron_Energy * math.sqrt(2)
                    self.InitialElectronEnergy = self.Max_Electron_Energy / 50
        else:
            # Given a specified upper energy limit, use it
            MixerFunc(self)
        print("")
        if self.Console_Output_Flag: print("Calculated the final energy = " + str(self.Max_Electron_Energy))

        # Run the simulation

        ELimNotYetFixed=1
        while ELimNotYetFixed==1:
            try:
                MonteCarloFunc.run(self)
            except ValueError as err:
                if "ENERGY INTEGRATION RANGE" in str(err).upper():
                    print("Electron max energy " + str(round(self.Max_Electron_Energy,2)) +" exceeded, increasing to " + str(round(self.Max_Electron_Energy * math.sqrt(2),2)) +" and trying again")
                    self.Max_Electron_Energy = self.Max_Electron_Energy * math.sqrt(2)
                    self.reset()
                    MixerFunc(self)
                else:
                    raise err
            else:
                ELimNotYetFixed=0

        # Closeout and end
        self.end()
        # Steady state
        if abs(self.ReducedIonization - self.ReducedAttachment) >= self.Steady_State_Threshold:
            if self.ReducedIonization ==0:
                print("Steady State Threshold has been crossed. Will not run the SST simulation as the ionisation rate is zero.")
                return
            if self.Console_Output_Flag: print("\n**Crossed the set Steady state simulation threshold = {}\n".format(self.Steady_State_Threshold))
            TownsendFunc.run(self)

        return
