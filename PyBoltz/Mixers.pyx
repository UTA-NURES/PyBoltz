from PyBoltz.Boltz cimport Boltz
from libc.math cimport sin, cos, acos, asin, log, sqrt
from PyGasMix.Gasmix cimport Gasmix
from PyBoltz.Ang cimport Ang
import sys
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef Mixer(Boltz object):
    """
    This function sets up the given Boltz object with the right values for the requested gas mixture. It uses the Gasmix object
    to get the momentum cross sections and all the needed values.

    The object parameter is the Boltz object to be setup.
    """
    cdef double AttachmentCrossSection[6][4000], ElectronCharge, JHI, JLOW, EnergyHigh, F2, BP, EnergyLow
    cdef int  iEnergy, GasIndex, iProcess, p, sum, J, i, j, iIonizaton, JJ, IL, I

    # Define a couple of useful constants
    ElectronCharge = 1.602176565e-19
    ElectronMass = 9.10938291e-31

    object.ElectronEnergyStep = object.Max_Electron_Energy / object.EnergySteps

    EnergyHalf = object.ElectronEnergyStep / 2
    object.E[0] = EnergyHalf

    # Set up a GasMix object and populate its data arrays
    for i in range(1, 4000):
        object.E[i] = EnergyHalf + object.ElectronEnergyStep * i
        object.SqrtEnergy[i] = sqrt(object.E[i])
    object.SqrtEnergy[0] = sqrt(EnergyHalf)
    object.MixObject.InitWithInfo(object.GasIDs, object.InelasticCrossSectionPerGas, object.N_Inelastic,
                                  object.PenningFraction,
                                  object.E, object.SqrtEnergy,
                                  object.NumberOfGases, object.EnergySteps, object.Which_Angular_Model,
                                  object.ElectronEnergyStep,
                                  object.Max_Electron_Energy, object.ThermalEnergy, object.TemperatureCentigrade,
                                  object.Pressure_Torr, object.PIR2, object.RhydbergConst)
    object.MixObject.Run()

    for iEnergy in range(4000):
        object.IonCollisionFreq[iEnergy] = 0.0
        object.AttCollisionFreq[iEnergy] = 0.0
        iProcess = 0
        for GasIndex in range(object.NumberOfGases):
            object.CollisionFrequencyNT[iEnergy][iProcess] = object.MixObject.Gases[GasIndex].Q[1][iEnergy] * \
                                                             object.VMoleculesPerCm3PerGas[GasIndex]
            object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
            object.AngleCutNT[iEnergy][iProcess] = 1
            object.AngularModelNT[iProcess] = 0
            AngObject = Ang()

            if object.MixObject.Gases[GasIndex].AngularModel[1] == 1:
                ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEElasticCrossSection[1][iEnergy]
                AngObject.ScatteringParameter1 = ScatteringParameter1
                AngObject.CalcAngCut()
                object.AngleCutNT[iEnergy][iProcess] = AngObject.AngCut
                object.ScatteringParameterNT[iEnergy][iProcess] = AngObject.ScatteringParameter2
                object.AngularModelNT[iProcess] = 1
            elif object.MixObject.Gases[GasIndex].AngularModel[1] == 2:
                object.ScatteringParameterNT[iEnergy][iProcess] = \
                object.MixObject.Gases[GasIndex].PEElasticCrossSection[1][iEnergy]
                object.AngularModelNT[iProcess] = 2

            if iEnergy == 0:
                RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                object.RGasNT[iProcess] = RGas
                L = 1
                object.InteractionTypeNT[iProcess] = L
                object.EnergyLevelsNT[iProcess] = 0.0
                object.ElectronNumChangeNT[iProcess] = 0

                object.PenningFractionNT[0][iProcess] = 0.0
                object.PenningFractionNT[1][iProcess] = 0.0
                object.PenningFractionNT[2][iProcess] = 0.0
                # IONISATION

            if object.Max_Electron_Energy >= object.MixObject.Gases[GasIndex].E[2]:
                if object.MixObject.Gases[GasIndex].N_Ionization <= 1:
                    iProcess += 1
                    object.CollisionFrequencyNT[iEnergy][iProcess] = object.MixObject.Gases[GasIndex].Q[2][iEnergy] * \
                                                                     object.VMoleculesPerCm3PerGas[GasIndex]
                    object.IonCollisionFreq[iEnergy] += object.CollisionFrequencyNT[iEnergy][iProcess - 1]
                    object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                    object.AngleCutNT[iEnergy][iProcess] = 1.0
                    object.AngularModelNT[iProcess] = 0
                    if object.MixObject.Gases[GasIndex].AngularModel[2] == 1:
                        ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEElasticCrossSection[2][iEnergy]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.CalcAngCut()
                        object.AngleCutNT[iEnergy][iProcess] = AngObject.AngCut
                        object.ScatteringParameterNT[iEnergy][iProcess] = AngObject.ScatteringParameter2
                        object.AngularModelNT[iProcess] = 1
                    elif object.MixObject.Gases[GasIndex].AngularModel[2] == 2:
                        object.ScatteringParameterNT[iEnergy][iProcess] = \
                        object.MixObject.Gases[GasIndex].PEElasticCrossSection[2][iEnergy]
                        object.AngularModelNT[iProcess] = 2
                elif object.MixObject.Gases[GasIndex].N_Ionization > 1:
                    for iIonizaton in range(object.MixObject.Gases[GasIndex].N_Ionization):
                        iProcess += 1
                        object.CollisionFrequencyNT[iEnergy][iProcess] = \
                        object.MixObject.Gases[GasIndex].IonizationCrossSection[iIonizaton][iEnergy] * \
                        object.VMoleculesPerCm3PerGas[GasIndex]
                        object.IonCollisionFreq[iEnergy] += object.CollisionFrequencyNT[iEnergy][iProcess - 1]
                        object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                        object.AngleCutNT[iEnergy][iProcess] = 1.0
                        object.AngularModelNT[iProcess] = 0
                        if object.MixObject.Gases[GasIndex].AngularModel[2] == 1:
                            ScatteringParameter1 = \
                            object.MixObject.Gases[GasIndex].PEIonizationCrossSection[iIonizaton][iEnergy]
                            AngObject.ScatteringParameter1 = ScatteringParameter1
                            AngObject.CalcAngCut()
                            object.AngleCutNT[iEnergy][iProcess] = AngObject.AngCut
                            object.ScatteringParameterNT[iEnergy][iProcess] = AngObject.ScatteringParameter2
                            object.AngularModelNT[iProcess] = 1
                        elif object.MixObject.Gases[GasIndex].AngularModel[2] == 2:
                            object.ScatteringParameterNT[iEnergy][iProcess] = \
                            object.MixObject.Gases[GasIndex].PEIonizationCrossSection[iIonizaton][iEnergy]
                            object.AngularModelNT[iProcess] = 2

                if iEnergy == 0:
                    if object.MixObject.Gases[GasIndex].N_Ionization <= 1:
                        RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGasNT[iProcess] = RGas
                        object.EnergyLevelsNT[iProcess] = object.MixObject.Gases[GasIndex].E[2] / RGas

                        object.NC0NT[iProcess] = object.MixObject.Gases[GasIndex].NC0[0]
                        object.EC0NT[iProcess] = object.MixObject.Gases[GasIndex].EC0[0]
                        object.WPLNT[iProcess] = object.MixObject.Gases[GasIndex].EOBY[0]
                        object.ElectronNumChangeNT[iProcess] = 1
                        L = 2
                        object.InteractionTypeNT[iProcess] = L
                        object.PenningFractionNT[0][iProcess] = 0.0
                        object.PenningFractionNT[1][iProcess] = 0.0
                        object.PenningFractionNT[2][iProcess] = 0.0
                    elif object.MixObject.Gases[GasIndex].N_Ionization > 1:
                        iProcess = iProcess - object.MixObject.Gases[GasIndex].N_Ionization
                        for iIonizaton in range(object.MixObject.Gases[GasIndex].N_Ionization):
                            iProcess = iProcess + 1
                            RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                            object.RGasNT[iProcess] = RGas
                            object.EnergyLevelsNT[iProcess] = object.MixObject.Gases[GasIndex].IonizationEnergy[
                                                                  iIonizaton] / RGas
                            object.EC0NT[iProcess] = object.MixObject.Gases[GasIndex].EC0[iIonizaton]
                            object.NC0NT[iProcess] = object.MixObject.Gases[GasIndex].NC0[iIonizaton]
                            object.WPLNT[iProcess ] = object.MixObject.Gases[GasIndex].EOBY[iIonizaton]
                            object.ElectronNumChangeNT[iProcess] = 1
                            L = 2
                            object.InteractionTypeNT[iProcess] = L
                            object.PenningFractionNT[0][iProcess] = 0.0
                            object.PenningFractionNT[1][iProcess] = 0.0
                            object.PenningFractionNT[2][iProcess] = 0.0

            if object.Max_Electron_Energy >= object.MixObject.Gases[GasIndex].E[3]:
                if object.MixObject.Gases[GasIndex].N_Attachment <= 1:
                    iProcess += 1
                    object.CollisionFrequencyNT[iEnergy][iProcess] = object.MixObject.Gases[GasIndex].Q[3][iEnergy] * \
                                                                     object.VMoleculesPerCm3PerGas[GasIndex]
                    object.AttCollisionFreq[iEnergy] += object.CollisionFrequencyNT[iEnergy][iProcess - 1]
                    object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                    object.AngleCutNT[iEnergy][iProcess] = 1.0
                    if iEnergy == 0:
                        RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGasNT[iProcess] = RGas
                        object.EnergyLevelsNT[iProcess] = 0.0
                        object.AngularModelNT[iProcess] = 0
                        object.ElectronNumChangeNT[iProcess] = -1
                        L = 3
                        object.InteractionTypeNT[iProcess] = L
                        object.PenningFractionNT[0][iProcess] = 0.0
                        object.PenningFractionNT[1][iProcess] = 0.0
                        object.PenningFractionNT[2][iProcess] = 0.0
                elif object.MixObject.Gases[GasIndex].N_Attachment > 1:
                    for JJ in range(int(object.MixObject.Gases[GasIndex].N_Attachment)):
                        iProcess += 1
                        object.CollisionFrequencyNT[iEnergy][iProcess] = \
                        object.MixObject.Gases[GasIndex].AttachmentCrossSection[JJ][iEnergy] * \
                        object.VMoleculesPerCm3PerGas[GasIndex]
                        object.AttCollisionFreq[iEnergy] += object.CollisionFrequencyNT[iEnergy][iProcess - 1]
                        object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                        object.AngleCutNT[iEnergy][iProcess] = 1.0
                        if iEnergy == 0:
                            RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                            object.RGasNT[iProcess] = RGas
                            object.EnergyLevelsNT[iProcess] = 0.0
                            object.AngularModelNT[iProcess] = 0
                            object.ElectronNumChangeNT[iProcess] = -1
                            L = 3
                            object.InteractionTypeNT[iProcess] = L
                            object.PenningFractionNT[0][iProcess] = 0.0
                            object.PenningFractionNT[1][iProcess] = 0.0
                            object.PenningFractionNT[2][iProcess] = 0.0

            # INELASTIC AND SUPERELASTIC
            if object.MixObject.Gases[GasIndex].N_Inelastic > 0:
                for J in range(int(object.MixObject.Gases[GasIndex].N_Inelastic)):
                    iProcess = iProcess + 1
                    object.CollisionFrequencyNT[iEnergy][iProcess] = \
                    object.MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][iEnergy] * \
                    object.VMoleculesPerCm3PerGas[GasIndex]
                    object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                    object.AngleCutNT[iEnergy][iProcess] = 1.0
                    object.AngularModelNT[iProcess] = 0
                    if object.MixObject.Gases[GasIndex].KIN[J] == 1:

                        ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][
                            iEnergy]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.CalcAngCut()
                        object.AngleCutNT[iEnergy][iProcess] = AngObject.AngCut
                        object.ScatteringParameterNT[iEnergy][iProcess] = AngObject.ScatteringParameter2
                        object.AngularModelNT[iProcess] = 1
                    elif object.MixObject.Gases[GasIndex].KIN[J] == 2:

                        object.ScatteringParameterNT[iEnergy][iProcess] = \
                        object.MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][iEnergy]
                        object.AngularModelNT[iProcess] = 2
                    if iEnergy == 0:

                        RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGasNT[iProcess] = RGas
                        object.EnergyLevelsNT[iProcess] = object.MixObject.Gases[GasIndex].EnergyLevels[J] / RGas
                        L = 4

                        if object.MixObject.Gases[GasIndex].EnergyLevels[J] < 0:
                            L = 5
                        object.ElectronNumChangeNT[iProcess] = 0
                        object.InteractionTypeNT[iProcess] = L
                        object.PenningFractionNT[0][iProcess] = object.MixObject.Gases[GasIndex].PenningFraction[0][J]
                        object.PenningFractionNT[1][iProcess] = object.MixObject.Gases[GasIndex].PenningFraction[1][
                                                                    J] * 1.0e-16 / sqrt(3)
                        object.PenningFractionNT[2][iProcess] = object.MixObject.Gases[GasIndex].PenningFraction[2][J]

            iProcess += 1

        object.NumMomCrossSectionPointsNT = iProcess
        object.ISIZENT = 1
        for I in range(1, 9):
            if object.NumMomCrossSectionPointsNT >= 2 ** I:
                object.ISIZENT = 2 ** I
            else:
                break

        # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
        object.TotalCollisionFrequencyNT[iEnergy] = 0.0
        for P in range(int(object.NumMomCrossSectionPointsNT)):
            object.TotalCollisionFrequencyNT[iEnergy] = object.TotalCollisionFrequencyNT[iEnergy] + \
                                                        object.CollisionFrequencyNT[iEnergy][P]
            if object.CollisionFrequencyNT[iEnergy][P] < 0:
                print("WARNING NEGATIVE COLLISION FREQUENCY")

        for P in range(int(object.NumMomCrossSectionPointsNT)):
            if object.TotalCollisionFrequencyNT[iEnergy] != 0.0:
                object.CollisionFrequencyNT[iEnergy][P] /= object.TotalCollisionFrequencyNT[iEnergy]
            else:
                object.CollisionFrequencyNT[iEnergy][P] = 0.0

        for P in range(1, int(object.NumMomCrossSectionPointsNT)):
            object.CollisionFrequencyNT[iEnergy][P] += object.CollisionFrequencyNT[iEnergy][P - 1]
        object.TotalCollisionFrequencyNT[iEnergy] *= object.SqrtEnergy[iEnergy]
        object.IonCollisionFreq[iEnergy] *= object.SqrtEnergy[iEnergy]
        object.AttCollisionFreq[iEnergy] *= object.SqrtEnergy[iEnergy]


        iProcess = 0
        object.NumMomCrossSectionPointsNullNT = 0
        N_NullSum = 0.0
        for I in range(object.NumberOfGases):
            N_NullSum += object.MixObject.Gases[GasIndex].N_Null
        if N_NullSum != 0:
            for I in range(object.NumberOfGases):
                if object.MixObject.Gases[GasIndex].N_Null > 0:
                    for J in range(object.MixObject.Gases[GasIndex].N_Null):
                        object.ScaleNullNT[iProcess] = object.MixObject.Gases[GasIndex].ScaleNull[J]
                        object.NullCollisionFreqNT[iEnergy][iProcess] = \
                        object.MixObject.Gases[GasIndex].NullCrossSection[J][iEnergy] * object.VMoleculesPerCm3PerGas[
                            GasIndex] * \
                        object.ScaleNullNT[iProcess]
                        iProcess += 1
            object.NumMomCrossSectionPointsNullNT = iProcess + 1
            object.TotalCollisionFrequencyNullNT[iEnergy] = 0.0
            for P in range(int(object.NumMomCrossSectionPointsNullNT)):
                object.TotalCollisionFrequencyNullNT[iEnergy] = object.TotalCollisionFrequencyNullNT[iEnergy] + \
                                                                object.NullCollisionFreqNT[iEnergy][P]
                if object.NullCollisionFreqNT[iEnergy][P] < 0:
                    print("WARNING NEGATIVE NULL COLLISION FREQUENCY")

            for P in range(int(object.NumMomCrossSectionPointsNullNT)):
                if object.TotalCollisionFrequencyNullNT[iEnergy] != 0.0:
                    object.NullCollisionFreqNT[iEnergy][P] /= object.TotalCollisionFrequencyNullNT[iEnergy]
                else:
                    object.NullCollisionFreqNT[iEnergy][P] = 0.0

            for P in range(1, int(object.NumMomCrossSectionPointsNullNT)):
                object.NullCollisionFreqNT[iEnergy][P] += object.NullCollisionFreqNT[iEnergy][P - 1]
            object.TotalCollisionFrequencyNullNT[iEnergy] *= object.SqrtEnergy[iEnergy]

    AngularModelSum = 0

    for GasIndex in range(object.NumberOfGases):
        for J in range(6):
            AngularModelSum += object.MixObject.Gases[GasIndex].AngularModel[J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            AngularModelSum += object.MixObject.Gases[GasIndex].KIN[J]

    if AngularModelSum > 0:
        object.AnisotropicDetected = 1

    BP = object.EField ** 2 * object.CONST1
    F2 = object.EField * object.CONST3
    EnergyLow = object.MaxCollisionTime * (object.MaxCollisionTime * BP - F2 * sqrt(
        0.5 * object.Max_Electron_Energy)) / object.ElectronEnergyStep - 1
    EnergyLow = min(EnergyLow, object.SmallNumber)
    EnergyHigh = object.MaxCollisionTime * (object.MaxCollisionTime * BP + F2 * sqrt(
        0.5 * object.Max_Electron_Energy)) / object.ElectronEnergyStep + 1
    if EnergyHigh > 10000:
        EnergyHigh = 10000
    for l in range(8):
        I = l + 1
        JLOW = 4000 - 500 * (9 - I) + 1 + int(EnergyLow)
        JHI = 4000 - 500 * (8 - I) + int(EnergyHigh)
        JLOW = max(JLOW, 0)
        JHI = min(JHI, 4000)
        for J in range(int(JLOW - 1), int(JHI)):
            if (object.TotalCollisionFrequencyNT[J] + object.TotalCollisionFrequencyNullNT[J] + abs(
                    object.FakeIonizations)) > object.MaxCollisionFreqNT[l]:
                object.MaxCollisionFreqNT[l] = object.TotalCollisionFrequencyNT[J] + \
                                               object.TotalCollisionFrequencyNullNT[J] + abs(object.FakeIonizations)
    for I in range(object.EnergySteps):
        object.TotalCrossSection[I] = object.MoleculesPerCm3PerGas[0] * object.MixObject.Gases[0].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[1] * object.MixObject.Gases[1].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[2] * object.MixObject.Gases[2].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[3] * object.MixObject.Gases[3].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[4] * object.MixObject.Gases[4].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[5] * object.MixObject.Gases[5].Q[0][I]
        object.ElasticCrossSection[I] = object.MoleculesPerCm3PerGas[0] * object.MixObject.Gases[0].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[1] * object.MixObject.Gases[1].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[2] * object.MixObject.Gases[2].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[3] * object.MixObject.Gases[3].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[4] * object.MixObject.Gases[4].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[5] * object.MixObject.Gases[5].Q[1][I]

        for GasIndex in range(object.NumberOfGases):
            object.IonizationCrossSection[GasIndex][I] = object.MixObject.Gases[GasIndex].Q[2][I] * \
                                                         object.MoleculesPerCm3PerGas[GasIndex]
            AttachmentCrossSection[GasIndex][I] = object.MixObject.Gases[GasIndex].Q[3][I] * \
                                                  object.MoleculesPerCm3PerGas[GasIndex]
            if object.MixObject.Gases[GasIndex].N_Ionization > 1:
                object.IonizationCrossSection[GasIndex][I] = 0.0
                for iIonizaton in range(object.MixObject.Gases[GasIndex].N_Ionization):
                    object.IonizationCrossSection[GasIndex][I] += \
                    object.MixObject.Gases[GasIndex].IonizationCrossSection[iIonizaton][I] * \
                    object.MoleculesPerCm3PerGas[GasIndex]
        object.RelativeIonMinusAttachCrossSection[I] = 0.0
        object.AttachmentSectionSum[I] = 0.0
        object.CrossSectionSum[I] = 0.0
        for J in range(object.NumberOfGases):
            object.CrossSectionSum[I] = object.CrossSectionSum[I] + object.IonizationCrossSection[J][I] + \
                                        AttachmentCrossSection[J][I]
            object.AttachmentSectionSum[I] = object.AttachmentSectionSum[I] + AttachmentCrossSection[J][I]
            object.RelativeIonMinusAttachCrossSection[I] = object.RelativeIonMinusAttachCrossSection[I] + \
                                                           object.IonizationCrossSection[J][I] - \
                                                           AttachmentCrossSection[J][I]
        for GasIndex in range(6):
            for J in range(int(object.MixObject.Gases[GasIndex].N_Inelastic)):
                object.CrossSectionSum[I] = object.CrossSectionSum[I] + \
                                            object.MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][I] * \
                                            object.MoleculesPerCm3PerGas[GasIndex]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef MixerT(Boltz object):
    """
    This function sets up the given Boltz object with the right values for the requested gas mixture. It uses the Gasmix object
    to get the momentum cross sections and all the needed values.

    The object parameter is the Boltz object to be setup.
    """
    cdef double AttachmentCrossSection[6][4000]
    cdef int  iEnergy, GasIndex, iProcess, p, sum, J, i, j, iIonizaton, JJ, IL, I
    ElectronCharge = 1.602176565e-19

    object.ElectronEnergyStep = object.Max_Electron_Energy / float(object.EnergySteps)

    EnergyHalf = object.ElectronEnergyStep / 2

    object.E[0] = EnergyHalf
    for i in range(1, 4000):
        object.E[i] = EnergyHalf + object.ElectronEnergyStep * i
        object.SqrtEnergy[i] = sqrt(object.E[i])
    object.SqrtEnergy[0] = sqrt(EnergyHalf)

    object.MixObject.InitWithInfo(object.GasIDs, object.InelasticCrossSectionPerGas, object.N_Inelastic,
                                  object.PenningFraction,
                                  object.E, object.SqrtEnergy,
                                  object.NumberOfGases, object.EnergySteps, object.Which_Angular_Model,
                                  object.ElectronEnergyStep,
                                  object.Max_Electron_Energy, object.ThermalEnergy, object.TemperatureCentigrade,
                                  object.Pressure_Torr, object.PIR2, object.RhydbergConst)
    object.MixObject.Run()

    #-----------------------------------------------------------------
    #     CALCULATION OF COLLISION FREQUENCIES FOR AN ARRAY OF
    #     ELECTRON ENERGiEnergyS IN THE RANGE ZERO TO FinalEnergy
    #
    #     L=1      ELASTIC NTH GAS
    #     L=2      IONISATION NTH GAS
    #     L=3      ATTACHMENT NTH GAS
    #     L=4      INELASTIC NTH GAS
    #     L=5      SUPERELASTIC NTH GAS
    #---------------------------------------------------------------
    ElectronMass = 9.10938291e-31

    for iEnergy in range(4000):
        object.IonCollisionFreq[iEnergy] = 0.0
        object.AttCollisionFreq[iEnergy] = 0.0
        for GasIndex in range(object.NumberOfGases):
            iProcess = 1

            object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1] = object.MixObject.Gases[GasIndex].Q[1][
                                                                             iEnergy] * object.VMoleculesPerCm3PerGas[
                                                                             GasIndex]
            object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = 0.5
            object.AngleCut[GasIndex][iEnergy][iProcess - 1] = 1
            object.AngularModel[GasIndex][iProcess - 1] = 0
            AngObject = Ang()

            if object.MixObject.Gases[GasIndex].AngularModel[1] == 1:
                ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEElasticCrossSection[1][iEnergy]
                AngObject.ScatteringParameter1 = ScatteringParameter1
                AngObject.CalcAngCut()
                object.AngleCut[GasIndex][iEnergy][iProcess - 1] = AngObject.AngCut
                object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = AngObject.ScatteringParameter2
                object.AngularModel[GasIndex][iProcess - 1] = 1
            elif object.MixObject.Gases[GasIndex].AngularModel[1] == 2:
                object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = \
                object.MixObject.Gases[GasIndex].PEElasticCrossSection[1][iEnergy]
                object.AngularModel[GasIndex][iProcess - 1] = 2

            if iEnergy == 0:
                RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                object.AMGAS[GasIndex] = 2 * ElectronMass / object.MixObject.Gases[GasIndex].E[1]
                object.RGas[GasIndex][iProcess - 1] = RGas
                L = 1
                object.InteractionType[GasIndex][iProcess - 1] = L
                object.EnergyLevels[GasIndex][iProcess - 1] = 0.0
                object.ElectronNumChange[GasIndex][iProcess - 1] = 0

                object.PenningFraction[GasIndex][0][iProcess - 1] = 0.0
                object.PenningFraction[GasIndex][1][iProcess - 1] = 0.0
                object.PenningFraction[GasIndex][2][iProcess - 1] = 0.0

            # IONISATION
            if object.Max_Electron_Energy >= object.MixObject.Gases[GasIndex].E[2]:
                if object.MixObject.Gases[GasIndex].N_Ionization <= 1:
                    iProcess += 1
                    object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1] = object.MixObject.Gases[GasIndex].Q[2][
                                                                                     iEnergy] * \
                                                                                 object.VMoleculesPerCm3PerGas[GasIndex]
                    object.IonCollisionFreq[iEnergy] += object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1]
                    object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = 0.5
                    object.AngleCut[GasIndex][iEnergy][iProcess - 1] = 1.0
                    object.AngularModel[GasIndex][iProcess - 1] = 0
                    if object.MixObject.Gases[GasIndex].AngularModel[2] == 1:
                        ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEElasticCrossSection[2][iEnergy]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.CalcAngCut()
                        object.AngleCut[GasIndex][iEnergy][iProcess - 1] = AngObject.AngCut
                        object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = AngObject.ScatteringParameter2
                        object.AngularModel[GasIndex][iProcess - 1] = 1
                    elif object.MixObject.Gases[GasIndex].AngularModel[2] == 2:
                        object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = \
                        object.MixObject.Gases[GasIndex].PEElasticCrossSection[2][iEnergy]
                        object.AngularModel[GasIndex][iProcess - 1] = 2
                elif object.MixObject.Gases[GasIndex].N_Ionization > 1:
                    for iIonizaton in range(object.MixObject.Gases[GasIndex].N_Ionization):
                        iProcess += 1
                        object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1] = \
                        object.MixObject.Gases[GasIndex].IonizationCrossSection[iIonizaton][iEnergy] * \
                        object.VMoleculesPerCm3PerGas[GasIndex]
                        object.IonCollisionFreq[iEnergy] += object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1]
                        object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = 0.5
                        object.AngleCut[GasIndex][iEnergy][iProcess - 1] = 1.0
                        object.AngularModel[GasIndex][iProcess - 1] = 0
                        if object.MixObject.Gases[0].AngularModel[2] == 1:
                            ScatteringParameter1 = \
                            object.MixObject.Gases[GasIndex].PEIonizationCrossSection[iIonizaton][iEnergy]
                            AngObject.ScatteringParameter1 = ScatteringParameter1
                            AngObject.CalcAngCut()
                            object.AngleCut[GasIndex][iEnergy][iProcess - 1] = AngObject.AngCut
                            object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = AngObject.ScatteringParameter2
                            object.AngularModel[GasIndex][iProcess - 1] = 1
                        elif object.MixObject.Gases[0].AngularModel[2] == 2:
                            object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = \
                            object.MixObject.Gases[GasIndex].PEIonizationCrossSection[iIonizaton][iEnergy]
                            object.AngularModel[GasIndex][iProcess - 1] = 2
                if iEnergy == 0:
                    if object.MixObject.Gases[GasIndex].N_Ionization <= 1:
                        RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGas[GasIndex][iProcess - 1] = RGas
                        object.EnergyLevels[GasIndex][iProcess - 1] = object.MixObject.Gases[GasIndex].E[2] / RGas
                        object.NC0[GasIndex][iProcess - 1] = object.MixObject.Gases[GasIndex].NC0[0]
                        object.EC0[GasIndex][iProcess - 1] = object.MixObject.Gases[GasIndex].EC0[0]
                        object.WPL[GasIndex][iProcess - 1] = object.MixObject.Gases[GasIndex].EOBY[0]
                        object.ElectronNumChange[GasIndex][iProcess - 1] = 1
                        L = 2
                        object.InteractionType[GasIndex][iProcess - 1] = L
                        object.PenningFraction[GasIndex][0][iProcess - 1] = 0.0
                        object.PenningFraction[GasIndex][1][iProcess - 1] = 0.0
                        object.PenningFraction[GasIndex][2][iProcess - 1] = 0.0
                    elif object.MixObject.Gases[GasIndex].N_Ionization > 1:
                        iProcess = iProcess - object.MixObject.Gases[GasIndex].N_Ionization
                        for iIonizaton in range(object.MixObject.Gases[GasIndex].N_Ionization):
                            iProcess = iProcess + 1
                            RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                            object.RGas[GasIndex][iProcess - 1] = RGas
                            object.EnergyLevels[GasIndex][iProcess - 1] = \
                            object.MixObject.Gases[GasIndex].IonizationEnergy[iIonizaton] / RGas
                            object.NC0[GasIndex][iProcess - 1] = object.MixObject.Gases[GasIndex].NC0[iIonizaton]
                            object.EC0[GasIndex][iProcess - 1] = object.MixObject.Gases[GasIndex].EC0[iIonizaton]
                            object.WPL[GasIndex][iProcess - 1] = object.MixObject.Gases[GasIndex].EOBY[iIonizaton]
                            object.ElectronNumChange[GasIndex][iProcess - 1] = 1
                            L = 2
                            object.InteractionType[GasIndex][iProcess - 1] = L
                            object.PenningFraction[GasIndex][0][iProcess - 1] = 0.0
                            object.PenningFraction[GasIndex][1][iProcess - 1] = 0.0
                            object.PenningFraction[GasIndex][2][iProcess - 1] = 0.0
            # ATTACHMENT
            if object.Max_Electron_Energy >= object.MixObject.Gases[GasIndex].E[3]:
                if object.MixObject.Gases[GasIndex].N_Attachment <= 1:
                    iProcess += 1
                    object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1] = object.MixObject.Gases[GasIndex].Q[3][
                                                                                     iEnergy] * \
                                                                                 object.VMoleculesPerCm3PerGas[GasIndex]
                    object.AttCollisionFreq[iEnergy] += object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1]
                    object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = 0.5
                    object.AngleCut[GasIndex][iEnergy][iProcess - 1] = 1.0
                    if iEnergy == 0:
                        RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGas[GasIndex][iProcess - 1] = RGas
                        object.EnergyLevels[GasIndex][iProcess - 1] = 0.0
                        object.AngularModel[GasIndex][iProcess - 1] = 0
                        object.ElectronNumChange[GasIndex][iProcess - 1] = -1
                        L = 3
                        object.InteractionType[GasIndex][iProcess - 1] = L
                        object.PenningFraction[GasIndex][0][iProcess - 1] = 0.0
                        object.PenningFraction[GasIndex][1][iProcess - 1] = 0.0
                        object.PenningFraction[GasIndex][2][iProcess - 1] = 0.0

                elif object.MixObject.Gases[GasIndex].N_Attachment > 1:
                    for JJ in range(int(object.MixObject.Gases[GasIndex].N_Attachment)):
                        iProcess += 1
                        object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1] = \
                        object.MixObject.Gases[GasIndex].AttachmentCrossSection[JJ][iEnergy] * \
                        object.VMoleculesPerCm3PerGas[GasIndex]
                        object.AttCollisionFreq[iEnergy] += object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1]
                        object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = 0.5
                        object.AngleCut[GasIndex][iEnergy][iProcess - 1] = 1.0
                        if iEnergy == 0:
                            RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                            object.RGas[GasIndex][iProcess - 1] = RGas
                            object.EnergyLevels[GasIndex][iProcess - 1] = 0.0
                            object.AngularModel[GasIndex][iProcess - 1] = 0
                            object.ElectronNumChange[GasIndex][iProcess - 1] = -1
                            L = 3
                            object.InteractionType[GasIndex][iProcess - 1] = L
                            object.PenningFraction[GasIndex][0][iProcess - 1] = 0.0
                            object.PenningFraction[GasIndex][1][iProcess - 1] = 0.0
                            object.PenningFraction[GasIndex][2][iProcess - 1] = 0.0
            # INELASTIC AND SUPERELASTIC
            if object.MixObject.Gases[GasIndex].N_Inelastic > 0:
                for J in range(int(object.MixObject.Gases[GasIndex].N_Inelastic)):
                    iProcess = iProcess + 1
                    object.CollisionFrequency[GasIndex][iEnergy][iProcess - 1] = \
                    object.MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][iEnergy] * \
                    object.VMoleculesPerCm3PerGas[GasIndex]
                    object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = 0.5
                    object.AngleCut[GasIndex][iEnergy][iProcess - 1] = 1.0
                    object.AngularModel[GasIndex][iProcess - 1] = 0
                    if object.MixObject.Gases[GasIndex].KIN[J] == 1:
                        ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][
                            iEnergy]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.CalcAngCut()
                        object.AngleCut[GasIndex][iEnergy][iProcess - 1] = AngObject.AngCut
                        object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = AngObject.ScatteringParameter2
                        object.AngularModel[GasIndex][iProcess - 1] = 1
                    elif object.MixObject.Gases[GasIndex].KIN[J] == 2:
                        object.ScatteringParameter[GasIndex][iEnergy][iProcess - 1] = \
                        object.MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][iEnergy]
                        object.AngularModel[GasIndex][iProcess - 1] = 2
                    if iEnergy == 0:
                        RGas = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGas[GasIndex][iProcess - 1] = RGas
                        object.EnergyLevels[GasIndex][iProcess - 1] = object.MixObject.Gases[GasIndex].EnergyLevels[
                                                                          J] / RGas
                        L = 4
                        if object.MixObject.Gases[GasIndex].EnergyLevels[J] < 0:
                            L = 5
                        object.ElectronNumChange[GasIndex][iProcess - 1] = 0
                        object.InteractionType[GasIndex][iProcess - 1] = L
                        object.PenningFraction[GasIndex][0][iProcess - 1] = \
                        object.MixObject.Gases[GasIndex].PenningFraction[0][J]
                        object.PenningFraction[GasIndex][1][iProcess - 1] = \
                        object.MixObject.Gases[GasIndex].PenningFraction[1][J] * 1.0e-16 / sqrt(3)
                        object.PenningFraction[GasIndex][2][iProcess - 1] = \
                        object.MixObject.Gases[GasIndex].PenningFraction[2][J]

            object.NumMomCrossSectionPoints[GasIndex] = iProcess
            object.ISIZE[GasIndex] = 1
            for I in range(1, 9):
                if object.NumMomCrossSectionPoints[GasIndex] >= 2 ** I:
                    object.ISIZE[GasIndex] = 2 ** I
                else:
                    break
            # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
            object.TotalCollisionFrequency[GasIndex][iEnergy] = 0.0
            for p in range(int(object.NumMomCrossSectionPoints[GasIndex])):
                object.TotalCollisionFrequency[GasIndex][iEnergy] = object.TotalCollisionFrequency[GasIndex][iEnergy] + \
                                                                    object.CollisionFrequency[GasIndex][iEnergy][p]
                if object.CollisionFrequency[GasIndex][iEnergy][p] < 0:
                    print ("WARNING NEGATIVE COLLISION FREQUENCY at gas " + str(p) + "  " + str(iEnergy))

            for p in range(int(object.NumMomCrossSectionPoints[GasIndex])):
                if object.TotalCollisionFrequency[GasIndex][iEnergy] == 0:
                    object.CollisionFrequency[GasIndex][iEnergy][p] = 0
                else:
                    object.CollisionFrequency[GasIndex][iEnergy][p] = object.CollisionFrequency[GasIndex][iEnergy][p] / \
                                                                      object.TotalCollisionFrequency[GasIndex][iEnergy]

            for p in range(1, int(object.NumMomCrossSectionPoints[GasIndex])):
                object.CollisionFrequency[GasIndex][iEnergy][p] = object.CollisionFrequency[GasIndex][iEnergy][p] + \
                                                                  object.CollisionFrequency[GasIndex][iEnergy][p - 1]
            object.TotalCollisionFrequency[GasIndex][iEnergy] = object.TotalCollisionFrequency[GasIndex][iEnergy] * \
                                                                object.SqrtEnergy[iEnergy]
            object.IonCollisionFreq[iEnergy] *= object.SqrtEnergy[iEnergy]
            object.AttCollisionFreq[iEnergy] *= object.SqrtEnergy[iEnergy]

    # CALCULATION OF NULL COLLISION FREQUENCIES
    for iEnergy in range(4000):
        sum = 0
        for i in range(object.NumberOfGases):
            object.NumMomCrossSectionPointsNull[i] = object.MixObject.Gases[i].N_Null
            sum += int(object.NumMomCrossSectionPointsNull[i])

            if sum == 0:
                break
        for i in range(object.NumberOfGases):
            if object.NumMomCrossSectionPointsNull[i] > 0:
                for J in range(int(object.NumMomCrossSectionPointsNull[i])):
                    object.ScaleNull[i][J] = object.MixObject.Gases[i].ScaleNull[J]
                    object.NullCollisionFreq[i][iEnergy][J] = object.MixObject.Gases[i].NullCrossSection[J][iEnergy] * \
                                                              object.VMoleculesPerCm3PerGas[i] * object.ScaleNull[i][J]

            # CALCULATE NULL COLLISION FREQUENCY FOR EACH GAS COMPONENT

        for GasIndex in range(object.NumberOfGases):
            object.TotalCollisionFrequencyNull[GasIndex][iEnergy] = 0.0
            for IL in range(int(object.NumMomCrossSectionPointsNull[GasIndex])):
                object.TotalCollisionFrequencyNull[GasIndex][iEnergy] = object.TotalCollisionFrequencyNull[GasIndex][
                                                                            iEnergy] + \
                                                                        object.NullCollisionFreq[GasIndex][iEnergy][IL]
                if object.NullCollisionFreq[GasIndex][iEnergy][IL] < 0:
                    print("WARNING NEGATIVE NULL COLLISION FREQUENCY")
            for IL in range(int(object.NumMomCrossSectionPointsNull[GasIndex])):
                if object.TotalCollisionFrequencyNull[GasIndex][iEnergy] == 0:
                    object.NullCollisionFreq[GasIndex][iEnergy][IL] = 0.0
                else:
                    object.NullCollisionFreq[GasIndex][iEnergy][IL] = object.NullCollisionFreq[GasIndex][iEnergy][IL] / \
                                                                      object.TotalCollisionFrequencyNull[GasIndex][
                                                                          iEnergy]

            for IL in range(1, int(object.NumMomCrossSectionPointsNull[GasIndex])):
                object.NullCollisionFreq[GasIndex][iEnergy][IL] = object.NullCollisionFreq[GasIndex][iEnergy][IL] + \
                                                                  object.NullCollisionFreq[GasIndex][iEnergy][IL - 1]
            object.TotalCollisionFrequencyNull[GasIndex][iEnergy] = object.TotalCollisionFrequencyNull[GasIndex][
                                                                        iEnergy] * object.SqrtEnergy[iEnergy]

    AngularModelSum = 0

    for GasIndex in range(object.NumberOfGases):
        for J in range(6):
            AngularModelSum += object.MixObject.Gases[GasIndex].AngularModel[J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            AngularModelSum += object.MixObject.Gases[GasIndex].KIN[J]

    if AngularModelSum > 0:
        object.AnisotropicDetected = 1

    # CALCULATE NULL COLLISION FREQUENCIES FOR EACH GAS COMPONENT
    tt = 0
    FAKEnergyLevels = abs(object.FakeIonizations) / object.NumberOfGases
    for GasIndex in range(object.NumberOfGases):
        object.MaxCollisionFreq[GasIndex] = 0.0
        for iEnergy in range(4000):
            if object.TotalCollisionFrequency[GasIndex][iEnergy] + object.TotalCollisionFrequencyNull[GasIndex][
                iEnergy] + FAKEnergyLevels >= object.MaxCollisionFreq[GasIndex]:
                object.MaxCollisionFreq[GasIndex] = object.TotalCollisionFrequency[GasIndex][iEnergy] + \
                                                    object.TotalCollisionFrequencyNull[GasIndex][
                                                        iEnergy] + FAKEnergyLevels

    # CALCULATE EACH GAS CUMLATIVE FRACTION NULL COLLISION FREQUENCIES
    object.MaxCollisionFreqTotal = 0.0
    for GasIndex in range(object.NumberOfGases):
        object.MaxCollisionFreqTotal = object.MaxCollisionFreqTotal + object.MaxCollisionFreq[GasIndex]
    for GasIndex in range(object.NumberOfGases):
        object.MaxCollisionFreqTotalG[GasIndex] = object.MaxCollisionFreq[GasIndex] / object.MaxCollisionFreqTotal
    for GasIndex in range(1, object.NumberOfGases):
        object.MaxCollisionFreqTotalG[GasIndex] = object.MaxCollisionFreqTotalG[GasIndex] + \
                                                  object.MaxCollisionFreqTotalG[GasIndex - 1]

    # CALCULATE MAXWELL BOLTZMAN VELOCITY FACTOR FOR EACH GAS COMPONENT

    for GasIndex in range(object.NumberOfGases):
        object.VTMB[GasIndex] = sqrt(2.0 * ElectronCharge * object.ThermalEnergy / object.AMGAS[GasIndex]) * 1e-12

    for I in range(object.EnergySteps):
        object.TotalCrossSection[I] = object.MoleculesPerCm3PerGas[0] * object.MixObject.Gases[0].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[1] * object.MixObject.Gases[1].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[2] * object.MixObject.Gases[2].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[3] * object.MixObject.Gases[3].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[4] * object.MixObject.Gases[4].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[5] * object.MixObject.Gases[5].Q[0][I]
        object.ElasticCrossSection[I] = object.MoleculesPerCm3PerGas[0] * object.MixObject.Gases[0].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[1] * object.MixObject.Gases[1].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[2] * object.MixObject.Gases[2].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[3] * object.MixObject.Gases[3].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[4] * object.MixObject.Gases[4].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[5] * object.MixObject.Gases[5].Q[1][I]

        for GasIndex in range(6):
            object.IonizationCrossSection[GasIndex][I] = object.MixObject.Gases[GasIndex].Q[2][I] * \
                                                         object.MoleculesPerCm3PerGas[GasIndex]
            AttachmentCrossSection[GasIndex][I] = object.MixObject.Gases[GasIndex].Q[3][I] * \
                                                  object.MoleculesPerCm3PerGas[GasIndex]
        object.RelativeIonMinusAttachCrossSection[I] = 0.0
        object.AttachmentSectionSum[I] = 0.0
        object.CrossSectionSum[I] = 0.0
        for J in range(object.NumberOfGases):
            object.CrossSectionSum[I] = object.CrossSectionSum[I] + object.IonizationCrossSection[J][I] + \
                                        AttachmentCrossSection[J][I]
            object.AttachmentSectionSum[I] = object.AttachmentSectionSum[I] + AttachmentCrossSection[J][I]
            object.RelativeIonMinusAttachCrossSection[I] = object.RelativeIonMinusAttachCrossSection[I] + \
                                                           object.IonizationCrossSection[J][I] - \
                                                           AttachmentCrossSection[J][I]
        for GasIndex in range(6):
            for J in range(int(object.MixObject.Gases[GasIndex].N_Inelastic)):
                object.CrossSectionSum[I] = object.CrossSectionSum[I] + \
                                            object.MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][I] * \
                                            object.MoleculesPerCm3PerGas[GasIndex]
    return
