from __future__ import print_function

from PyBoltz.Boltz cimport Boltz
import cython
from PyBoltz.Boltz cimport drand48
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow,log10
from libc.stdlib cimport malloc, free
from libc.string cimport memset

import numpy as np
cimport numpy as np

from PyBoltz.MBSorts cimport MBSort, MBSortT

# The implementation of the EnergyCuts functions mirrors the MONTEC functions.
# The later are more extensively commented and we guide developers there for
# more information.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int isNaN(double num):
    return num != num


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GenerateMaxBoltz(double RandomSeed, double *RandomMaxBoltzArray):
    cdef double Ran1, Ran2, TwoPi
    cdef int J
    for J in range(0, 5, 2):
        Ran1 = random_uniform(RandomSeed)
        Ran2 = random_uniform(RandomSeed)
        TwoPi = 2.0 * np.pi
        RandomMaxBoltzArray[J] = sqrt(-1 * log(Ran1)) * cos(Ran2 * TwoPi)
        RandomMaxBoltzArray[J + 1] = sqrt(-1 * log(Ran1)) * sin(Ran2 * TwoPi)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimit(Boltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the Boltz object to try a higher FinalEnergy value.
    This function is used when there is no magnetic field. 

    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    
    Electric field in z direction.

    The object parameter is the Boltz object to be setup and used in the simulation.
    """

    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE, INTEM
    cdef double SMALL, RandomSeed, E1, TDASH, CONST5, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, EOK, VelBeforeM1, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, CosTheta, RAN, EPSI, R4, Phi, SinPhi, CosPhi, ARG1
    cdef double D, Q, U, CosSquareTheta, CosZAngle, SinZAngle, ARGZ, CONST12, VXLAB, VYLAB, VZLAB, TEMP[4000],DELTAE
    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))

    CONST5 = Object.CONST3 / 2.0
    ISAMP = 10
    SMALL = 1.0e-20
    I = 0
    RandomSeed = Object.Random_Seed
    E1 = Object.InitialElectronEnergy
    N4000 = 4000
    TDASH = 0.0
    INTEM = 8
    for J in range(N4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]

    # INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    BP = (Object.EField ** 2) * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    F4 = 2 * acos(-1)
    DELTAE = Object.Max_Electron_Energy / float(INTEM)
    E1 = Object.InitialElectronEnergy
    J2M = Object.MaxNumberOfCollisions / ISAMP
    for J1 in range(int(J2M)):
        if J1 != 0  and not int(str(J1)[-int(log10(J1)):]) and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        elif J1>=1e6 and J1%1e5 ==0 and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        while True:
            R1 = random_uniform(RandomSeed)
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Object.MaxCollisionFreqNT[I]
            T = -1 * log(R1) / TLIM + TDASH
            TDASH = T
            AP = DirCosineZ1 * F2 * sqrt(E1)
            E = E1 + (AP + BP * T) * T
            IE = int(E / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += log(R1) / TLIM
                Object.MaxCollisionFreqNT[I] *= 1.05
                continue

            # Test FOR NULL COLLISIONS
            R5 = random_uniform(RandomSeed)
            Test1 = Object.TotalCollisionFrequencyNT[IE] / TLIM
            if R5<=Test1:
                break

        if IE == 3999:
            # Electron energy is out of limits
            return 1

        # CALCULATE DIRECTION COSINES AT INSTANT BEFORE COLLISION
        TDASH = 0.0
        CONST6 = sqrt(E1 / E)
        DirCosineX2 = DirCosineX1 * CONST6
        DirCosineY2 = DirCosineY1 * CONST6
        DirCosineZ2 = DirCosineZ1 * CONST6 + Object.EField * T * CONST5 / sqrt(E)
        R2 = random_uniform(RandomSeed)


        # Determination of real collision type
        I = MBSort(I, R2, IE, Object)
        # Find the location within 4 units in collision array
        while Object.CollisionFrequencyNT[IE][I] < R2:
            I = I + 1

        S1 = Object.RGasNT[I]
        EI = Object.EnergyLevelsNT[I]
        if Object.ElectronNumChangeNT[I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI


        # Generate scattering angles and update laboratory cosines after collision also update energy of electron
        IPT = Object.InteractionTypeNT[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random_uniform(RandomSeed)

        if Object.AngularModelNT[I] == 1:
            R31 = random_uniform(RandomSeed)
            CosTheta = 1.0 - R3 * Object.AngleCutNT[IE][I]
            if R31 > Object.ScatteringParameterNT[IE][I]:
                CosTheta = -1 * CosTheta
        elif Object.AngularModelNT[I] == 2:
            EPSI = Object.ScatteringParameterNT[IE][I]
            CosTheta = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            CosTheta = 1 - 2 * R3
        Theta = acos(CosTheta)
        R4 = random_uniform(RandomSeed)
        Phi = F4 * R4
        SinPhi = sin(Phi)
        CosPhi = cos(Phi)
        ARG1 = 1 - S1 * EI / E
        ARG1 = max(ARG1, SMALL)

        D = 1 - CosTheta * sqrt(ARG1)
        E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)

        E1 = max(E1, SMALL)
        Q = sqrt((E / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.AngleFromZ = asin(Q * sin(Theta))

        CosZAngle = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CosSquareTheta = CosTheta ** 2

        if CosTheta < 0 and CosSquareTheta > U:
            CosZAngle = -1 * CosZAngle

        SinZAngle = sin(Object.AngleFromZ)
        DirCosineZ2 = min(DirCosineZ2, 1)
        ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
        if ARGZ == 0:
            DirCosineZ1 = CosZAngle
            DirCosineX1 = CosPhi * SinZAngle
            DirCosineY1 = SinPhi * SinZAngle
        else:
            DirCosineZ1 = DirCosineZ2 * CosZAngle + ARGZ * SinZAngle * SinPhi
            DirCosineY1 = DirCosineY2 * CosZAngle + (SinZAngle / ARGZ) * (DirCosineX2 * CosPhi - DirCosineY2 * DirCosineZ2 * SinPhi)
            DirCosineX1 = DirCosineX2 * CosZAngle - (SinZAngle / ARGZ) * (DirCosineY2 * CosPhi + DirCosineX2 * DirCosineZ2 * SinPhi)

    return 0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitB(Boltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the Boltz object to try a higher FinalEnergy value.
    This function is used when the magnetic field angle is 90 degrees to the electric field. 

    The object parameter is the Boltz object to be setup and used in the simulation.
    
    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE, INTEM
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, EOK, VelBeforeM1, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, CosTheta, RAN, EPSI, R4, Phi, SinPhi, CosPhi, ARG1
    cdef double D, Q, U, CosSquareTheta, CosZAngle, SinZAngle, ARGZ, CONST12, VXLAB, VYLAB, VZLAB, TEMP[4000],DELTAE,EFieldTimes100
    TEMP = <double *> malloc(4000 * sizeof(double))

    memset(TEMP, 0, 4000 * sizeof(double))

    Object.SmallNumber =  1.0e-20
    ISAMP = 20
    EFieldTimes100 = Object.EField * 100
    RandomSeed = Object.Random_Seed
    E1 = Object.InitialElectronEnergy

    INTEM = 8
    TDASH = 0.0
    CONST9 = Object.CONST3 * 0.01

    # INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    for J in range(4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]

    VelTotal = CONST9 * sqrt(E1)
    VelXBefore = DirCosineX1 * VelTotal
    VelYBefore = DirCosineY1 * VelTotal
    VelZBefore = DirCosineZ1 * VelTotal

    F4 = 2 * acos(-1)

    DELTAE = Object.Max_Electron_Energy / float(INTEM)

    J2M = Object.MaxNumberOfCollisions / ISAMP

    for J1 in range(int(J2M)):

        if J1 != 0  and not int(str(J1)[-int(log10(J1)):]) and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        elif J1>=1e6 and J1%1e5 ==0 and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')

        while True:
            R1 = random_uniform(RandomSeed)
            I = int(E1 / DELTAE)+1
            I = min(I, INTEM) - 1
            TLIM = Object.MaxCollisionFreqNT[I]
            T = -1 * log(R1) / TLIM + TDASH
            TDASH = T
            WBT = Object.AngularSpeedOfRotation * T
            CosWT = cos(WBT)
            SinWT = sin(WBT)
            DZ = (VelZBefore * SinWT + (Object.EFieldOverBField - VelYBefore) * (1 - CosWT)) / Object.AngularSpeedOfRotation
            E = E1 + DZ * EFieldTimes100
            IE = int(E / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += log(R1) / TLIM
                Object.MaxCollisionFreqNT[I] *= 1.05
                continue
            R5 = random_uniform(RandomSeed)
            Test1 = Object.TotalCollisionFrequencyNT[IE] / TLIM

            # Test FOR REAL OR NULL COLLISION
            if R5<=Test1:
                break

        if IE == 3999:
            # Electron energy out of range
            return 1

        TDASH = 0.0
        VelXAfter = VelXBefore
        VelYAfter = (VelYBefore - Object.EFieldOverBField) * CosWT + VelZBefore * SinWT + Object.EFieldOverBField
        VelZAfter = VelZBefore * CosWT - (VelYBefore - Object.EFieldOverBField) * SinWT
        VelTotal = sqrt(VelXAfter ** 2 + VelYAfter ** 2 + VelZAfter ** 2)
        DirCosineX2 = VelXAfter / VelTotal
        DirCosineY2 = VelYAfter / VelTotal
        DirCosineZ2 = VelZAfter / VelTotal

        # DETERMINATION OF REAL COLLISION TYPE
        R2 = random_uniform(RandomSeed)

        # FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        I = MBSort(I, R2, IE, Object)
        while Object.CollisionFrequencyNT[IE][I] < R2:
            I = I + 1

        S1 = Object.RGasNT[I]
        EI = Object.EnergyLevelsNT[I]
        if Object.ElectronNumChangeNT[I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI

        # GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        # COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.InteractionTypeNT[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random_uniform(RandomSeed)

        if Object.AngularModelNT[I] == 1:
            R31 = random_uniform(RandomSeed)
            CosTheta = 1.0 - R3 * Object.AngleCutNT[IE][I]
            if R31 > Object.ScatteringParameterNT[IE][I]:
                CosTheta = -1 * CosTheta
        elif Object.AngularModelNT[I] == 2:
            EPSI = Object.ScatteringParameterNT[IE][I]
            CosTheta = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            CosTheta = 1 - 2 * R3
        Theta = acos(CosTheta)
        R4 = random_uniform(RandomSeed)
        Phi = F4 * R4
        SinPhi = sin(Phi)
        CosPhi = cos(Phi)
        ARG1 = 1 - S1 * EI / E
        ARG1 = max(ARG1, SMALL)

        D = 1 - CosTheta * sqrt(ARG1)
        E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = sqrt((E / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.AngleFromZ = asin(Q * sin(Theta))

        CosZAngle = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CosSquareTheta = CosTheta ** 2

        if CosTheta < 0 and CosSquareTheta > U:
            CosZAngle = -1 * CosZAngle
        SinZAngle = sin(Object.AngleFromZ)
        DirCosineZ2 = min(DirCosineZ2, 1)
        VelTotal = CONST9 * sqrt(E1)
        ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
        if ARGZ == 0:
            DirCosineZ1 = CosZAngle
            DirCosineX1 = CosPhi * SinZAngle
            DirCosineY1 = SinPhi * SinZAngle
        else:
            DirCosineZ1 = DirCosineZ2 * CosZAngle + ARGZ * SinZAngle * SinPhi
            DirCosineY1 = DirCosineY2 * CosZAngle + (SinZAngle / ARGZ) * (DirCosineX2 * CosPhi - DirCosineY2 * DirCosineZ2 * SinPhi)
            DirCosineX1 = DirCosineX2 * CosZAngle - (SinZAngle / ARGZ) * (DirCosineY2 * CosPhi + DirCosineX2 * DirCosineZ2 * SinPhi)
        VelXBefore = DirCosineX1 * VelTotal
        VelYBefore = DirCosineY1 * VelTotal
        VelZBefore = DirCosineZ1 * VelTotal

    return 0




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitBT(Boltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the Boltz object to try a higher FinalEnergy value.
    This function is used when the magnetic field angle is 90 degrees to the electric field. 

    The object parameter is the Boltz object to be setup and used in the simulation.
    
    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, EOK, VelBeforeM1, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, CosTheta, RAN, EPSI, R4, Phi, SinPhi, CosPhi, ARG1
    cdef double D, Q, U, CosSquareTheta, CosZAngle, SinZAngle, ARGZ, CONST12, VXLAB, VYLAB, EFieldTimes100,TLIM,VelXAfter,VelYAfter,VelZAfter

 

    ISAMP = 20
    SMALL = 1.0e-20
    EFieldTimes100 = Object.EField * 100
    RandomSeed = Object.Random_Seed
    E1 = Object.InitialElectronEnergy
    N4000 = 4000
    TDASH = 0.0

    # GENRATE RANDOM NUMBER FOR MAXWELL BOLTZMAN
    GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
    IMBPT = 0


    CONST9 = Object.CONST3 * 0.01
    CONST10 = CONST9 * CONST9

    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    VelTotal = CONST9 * sqrt(E1)
    VelXBefore = DirCosineX1 * VelTotal
    VelYBefore = DirCosineY1 * VelTotal
    VelZBefore = DirCosineZ1 * VelTotal

    F4 = 2 * acos(-1)
    J2M = Object.MaxNumberOfCollisions / ISAMP

    for J1 in range(int(J2M)):
        if J1 != 0  and not int(str(J1)[-int(log10(J1)):]) and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        elif J1>=1e6 and J1%1e5 ==0 and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')

        while True:
            R1 = random_uniform(RandomSeed)
            T = -1 * log(R1) / Object.MaxCollisionFreqTotal + TDASH
            TDASH = T
            WBT = Object.AngularSpeedOfRotation * T
            CosWT = cos(WBT)
            SinWT = sin(WBT)

            DZ = (VelZBefore * SinWT + (Object.EFieldOverBField - VelYBefore) * (1 - CosWT)) / Object.AngularSpeedOfRotation
            E = E1 + DZ * EFieldTimes100
            #CALC ELECTRON VELOCITY IN LAB FRAME
            VelXAfter = VelXBefore
            VelYAfter = (VelYBefore - Object.EFieldOverBField) * CosWT + VelZBefore * SinWT + Object.EFieldOverBField
            VelZAfter = VelZBefore * CosWT - (VelYBefore - Object.EFieldOverBField) * SinWT
            #FIND IDENTITY OF GAS FOR COLLISION
            GasIndex = 0
            R2 = random_uniform(RandomSeed)
            while Object.MaxCollisionFreqTotalG[GasIndex] < R2:
                GasIndex += 1
            #CALCULATE GAS VELOCITY VECTORS GasVelX,GasVelY,GasVelZ
            IMBPT += 1
            if IMBPT > 6:
                GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
                IMBPT = 1
            GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]

            #CALCULATE ENERGY WITH STATIONARY GAS TARGET , EOK
            EOK = (pow((VelXAfter - GasVelX), 2) + pow((VelYAfter - GasVelY), 2) + pow((VelZAfter - GasVelZ), 2)) / CONST10
            IE = int(EOK / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            #Test FOR REAL OR NULL COLLISION
            R5 = random_uniform(RandomSeed)
            TLIM = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
            if R5 <= TLIM:
                break
        if IE == 3999:
            #ELECTRON ENERGY OUT OF RANGE
            return 1

        TDASH = 0.0
        #CALCULATE DIRECTION COSINES OF ELECTRON IN 0 KELVIN FRAME
        VelBeforeM1 = 1.0 / (CONST9 * sqrt(EOK))
        DXCOM = (VelXAfter - GasVelX) * VelBeforeM1
        DYCOM = (VelYAfter - GasVelY) * VelBeforeM1
        DZCOM = (VelZAfter - GasVelZ) * VelBeforeM1


        #FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        R2 = random_uniform(RandomSeed)
        I = MBSortT(GasIndex, I, R2, IE, Object)
        while Object.CollisionFrequency[GasIndex][IE][I] < R2:
            I = I + 1

        S1 = Object.RGas[GasIndex][I]
        EI = Object.EnergyLevels[GasIndex][I]
        if Object.ElectronNumChange[GasIndex][I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI

        # GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        # COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.InteractionType[GasIndex][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = (S1 * S1) / (S1 - 1)
        R3 = random_uniform(RandomSeed)

        if Object.AngularModel[GasIndex][I] == 1:
            R31 = random_uniform(RandomSeed)
            CosTheta = 1.0 - R3 * Object.AngleCut[GasIndex][IE][I]
            if R31 > Object.ScatteringParameter[GasIndex][IE][I]:
                CosTheta = -1 * CosTheta
        elif Object.AngularModel[GasIndex][I] == 2:
            EPSI = Object.ScatteringParameter[GasIndex][IE][I]
            CosTheta = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            CosTheta = 1 - 2 * R3

        Theta = acos(CosTheta)
        R4 = random_uniform(RandomSeed)
        Phi = F4 * R4
        SinPhi = sin(Phi)
        CosPhi = cos(Phi)
        ARG1 = 1 - S1 * EI / EOK
        ARG1 = max(ARG1, SMALL)

        D = 1 - CosTheta * sqrt(ARG1)
        E1 = EOK * (1 - EI / (S1 * EOK) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = sqrt((EOK / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.AngleFromZ = asin(Q * sin(Theta))

        CosZAngle = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CosSquareTheta = CosTheta ** 2

        if CosTheta < 0 and CosSquareTheta > U:
            CosZAngle = -1 * CosZAngle
        SinZAngle = sin(Object.AngleFromZ)
        DirCosineZ2 = min(DZCOM, 1)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DirCosineZ1 = CosZAngle
            DirCosineX1 = CosPhi * SinZAngle
            DirCosineY1 = SinPhi * SinZAngle
        else:
            DirCosineZ1 = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi
            DirCosineY1 = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi)
            DirCosineX1 = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi)
        # TRANSFORM VELOCITY VECTORS TO LAB FRAME
        VelTotal = CONST9 * sqrt(E1)
        VelXBefore = DirCosineX1 * VelTotal + GasVelX
        VelYBefore = DirCosineY1 * VelTotal + GasVelY
        VelZBefore = DirCosineZ1 * VelTotal + GasVelZ
        #  CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
        E1 = (VelXBefore * VelXBefore + VelYBefore * VelYBefore + VelZBefore * VelZBefore) / CONST10
        VelBeforeM1 = 1.0 / (CONST9 * sqrt(E1))
        DirCosineX1 = VelXBefore * VelBeforeM1
        DirCosineY1 = VelYBefore * VelBeforeM1
        DirCosineZ1 = VelZBefore * VelBeforeM1

    return 0





@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitC(Boltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the Boltz object to try a higher FinalEnergy value.
    
    The object parameter is the Boltz object to be setup and used in the simulation.
    
    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE, INTEM
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, EOK, VelBeforeM1, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, CosTheta, RAN, EPSI, R4, Phi, SinPhi, CosPhi, ARG1
    cdef double D, Q, U, CosSquareTheta, CosZAngle, SinZAngle, ARGZ, CONST12, VXLAB, VYLAB, VZLAB, TEMP[4000],DELTAE,EFX100,EFZ100,RTHETA,
    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))
    ISAMP = 20
    SMALL = 1.0e-20
    RTHETA = Object.BField_Angle * acos(-1) / 180
    EFZ100 = Object.EField * 100 * sin(RTHETA)
    EFX100 = Object.EField * 100 * cos(RTHETA)
    F1 = Object.EField * Object.CONST2 * cos(RTHETA)
    EOVBR = Object.EFieldOverBField * sin(RTHETA)
    E1 = Object.InitialElectronEnergy
    INTEM = 8
    TDASH = 0.0
    CONST9 = Object.CONST3 * 0.01
    for J in range(4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]

    # INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)
    VelTotal = CONST9 * sqrt(E1)
    VelXBefore = DirCosineX1 * VelTotal
    VelYBefore = DirCosineY1 * VelTotal
    VelZBefore = DirCosineZ1 * VelTotal

    F4 = 2 * acos(-1)
    DELTAE = Object.Max_Electron_Energy / float(INTEM)
    J2M = Object.MaxNumberOfCollisions / ISAMP
    RandomSeed = Object.Random_Seed

    for J1 in range(int(J2M)):
        if J1 != 0  and not int(str(J1)[-int(log10(J1)):]) and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        elif J1>=1e6 and J1%1e5 ==0 and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        while True:
            R1 = random_uniform(RandomSeed)
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Object.MaxCollisionFreqNT[I]
            T = -1 * log(R1) / TLIM + TDASH
            TDASH = T
            WBT = Object.AngularSpeedOfRotation * T
            CosWT = cos(WBT)
            SinWT = sin(WBT)
            DZ = (VelZBefore * SinWT + (EOVBR - VelYBefore) * (1 - CosWT)) / Object.AngularSpeedOfRotation
            DX = VelXBefore * T + F1 * T * T
            E = E1 + DZ * EFZ100 + DX * EFX100
            IE = int(E / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += log(R1) / TLIM
                Object.MaxCollisionFreqNT[I] *= 1.05
                continue
            R5 = random_uniform(RandomSeed)
            Test1 = Object.TotalCollisionFrequencyNT[IE] / TLIM

            # Test FOR REAL OR NULL COLLISION
            if R5<= Test1:
                break

        if IE == 3999:
            # Electron energy out of range
            return 1

        TDASH = 0.0
        VelXAfter = VelXBefore + 2 * F1 * T
        VelYAfter = (VelYBefore - EOVBR) * CosWT + VelZBefore * SinWT + EOVBR
        VelZAfter = VelZBefore * CosWT - (VelYBefore - EOVBR) * SinWT
        VelTotal = sqrt(VelXAfter ** 2 + VelYAfter ** 2 + VelZAfter ** 2)
        DirCosineX2 = VelXAfter / VelTotal
        DirCosineY2 = VelYAfter / VelTotal
        DirCosineZ2 = VelZAfter / VelTotal
        R2 = random_uniform(RandomSeed)
        # DETERMINATION OF REAL COLLISION TYPE

        # FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        I = MBSort(I, R2, IE, Object)
        while Object.CollisionFrequencyNT[IE][I] < R2:
            I = I + 1

        S1 = Object.RGasNT[I]
        EI = Object.EnergyLevelsNT[I]
        if Object.ElectronNumChangeNT[I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI

        # GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        # COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.InteractionTypeNT[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random_uniform(RandomSeed)

        if Object.AngularModelNT[I] == 1:
            R31 = random_uniform(RandomSeed)
            CosTheta = 1.0 - R3 * Object.AngleCutNT[IE][I]
            if R31 > Object.ScatteringParameterNT[IE][I]:
                CosTheta = -1 * CosTheta
        elif Object.AngularModelNT[I] == 2:
            EPSI = Object.ScatteringParameterNT[IE][I]
            CosTheta = 1 - (2 * R3 * (1 - EPSI)/ (1 + EPSI * (1 - 2 * R3)))
        else:
            CosTheta = 1 - 2 * R3
        Theta = acos(CosTheta)
        R4 = random_uniform(RandomSeed)
        Phi = F4 * R4
        SinPhi = sin(Phi)
        CosPhi = cos(Phi)
        ARG1 = 1 - S1 * EI / E
        ARG1 = max(ARG1, SMALL)

        D = 1 - CosTheta * sqrt(ARG1)
        E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = sqrt((E / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.AngleFromZ = asin(Q * sin(Theta))

        CosZAngle = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CosSquareTheta = CosTheta ** 2

        if CosTheta < 0 and CosSquareTheta > U:
            CosZAngle = -1 * CosZAngle
        SinZAngle = sin(Object.AngleFromZ)
        DirCosineZ2 = min(DirCosineZ2, 1)
        VelTotal = CONST9 * sqrt(E1)
        ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
        if ARGZ == 0:
            DirCosineZ1 = CosZAngle
            DirCosineX1 = CosPhi * SinZAngle
            DirCosineY1 = SinPhi * SinZAngle
        else:
            DirCosineZ1 = DirCosineZ2 * CosZAngle + ARGZ * SinZAngle * SinPhi
            DirCosineY1 = DirCosineY2 * CosZAngle + (SinZAngle / ARGZ) * (DirCosineX2 * CosPhi - DirCosineY2 * DirCosineZ2 * SinPhi)
            DirCosineX1 = DirCosineX2 * CosZAngle - (SinZAngle / ARGZ) * (DirCosineY2 * CosPhi + DirCosineX2 * DirCosineZ2 * SinPhi)
        VelXBefore = DirCosineX1 * VelTotal
        VelYBefore = DirCosineY1 * VelTotal
        VelZBefore = DirCosineZ1 * VelTotal

    return 0




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitCT(Boltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the Boltz object to try a higher FinalEnergy value. 
    
    The object parameter is the Boltz object to be setup and used in the simulation.
    
    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, EOK, VelBeforeM1, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, CosTheta, RAN, EPSI, R4, Phi, SinPhi, CosPhi, ARG1,
    cdef double D, Q, U, CosSquareTheta, CosZAngle, SinZAngle, ARGZ,RTHETA, CONST12, VXLAB, VYLAB, EFX100,EFZ100,TLIM,VelXAfter,VelYAfter,VelZAfter,EOVBR

   

    ISAMP = 20
    SMALL = 1.0e-20

    I = 0


    RTHETA = Object.BField_Angle * np.pi / 180.0
    EFZ100 = Object.EField * 100 * sin(RTHETA)
    EFX100 = Object.EField * 100 * cos(RTHETA)
    F1 = Object.EField * Object.CONST2 * cos(RTHETA)
    F4 =2*np.pi
    EOVBR = Object.EFieldOverBField * sin(RTHETA)
    RandomSeed = Object.Random_Seed
    E1 = Object.InitialElectronEnergy
    TDASH =0.0

    CONST9 = Object.CONST3*0.01
    CONST10 = CONST9**2

    #GENERATE RANDOM NUMBER FOR MAXWELL BOLTZMAN
    GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
    IMBPT = 0

    #INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    VelTotal = CONST9 * sqrt(E1)
    VelXBefore = DirCosineX1 * VelTotal
    VelYBefore = DirCosineY1 * VelTotal
    VelZBefore = DirCosineZ1 * VelTotal

    J2M = Object.MaxNumberOfCollisions / ISAMP

    for J1 in range(int(J2M)):
        if J1 != 0  and not int(str(J1)[-int(log10(J1)):]) and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        elif J1>=1e6 and J1%1e5 ==0 and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        while True:
            R1 = random_uniform(RandomSeed)
            T = -1 * log(R1) / Object.MaxCollisionFreqTotal + TDASH
            TDASH = T
            WBT = Object.AngularSpeedOfRotation * T
            CosWT = cos(WBT)
            SinWT = sin(WBT)
            DZ = (VelZBefore * SinWT + (Object.EFieldOverBField - VelYBefore) * (1 - CosWT)) / Object.AngularSpeedOfRotation
            DX = VelXBefore *T+F1*T*T

            E = E1 + DZ * EFZ100+DX*EFX100

            VelXAfter = VelXBefore+2*F1*T
            VelYAfter = (VelYBefore - EOVBR) * CosWT + VelZBefore * SinWT + EOVBR
            VelZAfter = VelZBefore * CosWT - (VelYBefore - EOVBR) * SinWT

            #FIND IDENTITY OF GAS FOR COLLISION
            GasIndex = 0
            R2 = random_uniform(RandomSeed)
            while (Object.MaxCollisionFreqTotalG[GasIndex] < R2):
                GasIndex += 1
            #CALCULATE GAS VELOCITY VECTORS GasVelX,GasVelY,GasVelZ
            IMBPT += 1
            if IMBPT > 6:
                GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
                IMBPT = 1
            GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]

            EOK = ((VelXAfter - GasVelX) ** 2 + (VelYAfter - GasVelY) ** 2 + (VelZAfter - GasVelZ) ** 2) / CONST10
            IE = int(EOK / Object.ElectronEnergyStep)
            IE = min(IE, 3999)

            # Test FOR REAL OR NULL COLLISION
            R5 = random_uniform(RandomSeed)
            TLIM = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
            if R5 <= TLIM:
                break
        if IE == 3999:
            return 1

        #CALCULATE DIRECTION COSINES OF ELECTRON IN 0 KELVIN FRAME
        TDASH = 0.0
        VelBeforeM1 = 1.0 / (CONST9 * sqrt(EOK))
        DXCOM = (VelXAfter - GasVelX) * VelBeforeM1
        DYCOM = (VelYAfter - GasVelY) * VelBeforeM1
        DZCOM = (VelZAfter - GasVelZ) * VelBeforeM1

        # ---------------------------------------------------------------------
        #     DETERMINATION OF REAL COLLISION TYPE
        # ---------------------------------------------------------------------

        R2 = random_uniform(RandomSeed)

        #FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        I = MBSortT(GasIndex, I, R2, IE, Object)
        while Object.CollisionFrequency[GasIndex][IE][I] < R2:
            I = I + 1


        S1 = Object.RGas[GasIndex][I]
        EI = Object.EnergyLevels[GasIndex][I]
        if Object.ElectronNumChange[GasIndex][I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI

        #  GENEERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        #   COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.InteractionType[GasIndex][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = (S1 * S1) / (S1 - 1)
        R3 = random_uniform(RandomSeed)

        if Object.AngularModel[GasIndex][I] == 1:
            R31 = random_uniform(RandomSeed)
            CosTheta = 1- R3 *Object.AngleCut[GasIndex][IE][I]
            if R31 > Object.ScatteringParameter[GasIndex][IE][I]:
                CosTheta = -1 * CosTheta
            elif Object.AngularModel[GasIndex][I] == 2:
                EPSI = Object.ScatteringParameter[GasIndex][IE][I]
                CosTheta = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
            else:
                CosTheta = 1 - 2 * R3
        Theta = acos(CosTheta)
        R4 = random_uniform(RandomSeed)
        Phi = F4 * R4
        SinPhi = sin(Phi)
        CosPhi = cos(Phi)
        ARG1 = 1 - S1 * EI / EOK
        ARG1 = max(ARG1, SMALL)

        D = 1 - CosTheta * sqrt(ARG1)
        E1 = EOK * (1 - EI / (S1 * EOK) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = sqrt((EOK / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.AngleFromZ = asin(Q * sin(Theta))

        CosZAngle = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CosSquareTheta = CosTheta ** 2

        if CosTheta < 0 and CosSquareTheta > U:
            CosZAngle = -1 * CosZAngle
        SinZAngle = sin(Object.AngleFromZ)
        DZCOM = min(DZCOM, 1)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DirCosineZ1 = CosZAngle
            DirCosineX1 = CosPhi * SinZAngle
            DirCosineY1 = SinPhi * SinZAngle
        else:
            DirCosineZ1 = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi
            DirCosineY1 = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi)
            DirCosineX1 = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi)
        # TRANSFORM VELOCITY VECTORS TO LAB FRAME
        VelTotal = CONST9 * sqrt(E1)
        VelXBefore = DirCosineX1 * VelTotal + GasVelX
        VelYBefore = DirCosineY1 * VelTotal + GasVelY
        VelZBefore = DirCosineZ1 * VelTotal + GasVelZ
        #  CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
        E1 = (VelXBefore * VelXBefore + VelYBefore * VelYBefore + VelZBefore * VelZBefore) / CONST10
        VelBeforeM1 = 1.0 / (CONST9 * sqrt(E1))
        DirCosineX1 = VelXBefore * VelBeforeM1
        DirCosineY1 = VelYBefore * VelBeforeM1
        DirCosineZ1 = VelZBefore * VelBeforeM1

    return 0





@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitT(Boltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the Boltz object to try a higher FinalEnergy value.
    This function is used when there is no magnetic field. 

    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    
    Electric field in z direction.

    The object parameter is the Boltz object to be setup and used in the simulation.
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1,AP, BP, F1, F2, F4, J2M, R5, Test1, R1, T,  E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, EOK, VelBeforeM1, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, CosTheta, RAN, EPSI, R4, Phi, SinPhi, CosPhi, ARG1
    cdef double D, Q, U, CosSquareTheta, CosZAngle, SinZAngle, ARGZ, CONST12, VXLAB, VYLAB, VZLAB


    ISAMP = 10
    SMALL = 1.0e-20
    RandomSeed = Object.Random_Seed
    E1 = Object.InitialElectronEnergy
    N4000 = 4000
    TDASH = 0.0
    CONST5 = Object.CONST3 / 2.0
    CONST9 = Object.CONST3 * 0.01
    CONST10 = CONST9 * CONST9
    GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
    IMBPT = 0
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    BP = pow(Object.EField, 2) * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    F4 = 2 * acos(-1)
    J2M = Object.MaxNumberOfCollisions / ISAMP
    for J1 in range(int(J2M)):
        if J1 != 0  and not int(str(J1)[-int(log10(J1)):]) and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        elif J1>=1e6 and J1%1e5 ==0 and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(Object.Max_Electron_Energy, J1), end='\r')
        while True:
            R1 = random_uniform(RandomSeed)
            T = -1 * log(R1) / Object.MaxCollisionFreqTotal + TDASH
            TDASH = T
            AP = DirCosineZ1 * F2 * sqrt(E1)
            E = E1 + (AP + BP * T) * T
            CONST6 = sqrt(E1 / E)
            DirCosineX2 = DirCosineX1 * CONST6
            DirCosineY2 = DirCosineY1 * CONST6
            DirCosineZ2 = DirCosineZ1 * CONST6 + Object.EField * T * CONST5 / sqrt(E)
            R2 = random_uniform(RandomSeed)
            GasIndex = 0
            for GasIndex in range(Object.NumberOfGases):
                if Object.MaxCollisionFreqTotalG[GasIndex] >= R2:
                    break
            IMBPT += 1
            if (IMBPT > 6):
                GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
                IMBPT = 1
            GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1)]
            IMBPT += 1
            GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1)]
            IMBPT += 1
            GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1)]
            # CALCULATE ELECTRON VELOCITY VECTORS VEX VEY VEZ
            VEX = DirCosineX2 * CONST9 * sqrt(E)
            VEY = DirCosineY2 * CONST9 * sqrt(E)
            VEZ = DirCosineZ2 * CONST9 * sqrt(E)

            EOK = (pow((VEX - GasVelX), 2) + pow((VEY - GasVelY), 2) + pow((VEZ - GasVelZ), 2)) / CONST10
            IE = int(EOK / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            R5 = random_uniform(RandomSeed)
            Test1 = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
            if R5 <= Test1:
                break

        if IE == 3999:
            return 1

        TDASH = 0.0

        VelBeforeM1 = 1 / (CONST9 * sqrt(EOK))
        DXCOM = (VEX - GasVelX) * VelBeforeM1
        DYCOM = (VEY - GasVelY) * VelBeforeM1
        DZCOM = (VEZ - GasVelZ) * VelBeforeM1

        # Determination of real collision type
        R3 = random_uniform(RandomSeed)
        # Find location within 4 units in collision array
        I = MBSortT(GasIndex, I, R3, IE, Object)
        while Object.CollisionFrequency[GasIndex][IE][I] < R3:
            I += 1
        S1 = Object.RGas[GasIndex][I]
        EI = Object.EnergyLevels[GasIndex][I]

        if Object.ElectronNumChange[GasIndex][I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI
        IPT = Object.InteractionType[GasIndex][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = pow(S1 , 2) / (S1 - 1.0)

        # Anisotropic scattering
        R3 = random_uniform(RandomSeed)
        if Object.AngularModel[GasIndex][I] == 1:
            R31 = random_uniform(RandomSeed)
            CosTheta = 1.0 - R3 * Object.AngleCut[GasIndex][IE][I]
            if R31 > Object.ScatteringParameter[GasIndex][IE][I]:
                CosTheta = -1.0 * CosTheta
        elif Object.AngularModel[GasIndex][I] == 2:
            EPSI = Object.ScatteringParameter[GasIndex][IE][I]
            CosTheta = 1.0 - (2.0 * R3 * (1.0 - EPSI) / (1.0 + EPSI * (1.0 - 2.0 * R3)))
        else:
            # Isotropic scattering
            CosTheta = 1.0 - 2.0 * R3

        Theta = acos(CosTheta)
        R4 = random_uniform(RandomSeed)
        Phi = F4 * R4
        SinPhi = sin(Phi)
        CosPhi = cos(Phi)
        ARG1 = 1 - S1 * EI / EOK
        ARG1 = max(ARG1, SMALL)

        D = 1 - CosTheta * sqrt(ARG1)
        E1 = EOK * (1 - EI / (S1 * EOK) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = sqrt((EOK / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.AngleFromZ = asin(Q * sin(Theta))

        CosZAngle = cos(Object.AngleFromZ)
        U = (S1 - 1.0) * (S1 - 1.0) / ARG1
        CosSquareTheta = pow(CosTheta, 2)

        if CosTheta < 0 and CosSquareTheta > U:
            CosZAngle = -1 * CosZAngle
        SinZAngle = sin(Object.AngleFromZ)
        DZCOM = min(DZCOM, 1.0)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DirCosineZ1 = CosZAngle
            DirCosineX1 = CosPhi * SinZAngle
            DirCosineY1 = SinPhi * SinZAngle
        else:
            DirCosineZ1 = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi
            DirCosineY1 = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi)
            DirCosineX1 = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi)


        # Transform velocity vectors to lab frame
        CONST12 = CONST9 * sqrt(E1)
        VXLAB = DirCosineX1 * CONST12 + GasVelX
        VYLAB = DirCosineY1 * CONST12 + GasVelY
        VZLAB = DirCosineZ1 * CONST12 + GasVelZ
        # Calculate energy and direction cosines in lab frame
        E1 = (VXLAB * VXLAB + VYLAB * VYLAB + VZLAB * VZLAB) / CONST10
        VelBeforeM1 = 1.0 / (CONST9 * sqrt(E1))
        DirCosineX1 = VXLAB * VelBeforeM1
        DirCosineY1 = VYLAB * VelBeforeM1
        DirCosineZ1 = VZLAB * VelBeforeM1

    return 0
