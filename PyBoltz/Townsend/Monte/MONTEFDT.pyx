from PyBoltz.Boltz cimport Boltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan
from libc.string cimport memset
from PyBoltz.Boltz cimport drand48
from PyBoltz.MBSorts cimport MBSortT
from libc.stdlib cimport malloc, free
from PyBoltz.Townsend.Monte.MonteVars cimport MonteVars
import cython
import numpy as np
cimport numpy as np
import sys
import sys

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void TCALCT(Boltz Object, MonteVars*MV):
    '''
    :param Object: 
    :param MV: 
    :return: 
    Calculate elapsed time, timestop1, until arrival at next plane, IPlane. 
    If two positive solutions exist, set ISolution to 2, and calculate the second solution, TimeStop1.   
    '''
    cdef double A, B, B2, C1 = 0, C2 = 0, Discriminant, TimeStop1, TimeStop2, PlaneFoundFlag = 0

    # Start by assuming that there is only one real solution that is positive.
    MV.ISolution = 1
    # Calculate the coefficients of the equation
    A = Object.EField * Object.CONST2
    B = sqrt(MV.StartingEnergy) * Object.CONST3 * 0.01 * MV.DirCosineZ1
    B2 = B * B

    PlaneFoundFlag = 0
    # Find the two planes that the electron is between. This is to find the other coefficients of the equation.
    if Object.Z < MV.ZPlanes[1]:
        MV.IPlane = 1
        C1 = Object.Z - MV.ZPlanes[1]
        C2 = 0
    else:
        for J in range(2, Object.NumberOfSpaceSteps + 1):
            if Object.Z < MV.ZPlanes[J]:
                MV.IPlane = J
                C1 = Object.Z - MV.ZPlanes[J]
                C2 = Object.Z - MV.ZPlanes[J - 1]
                PlaneFoundFlag = 1
                break
        if PlaneFoundFlag == 0:
            MV.IPlane = 9
            C1 = Object.Z - MV.ZPlanes[8] - 10 * Object.SpaceStepZ
            C2 = Object.Z - MV.ZPlanes[8]
    PlaneFoundFlag = 0
    # check plane in drift direction( only one positive solution)
    Discriminant = B2 - 4 * A * C1
    if Discriminant < 0.0:
        # Passed final plane (runaway electron)
        MV.TimeStop = -99
        return
    # Calculate TimeStop solutions.
    TimeStop1 = (-1 * B + sqrt(B2 - 4 * A * C1)) / (2 * A)
    TimeStop2 = (-1 * B - sqrt(B2 - 4 * A * C1)) / (2 * A)

    # Find the smallest time that is not negative.
    if TimeStop1 < TimeStop2:
        if TimeStop1 < 0.0:
            MV.TimeStop = TimeStop2
        else:
            MV.TimeStop = TimeStop1

        if MV.IPlane == 1:
            return
    else:
        if TimeStop2 >= 0.0:
            MV.TimeStop = TimeStop2
        else:
            MV.TimeStop = TimeStop1

        if MV.IPlane == 1:
            return
    # check plane in backward direction( only one positive solution)
    Discriminant = B2 - 4 * A * C2
    if Discriminant < 0.0:
        # If it reaches this point then it is not a run-away electron, and the solutions above are the right ones.
        return

    TimeStop1 = (-1 * B + sqrt(Discriminant)) / (2 * A)
    TimeStop2 = (-1 * B - sqrt(Discriminant)) / (2 * A)

    if TimeStop1 < 0.0:
        # If it reaches this point then it is not a run-away electron, and the solutions above are the right ones.
        return

    # Found backward solution. There is two TimeStop values. Set ISolution as 2 as an indicator.
    MV.ISolution = 2
    MV.IPlane -= 1

    # Set the smallest non-negative value as TimeStop, and the other TimeStop1 as the other solution.
    if TimeStop1 >= TimeStop2:
        MV.TimeStop = TimeStop2
        MV.TimeStop1 = TimeStop1
    else:
        MV.TimeStop = TimeStop1
        MV.TimeStop1 = TimeStop2

    return

#Sample Maxwell Boltzman distribution for gas velocities
cdef void GenerateMaxBoltz(double RandomSeed, double *RandomMaxBoltzArray):
    cdef double Ran1, Ran2, TwoPi
    cdef int J
    for J in range(0, 5, 2):
        Ran1 = random_uniform(RandomSeed)
        Ran2 = random_uniform(RandomSeed)
        TwoPi = 2.0 * np.pi
        RandomMaxBoltzArray[J] = sqrt(-1 * log(Ran1)) * cos(Ran2 * TwoPi)
        RandomMaxBoltzArray[J + 1] = sqrt(-1 * log(Ran1)) * sin(Ran2 * TwoPi)

# Function used to start a new primary electron
cdef int NewPrimary(Boltz Object, MonteVars*MV):
    '''
    
    :param Object: 
    :param MV: 
    :return: either a 0 or 1, 1 indicates success. 0 indicates failure.
    '''
    # Add one to the count of primaries
    Object.IPrimary += 1
    # Set the plane to 0, as this is the starting plane
    MV.IPlane = 0
    # This TimeStop value is used so that when the simulation starts, we will get to calculate the real time.
    MV.TimeStop = 1000.0

    if Object.IPrimary > 1:
        # If a primary has been simulated already, and the simulation simulated all the collisions then return 0 to
        # indicate that the simulation is done.
        if MV.Iterator > MV.NumberOfMaxColli:
            Object.IPrimary -= 1
            return 0
        else:
            # Set the values corresponding with the new primary.

            # Zero out all the positions.
            Object.X = 0.0
            Object.Y = 0.0
            Object.Z = 0.0

            # Take the cosine and the energy of the last 200th electron from the simulation.
            MV.DirCosineX1 = MV.DirCosineX100
            MV.DirCosineY1 = MV.DirCosineY100
            MV.DirCosineZ1 = MV.DirCosineZ100
            MV.StartingEnergy = MV.Energy100
            MV.TotalNumberOfElectrons += 1

            # Zero out the times and the number of the plane the electron is starting in.
            Object.TimeSum = 0.0
            MV.TimeSumStart = 0.0
            MV.SpaceZStart = 0.0
            MV.IPlane = 0
    if Object.IPrimary > 10000000:
        print("Too many primaries program stopped!")
        return 0
    return 1


cdef void NewElectron(Boltz Object, MonteVars*MV):
    '''
    Simple function used to zero out the TDash value for the new electron, and to add one to the counter of 
    simulated electrons.
    :param Object: 
    :param MV: 
    :return: 
    '''
    MV.TDash = 0.0
    MV.NumberOfElectron += 1

# Function used to bin the simulation results for each plane.
cdef void SpacePlaneUpdate(Boltz Object, MonteVars*MV):
    cdef double CurrentTime, TimeLeft, A, B, EPlane, VelocityRatio, TimeLeft2, DirCosineZ2
    cdef double XPlane, YPlane, ZPlane, VZPlane, WGHT, RPlane

    # If the given plane is bigger than 8, then this electron is out of range. There is only 8 space planes.
    if MV.IPlane > 8:
        return

    # Get the time the electron needs to get to this plane.
    TimeLeft = MV.TimeStop
    TimeLeft2 = TimeLeft * TimeLeft


    A = MV.AP * TimeLeft
    B = MV.BP * TimeLeft2

    # Calculate the Energy at this plane,
    EPlane = MV.StartingEnergy + A + B

    # Calculate the angle of this electron at the current plane.
    VelocityRatio = sqrt(MV.StartingEnergy / EPlane)
    DirCosineZ2 = (MV.DirCosineZ1 * VelocityRatio + TimeLeft * MV.F2 / (2.0 * sqrt(EPlane)))


    # Calculate the spatial values for this electron at the current plane
    XPlane = Object.X + MV.DirCosineX1 * TimeLeft * sqrt(MV.StartingEnergy) * Object.CONST3 * 0.01

    YPlane = Object.Y + MV.DirCosineY1 * TimeLeft * sqrt(MV.StartingEnergy) * Object.CONST3 * 0.01

    ZPlane = Object.Z + MV.DirCosineZ1 * TimeLeft * sqrt(
        MV.StartingEnergy) * Object.CONST3 * 0.01 + TimeLeft2 * Object.EField * Object.CONST2

    # Velocity in the Z direction
    VZPlane = DirCosineZ2 * sqrt(EPlane) * Object.CONST3 * 0.01

    # This is the absoulte value of the recipocal of velocity.
    WGHT = abs(1 / VZPlane)

    # Radial plane
    RPlane = sqrt(XPlane ** 2 + YPlane ** 2)

    # Normalize the spatial values with respect of the velocity and sum them through all the electrons.
    Object.SXPlanes[MV.IPlane] += XPlane * WGHT
    Object.SYPlanes[MV.IPlane] += YPlane * WGHT
    Object.SZPlanes[MV.IPlane] += ZPlane * WGHT
    Object.SRPlanes[MV.IPlane] += RPlane * WGHT

    # NOT SURE
    Object.TMSPL[MV.IPlane] += (Object.TimeSum + TimeLeft) * WGHT
    Object.TTMSPL[MV.IPlane] += (Object.TimeSum + TimeLeft) * (Object.TimeSum + TimeLeft) * WGHT

    # add the square of the spatial values normalised by the velocity
    Object.SX2Planes[MV.IPlane] += XPlane * XPlane * WGHT
    Object.SY2Planes[MV.IPlane] += YPlane * YPlane * WGHT
    Object.SZ2Planes[MV.IPlane] += ZPlane * ZPlane * WGHT
    Object.SR2Planes[MV.IPlane] += RPlane * RPlane * WGHT

    # Sum the energy and time for all the electrons normalised by the velocity
    Object.SEPlanes[MV.IPlane] += EPlane * WGHT
    Object.STPlanes[MV.IPlane] += WGHT / (Object.TimeSum + TimeLeft)

    # Sum of the velocity normalised by the absolute value of itself. This will be a an integer indicating the number
    # of electrons done.
    Object.SVZPlanes[MV.IPlane] += VZPlane * WGHT

    # store the inverse of the velocity and its square
    Object.STSPlanes[MV.IPlane] += WGHT
    Object.STS2Planes[MV.IPlane] += WGHT * WGHT


cdef int TimeCalculations(Boltz Object, MonteVars*MV):
    '''
    Function used to try to find the right TimeStop values for the elctrons, and control whether a new electron is 
    needed. This happens if the electron crossed the final plane, or if it is a run-away electron.
    :param Object: 
    :param MV: 
    :return: This function return a 0,1, or 2. 0 indicates failure, this means the whole simulation is over. 
    1 indicates a new electron was taken of the storage stack. 
    2 indicates success in calculating the TimeStop values for the current electron. 
    '''
    cdef int Flag
    cdef double EPOT
    while (1):
        #TODO: change to NumberOfSpaceSteps-1 for anode termination
        if MV.IPlane >= Object.NumberOfSpaceSteps + 1 or MV.TimeCalculationFlag == 1:
            # Add up the total space travelled by the primary and secondary electrons.
            Object.TotalSpaceZPrimary += Object.Z
            Object.TotalTimePrimary += Object.TimeSum
            Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart
            Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart

            # If we will be crossing the total number of electrons we have, start a new primary.
            if MV.NumberOfElectron == MV.TotalNumberOfElectrons + 1:
                # Create primary electron
                Flag = NewPrimary(Object, MV)
                NewElectron(Object, MV)
                if Flag == 0:
                    return 0
                MV.NewTimeFlag = 0
                # if we reached this point this means that we need to restart the simulation for this new electron.
                return 1
            # Set the TimeCalculationFlag to 0 for now.
            MV.TimeCalculationFlag = 0

            # Take an electron from the store
            Object.X = MV.XS[MV.ElectronStorageTop]
            Object.Y = MV.YS[MV.ElectronStorageTop]
            Object.Z = MV.ZS[MV.ElectronStorageTop]
            Object.TimeSum = MV.TS[MV.ElectronStorageTop]
            MV.StartingEnergy = MV.ES[MV.ElectronStorageTop]
            MV.DirCosineX1 = MV.DirCosineX[MV.ElectronStorageTop]
            MV.DirCosineY1 = MV.DirCosineY[MV.ElectronStorageTop]
            MV.DirCosineZ1 = MV.DirCosineZ[MV.ElectronStorageTop]
            MV.IPlane = MV.IPlaneS[MV.ElectronStorageTop]
            MV.ElectronStorageTop -= 1
            MV.SpaceZStart = Object.Z
            MV.TimeSumStart = Object.TimeSum

            # New electron is out of the simulation region.
            if Object.Z > Object.MaxSpaceZ:
                # Check if the electron has enough energy to go back to the final z plane.
                EPOT = Object.EField * (Object.Z - Object.MaxSpaceZ) * 100

                # If the starting energy of this electron is less than the potential energy needed, then take another
                # electron from the store and try to calculate the TimeStop value.
                if MV.StartingEnergy < EPOT:
                    MV.NumberOfElectron += 1
                    MV.ISolution = 1
                    # Set this flag to one so that we can go back into this if statement.
                    MV.TimeCalculationFlag = 1
                    continue
            # Try to calculate the TimeStop value
            TCALCT(Object, MV)

            # if the values is -99, then it is a run-away electron.
            if MV.TimeStop == -99:
                # Get a new electron and try again.
                MV.NumberOfElectron += 1
                MV.ISolution = 1
                # Set this flag to one so that we can go back into this if statement.
                MV.TimeCalculationFlag = 1
                continue

            # If there is no problems with the new electron, register it, and indicate to the simualtion that there is a
            # need to calculate its time values using the null skullerad method. This is not the TimeStop value.
            NewElectron(Object, MV)
            MV.NewTimeFlag = 0
            return 1
        if MV.ISolution == 2:
            # If there is two solutions, try the other solution.
            MV.TimeStop = MV.TimeStop1
            MV.ISolution = 1
            # Check if the new time values crossed the TimeStop value. If this happens, update the planes and calculate
            # a new TimeStop value.
            if MV.T >= MV.TimeStop and MV.TOld < MV.TimeStop:
                MV.TimeCalculationFlag = 0
                SpacePlaneUpdate(Object, MV)
                continue
            else:
                # Electron did not cross any of the set time and space boundaries.
                return 2
        else:
            # Electron did not cross any of the set time and space boundaries.
            return 2

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(Boltz Object, int ConsoleOuput):
    '''
    Main function to simulate the primary and secondary electrons going through the 8 sequential space planes.
    :param Object: 
    :param ConsoleOuput: 
    :return: 
    '''
    cdef MonteVars MV
    cdef double VelBefore, VelXBefore, VelYBefore, VelZBefore, VelBeforeM1, DZCOM, DYCOM, DXCOM
    cdef double PenningTransferTime, TempTime, S2, RandomNum2, ARG1, D, U, ARGZ, TempSinZ, TempCosZ, TempPhi, TempSinPhi, TempCosPhi
    cdef double RandomSeed = 0.3, RandomNum, S1, EI, ESEC, EISTR, CosTheta, SinTheta, Phi, SinPhi, CosPhi, Sign, RandomNum1
    cdef double  EAuger, AIS, DirCosineZ2, DirCosineX2, DirCosineY2, EPOT
    cdef double GasVelX, GasVelY, GasVelZ, VelocityRatio, VelXAfter, VelYAfter, VelZAfter, COMEnergy, Test1, A, VelocityInCOM, T2
    cdef int Flag = 1, GasIndex, MaxBoltzNumsUsed, NumCollisions = 0, I, IPT, SecondaryElectronIndex, IAuger, J, NAuger
    cdef int TempPlane, JPrint, IDM1

    # Set up for the simulation variables
    MV.TimeCalculationFlag = 0
    MV.NewTimeFlag = 0
    Object.TimeSum = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.TotalSpaceZPrimary = 0.0
    Object.TotalSpaceZSecondary = 0.0
    Object.IPrimary = 0
    Object.TotalTimePrimary = 0.0
    Object.TotalTimeSecondary = 0.0

    # Calculate the z value at each space plane
    MV.StartingEnergy = Object.InitialElectronEnergy
    for J in range(Object.NumberOfSpaceSteps + 1):
        MV.ZPlanes[J] = Object.SpaceStepZ * J

    MV.Sqrt2M = Object.CONST3 * 0.01  # This should be: sqrt(2m)
    MV.TwoM = pow(MV.Sqrt2M, 2)  # This should be: 2m
    MV.TwoPi = 2.0 * np.pi  # This should be: 2 Pi

    # Reset all the counters, this includes the type of collisions per gas per type, etc...
    Object.reset()

    # Zero out all the plane arrays.
    for J in range(9):
        Object.SXPlanes[J] = 0.0
        Object.SYPlanes[J] = 0.0
        Object.SZPlanes[J] = 0.0
        Object.STPlanes[J] = 0.0
        Object.SEPlanes[J] = 0.0
        Object.SX2Planes[J] = 0.0
        Object.SY2Planes[J] = 0.0
        Object.SZ2Planes[J] = 0.0
        Object.SVZPlanes[J] = 0.0
        Object.STSPlanes[J] = 0.0
        Object.STS2Planes[J] = 0.0
        Object.TMSPL[J] = 0.0
        Object.TTMSPL[J] = 0.0
        Object.SRPlanes[J] = 0.0
        Object.SR2Planes[J] = 0.0
        Object.NumberOfElectronSST[J] = 0.0


    # Zero out the simulation variables to be able to start. (check MonteVars.pxd and Boltz.pxd or the documentation
    # for explanation of each variable here).
    Object.NumberOfElectronSST[9] = 0.0
    MV.ID = 0
    MV.I100 = 0
    MV.TotalNumberOfElectrons = 0
    MV.ElectronStorageTop = 0
    MV.SecondaryElectronFlag = 0
    MV.NumberOfElectronAtt = 0
    MV.Iterator = 0
    MV.NumberOfElectron = 0
    MV.NumberOfCollision = 0
    MV.TotalNumberOfElectrons = 0
    MV.NumberOfNullCollision = 0
    Object.TotalSpaceZPrimary = 0
    Object.TotalTimePrimary = 0
    Object.TotalSpaceZSecondary = 0
    Object.TotalTimeSecondary = 0
    MV.TimeSumStart = 0
    MV.SpaceZStart = 0

    # Sum up the collision frequencies into a single array


    cdef int i = 0, K
    cdef double ** TotalCollFreqIncludingNull = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TotalCollFreqIncludingNull[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TotalCollFreqIncludingNull[K][J] = Object.TotalCollisionFrequency[K][J] + \
                                               Object.TotalCollisionFrequencyNull[K][J]
    AbsFakeIoniz = abs(Object.FakeIonisationsEstimate)

    for J in range(9):
        MV.FakeIonisationsTime[J] = 0.0
        MV.FakeIonisationsSpace[J] = 0.0

    # Generate initial random maxwell boltzman numbers
    GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
    MaxBoltzNumsUsed = 0

    # Initial direction cosines
    MV.DirCosineZ1 = cos(Object.AngleFromZ)
    MV.DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    MV.DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    # Current 200th electron values, used if there is a need for a new primary in under 200 electrons.
    MV.Energy100 = MV.StartingEnergy
    MV.DirCosineZ100 = MV.DirCosineZ1
    MV.DirCosineX100 = MV.DirCosineX1
    MV.DirCosineY100 = MV.DirCosineY1

    MV.NumberOfMaxColli = <int> Object.MaxNumberOfCollisions
    # Here are some constants we will use
    MV.BP = pow(Object.EField, 2) * Object.CONST1  # This should be: 1/2 m e^2 EField^2
    MV.F1 = Object.EField * Object.CONST2
    MV.F2 = Object.EField * Object.CONST3  # This should be: sqrt( m / 2) e EField

    # Create primary electron
    NewPrimary(Object, &MV)
    # register this electron
    NewElectron(Object, &MV)


    while (1):
        # Check if there is a need to calculate a new time for the current electron
        if MV.NewTimeFlag == 0:
            # Sample random time to next collision. T is global total time.
            RandomNum = random_uniform(RandomSeed)
            # This is the formula from Skullerud
            MV.T = -log(RandomNum) / Object.MaxCollisionFreqTotal + MV.TDash
            MV.TOld = MV.TDash
            MV.TDash = MV.T
            MV.AP = MV.DirCosineZ1 * MV.F2 * sqrt(MV.StartingEnergy)

            if MV.T >= MV.TimeStop and MV.TOld < MV.TimeStop:
                SpacePlaneUpdate(Object, &MV)
                MV.TimeCalculationFlag = 0
                Flag = TimeCalculations(Object, &MV)
                if Flag == 1:
                    continue
                elif Flag == 0:
                    break
        # Zero out the flag
        MV.NewTimeFlag = 0
        # similar to MONTET, calculate the energy of the electron after the collision
        MV.Energy = MV.StartingEnergy + (MV.AP + MV.BP * MV.T) * MV.T
        if MV.Energy <= 0.0:
            print("Energy is negative!")
            MV.Energy = 0.001


        # Randomly choose gas to scatter from, based on expected collision freqs.
        GasIndex = 0
        if Object.NumberOfGases == 1:
            RandomNum = random_uniform(RandomSeed)
            GasIndex = 0
        else:
            RandomNum = random_uniform(RandomSeed)
            while (Object.MaxCollisionFreqTotalG[GasIndex] < RandomNum):
                GasIndex = GasIndex + 1

        # Pick random gas molecule velocity for collision
        MaxBoltzNumsUsed += 1
        if (MaxBoltzNumsUsed > 6):
            GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
            MaxBoltzNumsUsed = 1
        GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]
        MaxBoltzNumsUsed += 1
        GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]
        MaxBoltzNumsUsed += 1
        GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]

        # Calculate the values of the new cosines after the collisions.
        VelocityRatio = sqrt(MV.StartingEnergy / MV.Energy)
        AIS = 1.0
        DirCosineZ2 = (MV.DirCosineZ1 * VelocityRatio + MV.T * MV.F2 / (2.0 * sqrt(MV.Energy)))
        DirCosineX2 = MV.DirCosineX1 * VelocityRatio
        DirCosineY2 = MV.DirCosineY1 * VelocityRatio
        if DirCosineZ2 < 0.0:
            AIS = -1.0

        # There is some percision error around 1e-16, that could cause the simulation to crash.
        # Thus the + 2e-15. This willl ensure that whatever in the square root is always positive.
        if 1.0 - (DirCosineX2 ** 2) - (DirCosineY2 ** 2) < 0:
            DirCosineZ2 = 0
        else:
            DirCosineZ2 = AIS * sqrt(1.0 - (DirCosineX2 ** 2) - (DirCosineY2 ** 2))

        # Calculate electron velocity after
        VelXAfter = DirCosineX2 * MV.Sqrt2M * sqrt(MV.Energy)
        VelYAfter = DirCosineY2 * MV.Sqrt2M * sqrt(MV.Energy)
        VelZAfter = DirCosineZ2 * MV.Sqrt2M * sqrt(MV.Energy)

        # Calculate energy in center of mass frame
        #   E = 1/2 m dx^2 + dvy^2 + dvz^2
        #   works if TwoM = 2m
        COMEnergy = (pow((VelXAfter - GasVelX), 2) + pow((VelYAfter - GasVelY), 2) + pow((VelZAfter - GasVelZ),
                                                                                         2)) / MV.TwoM

        # Now the Skullerud null collision method
        RandomNum = random_uniform(RandomSeed)
        MV.iEnergyBin = <int> (COMEnergy / Object.ElectronEnergyStep)
        MV.iEnergyBin = min(3999, MV.iEnergyBin)
        # If we draw below this number, we will null-scatter (no mom xfer)
        Test1 = Object.TotalCollisionFrequency[GasIndex][MV.iEnergyBin] / Object.MaxCollisionFreq[GasIndex]
        if RandomNum > Test1:

            Test2 = TotalCollFreqIncludingNull[GasIndex][MV.iEnergyBin] / Object.MaxCollisionFreq[GasIndex]
            if RandomNum < Test2:
                if Object.NumMomCrossSectionPointsNull[GasIndex] == 0:
                    continue
                RandomNum = random_uniform(RandomSeed)
                I = 0
                while Object.NullCollisionFreq[GasIndex][MV.iEnergyBin][I] < RandomNum:
                    # Add a null scatter
                    I += 1
                Object.ICOLNN[GasIndex][I] += 1
                continue
            else:
                Test3 = (TotalCollFreqIncludingNull[GasIndex][MV.iEnergyBin] + AbsFakeIoniz) / Object.MaxCollisionFreq[
                    GasIndex]
                if RandomNum < Test3:
                    # Increment fake ionization counter
                    Object.FakeIonizations += 1
                    MV.FakeIonisationsSpace[MV.IPlane] += 1
                    if Object.FakeIonisationsEstimate < 0.0:
                        MV.NumberOfElectronAtt += 1
                        # Fake attachment start a new electron
                        if MV.NumberOfElectron == MV.TotalNumberOfElectrons + 1:
                            # Create primary electron if the number of max cascaded electrons is reached
                            Flag = NewPrimary(Object, &MV)
                            NewElectron(Object, &MV)
                            if Flag == 0:
                                # Exit the simulation, something wrong happened.
                                break
                            # We have a new electron ensure that its needed time values are calculated by setting this
                            # flag to 0.
                            MV.NewTimeFlag = 0
                            continue
                        # There is no need to create a primary electron if the simulation reached this point.
                        # Take an electron from the storage stack. This means that we need to calculate the TimeStop value.
                        MV.TimeCalculationFlag = 1

                        # This is done because the TimeCalculation function will add these values back to the same variable.
                        # Even though we don't need to do that since it is a fake attachment.
                        Object.TotalSpaceZPrimary -= Object.Z
                        Object.TotalTimePrimary -= Object.TimeSum
                        Object.TotalTimeSecondary -= Object.TimeSum - MV.TimeSumStart
                        Object.TotalSpaceZSecondary -= Object.Z - MV.SpaceZStart

                        # Call the TimeCalculations function
                        Flag = TimeCalculations(Object, &MV)
                        if Flag == 1:
                            # This means that we have a new electron so simulation needs to calculate its time values.
                            MV.NewTimeFlag = 0
                            continue
                        elif Flag == 0:
                            # This means that something went wrong exit the simulation.
                            break
                        elif Flag == 2:
                            # This means that there is no need for a new electron, or a need to calculate new time values.
                            MV.NewTimeFlag = 1
                            continue

                    # fake Ionisation add electron to the store (S)
                    MV.TotalNumberOfElectrons += 1
                    MV.ElectronStorageTop += 1

                    if MV.ElectronStorageTop > 2000:
                        raise ValueError("More than 2000 stored electrons")

                    # Calculate the spatial values for the electron added to the store.
                    A = MV.T * MV.Sqrt2M * sqrt(MV.StartingEnergy)
                    MV.XS[MV.ElectronStorageTop] = Object.X + MV.DirCosineX1 * A
                    MV.YS[MV.ElectronStorageTop] = Object.Y + MV.DirCosineY1 * A
                    MV.ZS[MV.ElectronStorageTop] = Object.Z + MV.DirCosineZ1 * A + MV.T * MV.T * MV.F1
                    MV.TS[MV.ElectronStorageTop] = Object.TimeSum + MV.T
                    MV.ES[MV.ElectronStorageTop] = MV.Energy

                    # Calculate the cosines of this newly stored electron.
                    AIS = 1.0
                    if DirCosineZ2 < 0.0:
                        AIS = -1.0
                    MV.DirCosineX[MV.ElectronStorageTop] = DirCosineX2
                    MV.DirCosineY[MV.ElectronStorageTop] = DirCosineY2
                    MV.DirCosineZ[MV.ElectronStorageTop] = AIS * sqrt(1.0 - (DirCosineX2 ** 2) - (DirCosineY2 ** 2))
                    IDM1 = 1 + int(MV.ZS[MV.ElectronStorageTop] / Object.SpaceStepZ)
                    if IDM1 < 1: IDM1 = 1
                    if IDM1 > 9: IDM1 = 9

                    # Save the plane the new electron is in.
                    MV.IPlaneS[MV.ElectronStorageTop] = IDM1
                    Object.NumberOfElectronSST[MV.IPlaneS[MV.ElectronStorageTop]] += 1
                    continue
                continue
            continue


        # If we got this far, we have a collision.
        NumCollisions += 1

        #  sqrt(2m E_com) = |v_com|
        VelocityInCOM = (MV.Sqrt2M * sqrt(COMEnergy))

        # Calculate direction cosines of electron in 0 kelvin frame
        DXCOM = (VelXAfter - GasVelX) / VelocityInCOM
        DYCOM = (VelYAfter - GasVelY) / VelocityInCOM
        DZCOM = (VelZAfter - GasVelZ) / VelocityInCOM

        # Keep a running average of mean time between real collisions
        Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * MV.TDash

        # Reset time-to-next-real-collision clock
        MV.TDash = 0.0

        # From above, A = m VBefore a T_total
        #             B = 1/2 m a^2 T_total^2
        # which is accurate, because only null collisions happened,
        # so we had simple uniform acceleration.
        #
        # Calculate the positions before collision and update diffusion and energy calculation
        T2 = MV.T * MV.T
        A = MV.T * MV.Sqrt2M * sqrt(MV.StartingEnergy)
        Object.X += MV.DirCosineX1 * A
        Object.Y += MV.DirCosineY1 * A
        Object.Z += MV.DirCosineZ1 * A + T2 * MV.F1
        # update energy and time bins
        Object.TimeSum += MV.T
        Object.CollisionTimes[min(299, int(MV.T + 1))] += 1
        Object.CollisionEnergies[MV.iEnergyBin] += 1

        # Randomly pick the type of collision we will have
        RandomNum = random_uniform(RandomSeed)

        # Find location within 4 units in collision array
        I = MBSortT(GasIndex, I, RandomNum, MV.iEnergyBin, Object)
        while Object.CollisionFrequency[GasIndex][MV.iEnergyBin][I] < RandomNum:
            I += 1

        S1 = Object.RGas[GasIndex][I]
        EI = Object.EnergyLevels[GasIndex][I]

        if COMEnergy < EI:
            EI = COMEnergy - 0.0001

        if Object.ElectronNumChange[GasIndex][I] == -1:
            # An attachment happened
            MV.NumberOfElectronAtt += 1

            IPT = <long long> Object.InteractionType[GasIndex][I]
            MV.ID += 1
            MV.Iterator += 1

            # Update the relevant counters.
            Object.CollisionsPerGasPerType[GasIndex][<int> IPT - 1] += 1
            Object.ICOLN[GasIndex][I] += 1
            Object.CollisionTimes[min(299, int(MV.T + 1))] += 1

            # This is the end of this electron so add its spatial value to the total.
            Object.TotalSpaceZPrimary += Object.Z
            Object.TotalTimePrimary += Object.TimeSum
            Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart
            Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart

            IDM1 = 1 + int(Object.Z / Object.SpaceStepZ)
            if IDM1 < 1: IDM1 = 1
            if IDM1 > 9: IDM1 = 9
            Object.NumberOfElectronSST[IDM1] -= 1

            # Simulation needs a new electron. Either a primary or a secondary.

            if MV.NumberOfElectron == MV.TotalNumberOfElectrons + 1:
                # Electron captured start a new primary
                Flag = NewPrimary(Object, &MV)
                NewElectron(Object, &MV)
                if Flag == 0:
                    # Something went wrong, exit the simulation.
                    break
                # making a primary was a success calculate the Time values.
                MV.NewTimeFlag = 0
                continue


            # There is no need to create a primary electron if the simulation reached this point.
            # Take an electron from the storage stack. This means that we need to calculate the TimeStop value.
            MV.TimeCalculationFlag = 1

            # This is done because the TimeCalculation function will add these values back to the same variable.
            # Even though we don't need to do that since it is a fake attachment.
            Object.TotalSpaceZPrimary -= Object.Z
            Object.TotalTimePrimary -= Object.TimeSum
            Object.TotalTimeSecondary -= Object.TimeSum - MV.TimeSumStart
            Object.TotalSpaceZSecondary -= Object.Z - MV.SpaceZStart

            # Call the TimeCalculations function
            Flag = TimeCalculations(Object, &MV)
            if Flag == 1:
                # This means that we have a new electron so simulation needs to calculate its time values.
                MV.NewTimeFlag = 0
                continue
            elif Flag == 0:
                # This means that something went wrong exit the simulation.
                break
            elif Flag == 2:
                # This means that there is no need for a new electron, or a need to calculate new time values.
                MV.NewTimeFlag = 1
                continue

        elif Object.ElectronNumChange[GasIndex][I] != 0:
            # An ionisation happened
            EISTR = EI
            RandomNum = random_uniform(RandomSeed)

            # Use Opal Peterson and Beaty splitting factor. This is done to account for the secondary electron
            # emission.
            ESEC = Object.WPL[GasIndex][I] * tan(RandomNum * atan((COMEnergy - EI) / (2 * Object.WPL[GasIndex][I])))
            ESEC = Object.WPL[GasIndex][I] * (ESEC / Object.WPL[GasIndex][I]) ** 0.9524

            EI = ESEC + EI
            if EI!=EI:
                sys.exit()
            # Store position, energy, direction, and time of generation of ionisation electron
            MV.TotalNumberOfElectrons += 1
            MV.ElectronStorageTop += 1

            if MV.ElectronStorageTop > 2000:
                raise ValueError("More than 2000 stored electrons")

            MV.XS[MV.ElectronStorageTop] = Object.X
            MV.YS[MV.ElectronStorageTop] = Object.Y
            MV.ZS[MV.ElectronStorageTop] = Object.Z
            MV.TS[MV.ElectronStorageTop] = Object.TimeSum
            # All spatial values are the same as the original electron. The Energy is split however.
            MV.ES[MV.ElectronStorageTop] = ESEC

            # Set the flag so that the simulation knows that it needs to account for its cosines.
            # Note that they are not stored here.
            MV.SecondaryElectronFlag = 1
            SecondaryElectronIndex = MV.ElectronStorageTop
            IDM1 = 1 + int(Object.Z / Object.SpaceStepZ)
            if IDM1 < 1: IDM1 = 1
            if IDM1 > 9: IDM1 = 9
            MV.IPlaneS[MV.ElectronStorageTop] = IDM1
            Object.NumberOfElectronSST[MV.IPlaneS[MV.ElectronStorageTop]] += 1

            # Store possible shell emissions auger or fluorescence, update the angles and cosines
            if EISTR > 30.0:
                # Auger Emission without fluorescence
                NAuger = <int> Object.NC0[GasIndex][I]
                EAuger = Object.EC0[GasIndex][I] / NAuger
                for J in range(NAuger):
                    # For each one, register the electron and store it.
                    MV.TotalNumberOfElectrons += 1
                    MV.ElectronStorageTop += 1
                    MV.XS[MV.ElectronStorageTop] = Object.X
                    MV.YS[MV.ElectronStorageTop] = Object.Y
                    MV.ZS[MV.ElectronStorageTop] = Object.Z
                    MV.TS[MV.ElectronStorageTop] = Object.TimeSum
                    MV.ES[MV.ElectronStorageTop] = EAuger
                    RandomNum = random_uniform(RandomSeed)

                    # Angular distribution (isotropic scattering)
                    CosTheta = 1.0 - 2 * RandomNum
                    Theta = acos(CosTheta)
                    CosTheta = cos(Theta)
                    SinTheta = sin(Theta)

                    RandomNum = random_uniform(RandomSeed)
                    Phi = MV.TwoPi * RandomNum
                    SinPhi = sin(Phi)
                    CosPhi = cos(Phi)
                    MV.DirCosineZ[MV.ElectronStorageTop] = CosTheta
                    MV.DirCosineX[MV.ElectronStorageTop] = CosPhi * SinTheta
                    MV.DirCosineY[MV.ElectronStorageTop] = SinPhi * SinTheta
                    IDM1 = 1 + int(MV.ZS[MV.ElectronStorageTop] / Object.SpaceStepZ)
                    if IDM1 < 1: IDM1 = 1
                    if IDM1 > 9: IDM1 = 9
                    MV.IPlaneS[MV.ElectronStorageTop] = IDM1
                    Object.NumberOfElectronSST[MV.IPlaneS[MV.ElectronStorageTop]] += 1

        # Generate scattering angles and update laboratory cosines after collision also update energy of electron
        IPT = <long long> Object.InteractionType[GasIndex][I]
        MV.ID += 1
        MV.Iterator += 1
        Object.CollisionsPerGasPerType[GasIndex][<int> IPT - 1] += 1
        Object.ICOLN[GasIndex][I] += 1
        # If it is an excitation then add the probability
        # of transfer to give ionisation of the other gases in the mixture
        if Object.Enable_Penning != 0:
            if Object.PenningFraction[GasIndex][0][I] != 0:
                RandomNum = random_uniform(RandomSeed)
                if RandomNum <= Object.PenningFraction[GasIndex][0][I]:
                    MV.TotalNumberOfElectrons += 1
                    MV.ElectronStorageTop += 1
                    if MV.ElectronStorageTop > 2000:
                        raise ValueError("More than 2000 stored electrons")
                    # Possible delocalisation length for penning transfer
                    if Object.PenningFraction[GasIndex][1][I] == 0.0:
                        MV.XS[MV.ElectronStorageTop] = Object.X
                        MV.YS[MV.ElectronStorageTop] = Object.Y
                        MV.ZS[MV.ElectronStorageTop] = Object.Z
                    else:
                        Sign = 1.0
                        RandomNum = random_uniform(RandomSeed)
                        if RandomNum < 0.5:
                            Sign = -1 * Sign
                        RandomNum = random_uniform(RandomSeed)
                        MV.XS[MV.ElectronStorageTop] = Object.X - log(RandomNum) * Object.PenningFraction[GasIndex][1][
                            I] * Sign

                        RandomNum = random_uniform(RandomSeed)
                        if RandomNum < 0.5:
                            Sign = -1 * Sign
                        RandomNum = random_uniform(RandomSeed)
                        MV.YS[MV.ElectronStorageTop] = Object.Y - log(RandomNum) * Object.PenningFraction[GasIndex][1][
                            I] * Sign

                        RandomNum = random_uniform(RandomSeed)
                        if RandomNum < 0.5:
                            Sign = -1 * Sign
                        RandomNum = random_uniform(RandomSeed)
                        MV.ZS[MV.ElectronStorageTop] = Object.Z - log(RandomNum) * Object.PenningFraction[GasIndex][1][
                            I] * Sign
                    if MV.ZS[MV.ElectronStorageTop] < 0.0 or MV.ZS[MV.ElectronStorageTop] > Object.MaxSpaceZ:
                        # Penning happens after final time plane, dont store the electron
                        MV.ElectronStorageTop -= 1
                        MV.TotalNumberOfElectrons -= 1
                    else:
                        # Possible penning transfer time
                        PenningTransferTime = Object.TimeSum
                        if Object.PenningFraction[GasIndex][2][I] != 0.0:
                            RandomNum = random_uniform(RandomSeed)
                            PenningTransferTime = Object.TimeSum - log(RandomNum) * Object.PenningFraction[GasIndex][2][
                                I]
                        MV.TS[MV.ElectronStorageTop] = PenningTransferTime
                        MV.ES[MV.ElectronStorageTop] = 1.0
                        MV.DirCosineX[MV.ElectronStorageTop] = MV.DirCosineX1
                        MV.DirCosineY[MV.ElectronStorageTop] = MV.DirCosineY1
                        MV.DirCosineZ[MV.ElectronStorageTop] = MV.DirCosineZ1
                        IDM1 = 1 + int(MV.ZS[MV.ElectronStorageTop] / Object.SpaceStepZ)
                        if IDM1 < 1: IDM1 = 1
                        if IDM1 > 9: IDM1 = 9
                        MV.IPlaneS[MV.ElectronStorageTop] = IDM1
                        Object.NumberOfElectronSST[MV.IPlaneS[MV.ElectronStorageTop]] += 1

        S2 = (S1 * S1) / (S1 - 1.0)

        # Anisotropic scattering - pick the scattering angle theta depending on scatter type
        RandomNum = random_uniform(RandomSeed)
        if Object.AngularModel[GasIndex][I] == 1:
            # Use method of Capitelli et al
            RandomNum2 = random_uniform(RandomSeed)
            CosTheta = 1.0 - RandomNum * Object.AngleCut[GasIndex][MV.iEnergyBin][I]
            if RandomNum2 > Object.ScatteringParameter[GasIndex][MV.iEnergyBin][I]:
                CosTheta = -1.0 * CosTheta
        elif Object.AngularModel[GasIndex][I] == 2:
            # Use method of Okhrimovskyy et al
            EpsilonOkhr = Object.ScatteringParameter[GasIndex][MV.iEnergyBin][I]
            CosTheta = 1.0 - (2.0 * RandomNum * (1.0 - EpsilonOkhr) / (1.0 + EpsilonOkhr * (1.0 - 2.0 * RandomNum)))
        else:
            # Isotropic scattering
            CosTheta = 1.0 - 2.0 * RandomNum

        Theta = acos(CosTheta)

        # Pick a random Phi - must be uniform by symmetry of the gas
        RandomNum = random_uniform(RandomSeed)
        Phi = MV.TwoPi * RandomNum
        SinPhi = sin(Phi)
        CosPhi = cos(Phi)

        if COMEnergy < EI:
            EI = 0.0

        ARG1 = max(1.0 - S1 * EI / COMEnergy, Object.SmallNumber)

        D = 1.0 - CosTheta * sqrt(ARG1)
        U = (S1 - 1) * (S1 - 1) / ARG1

        # Update the energy to start drifing for the next round.
        #  If its zero, make it small but nonzero.
        MV.StartingEnergy = max(COMEnergy * (1.0 - EI / (S1 * COMEnergy) - 2.0 * D / S2), Object.SmallNumber)

        Q = min(sqrt((COMEnergy / MV.StartingEnergy) * ARG1) / S1, 1.0)

        # Calculate angle of scattering from Z direction
        Object.AngleFromZ = asin(Q * sin(Theta))
        CosZAngle = cos(Object.AngleFromZ)

        # Find new directons after scatter
        if CosTheta < 0 and CosTheta * CosTheta > U:
            CosZAngle = -1 * CosZAngle
        SinZAngle = sin(Object.AngleFromZ)
        DZCOM = min(DZCOM, 1.0)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)

        if ARGZ == 0.0:
            # If scattering frame is same as lab frame, do this;
            MV.DirCosineZ1 = CosZAngle
            MV.DirCosineX1 = CosPhi * SinZAngle
            MV.DirCosineY1 = SinPhi * SinZAngle
            if MV.SecondaryElectronFlag == 1:
                # Use free kinematics for ionisation secondary angle for the secondary electron.
                TempSinZ = SinZAngle * sqrt(MV.StartingEnergy / MV.ES[SecondaryElectronIndex])
                if TempSinZ > 1:
                    TempSinZ = 1.0

                TempSinZ = sin(asin(TempSinZ))
                TempCosZ = cos(asin(TempSinZ))

                if TempCosZ < 0.0:
                    TempCosZ = -1 * TempCosZ

                TempPhi = Phi + np.pi

                if TempPhi > MV.TwoPi:
                    TempPhi = Phi - MV.TwoPi

                TempSinPhi = sin(TempPhi)
                TempCosPhi = cos(TempPhi)

                MV.DirCosineZ[SecondaryElectronIndex] = TempCosZ
                MV.DirCosineX[SecondaryElectronIndex] = TempCosPhi * TempSinZ
                MV.DirCosineY[SecondaryElectronIndex] = TempSinPhi * TempSinZ
                MV.SecondaryElectronFlag = 0
        else:
            # Otherwise do this.
            MV.DirCosineZ1 = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi
            MV.DirCosineY1 = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi)
            MV.DirCosineX1 = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi)
            if MV.SecondaryElectronFlag == 1:
                # Use free kinematics for ionisation secondary angle for the secondary electron
                TempSinZ = SinZAngle * sqrt(MV.StartingEnergy / MV.ES[SecondaryElectronIndex])
                if TempSinZ > 1:
                    TempSinZ = 1.0

                TempSinZ = sin(asin(TempSinZ))
                TempCosZ = cos(asin(TempSinZ))

                if TempCosZ < 0.0:
                    TempCosZ = -1 * TempCosZ

                TempPhi = Phi + np.pi

                if TempPhi > MV.TwoPi:
                    TempPhi = Phi - MV.TwoPi

                TempSinPhi = sin(TempPhi)
                TempCosPhi = cos(TempPhi)

                MV.DirCosineZ[SecondaryElectronIndex] = DZCOM * TempCosZ + ARGZ * TempSinZ * TempSinPhi

                MV.DirCosineX[SecondaryElectronIndex] = DXCOM * TempCosZ - (TempSinZ / ARGZ) * (
                        DYCOM * TempCosPhi + DXCOM * DZCOM * TempSinPhi)

                MV.DirCosineY[SecondaryElectronIndex] = DYCOM * TempCosZ + (TempSinZ / ARGZ) * (
                        DXCOM * TempCosPhi - DYCOM * DZCOM * TempSinPhi)
                MV.SecondaryElectronFlag = 0

        # Transform velocity vectors to lab frame
        VelBefore = MV.Sqrt2M * sqrt(MV.StartingEnergy)
        VelXBefore = MV.DirCosineX1 * VelBefore + GasVelX
        VelYBefore = MV.DirCosineY1 * VelBefore + GasVelY
        VelZBefore = MV.DirCosineZ1 * VelBefore + GasVelZ

        # Calculate energy and direction cosines in lab frame
        MV.StartingEnergy = (VelXBefore * VelXBefore + VelYBefore * VelYBefore + VelZBefore * VelZBefore) / MV.TwoM
        VelBeforeM1 = 1 / (MV.Sqrt2M * sqrt(MV.StartingEnergy))
        MV.DirCosineX1 = VelXBefore * VelBeforeM1
        MV.DirCosineY1 = VelYBefore * VelBeforeM1
        MV.DirCosineZ1 = VelZBefore * VelBeforeM1

        MV.I100 += 1
        if MV.I100 == 200:
            # These willl be the value of the starting cosines for the new primary.
            MV.DirCosineZ100 = MV.DirCosineZ1
            MV.DirCosineX100 = MV.DirCosineX1
            MV.DirCosineY100 = MV.DirCosineY1
            MV.Energy100 = MV.StartingEnergy
            MV.I100 = 0

        # Check if there is a need to do a TimeCalculation. This will happen if the electron crosses the final plane.
        if Object.Z > Object.MaxSpaceZ:
            EPOT = Object.EField * (Object.Z - Object.MaxSpaceZ) * 100
            if MV.StartingEnergy < EPOT:
                MV.TimeCalculationFlag = 1
                Flag = TimeCalculations(Object, &MV)
                if Flag == 1:
                    MV.NewTimeFlag = 0
                    continue
                elif Flag == 0:
                    break
                elif Flag == 2:
                    MV.NewTimeFlag = 1
                    continue
        # Calculate new TimeStop value
        TCALCT(Object, &MV)

        # If it is a run-away electron, call the TimeCalculations function to get a new electron and try again.
        if MV.TimeStop == -99:
            MV.TimeCalculationFlag = 1
            Flag = TimeCalculations(Object, &MV)
            if Flag == 1:
                MV.NewTimeFlag = 0
                continue
            elif Flag == 0:
                break
            elif Flag == 2:
                MV.NewTimeFlag = 1
                continue

    # Get the ratios of attachment and ionisation into the desired forms for the later functions
    if MV.NumberOfElectron > Object.IPrimary:
        Object.AttachmentOverIonisation = MV.NumberOfElectronAtt / (MV.NumberOfElectron - Object.IPrimary)
        Object.AttachmentErr = sqrt(MV.NumberOfElectronAtt) / MV.NumberOfElectronAtt
        Object.AttachmentOverIonisationErr = sqrt(MV.NumberOfElectron - Object.IPrimary) / (
                    MV.NumberOfElectron - Object.IPrimary)
    else:
        Object.AttachmentOverIonisation = -1
        Object.AttachmentErr = sqrt(MV.NumberOfElectronAtt) / MV.NumberOfElectronAtt

    if ConsoleOuput:
        print(
            '\nSimulation through {} Space planes:\n Total number of Electrons: {:10.1f}\n Number of Negative Ions: {:10.1f}\n Number of primaries: {:10.1f}\n'.format(
                Object.NumberOfSpaceSteps,
                MV.NumberOfElectron,
                MV.NumberOfElectronAtt, Object.IPrimary))

    for i in range(6):
        free(TotalCollFreqIncludingNull[i])
    free(TotalCollFreqIncludingNull)
