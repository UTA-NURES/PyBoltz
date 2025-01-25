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
    Object.IPrimary += 1
    if Object.IPrimary > 1:
        # If a primary has been simulated already, and the simulation simulated all the collisions then return 0 to
        # indicate that the simulation is done.
        if MV.Iterator > MV.NumberOfMaxColli:
            Object.IPrimary -= 1
            return 0
        else:
            # Set the values corresponding with the new primary

            # Zero out all the positions
            Object.X = 0.0
            Object.Y = 0.0
            Object.Z = 0.0

            # Take the cosine and the energy of the last 200th electron from the simulation
            MV.DirCosineX1 = MV.DirCosineX100
            MV.DirCosineY1 = MV.DirCosineY100
            MV.DirCosineZ1 = MV.DirCosineZ100
            MV.StartingEnergy = MV.Energy100

            MV.TotalNumberOfElectrons += 1

            # Zero out the times and the number of the plane the electron is starting in
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
    simulated electrons. This is where the TimeStop is calculated (much simpler than the steady state simulation since
    This simulation is run on sequential time planes).
    :param Object: 
    :param MV: 
    :return: 
    '''
    MV.TDash = 0.0
    MV.NumberOfElectron += 1
    MV.TimeStop = Object.TimeStep + MV.IPlane * Object.TimeStep

# Function used to bin the simulation results for each plane.
cdef void TimePlanesUpdate(Boltz Object, MonteVars*MV):
    cdef double CurrentTime, TimeLeft, A, B, EPlane, VelocityRatio, TimeLeft2, DirCosineZ2
    cdef double XPlane, YPlane, ZPlane, VZPlane
    # Get the time the electron needs to get to this plane
    CurrentTime = MV.IPlane * Object.TimeStep
    # Time left to reach IPlane
    TimeLeft = CurrentTime - Object.TimeSum
    TimeLeft2 = TimeLeft * TimeLeft


    A = MV.AP * TimeLeft
    B = MV.BP * TimeLeft2

    # Calculate the Energy at this plane
    EPlane = MV.StartingEnergy + A + B

    # Calculate the angle of this electron at the current plane
    VelocityRatio = sqrt(MV.StartingEnergy / EPlane)
    DirCosineZ2 = (MV.DirCosineZ1 * VelocityRatio + TimeLeft * MV.F2 / (2.0 * sqrt(EPlane)))

    # Calculate the spatial values for this electron at the current plane.
    XPlane = Object.X + MV.DirCosineX1 * TimeLeft * sqrt(MV.StartingEnergy) * Object.CONST3 * 0.01
    YPlane = Object.Y + MV.DirCosineY1 * TimeLeft * sqrt(MV.StartingEnergy) * Object.CONST3 * 0.01
    ZPlane = Object.Z + MV.DirCosineZ1 * TimeLeft * sqrt(
        MV.StartingEnergy) * Object.CONST3 * 0.01 + TimeLeft2 * Object.EField * Object.CONST2

    # Sum the new positions of this electron in this plane
    Object.TXPlanes[MV.IPlane] += XPlane
    Object.TYPlanes[MV.IPlane] += YPlane
    Object.TZPlanes[MV.IPlane] += ZPlane

    # Velocity in the Z direction
    VZPlane = DirCosineZ2 * sqrt(EPlane) * Object.CONST3 * 0.01

    # Sum the square of the new positions of this electron in the plane
    Object.TX2Planes[MV.IPlane] += XPlane * XPlane
    Object.TY2Planes[MV.IPlane] += YPlane * YPlane
    Object.TZ2Planes[MV.IPlane] += ZPlane * ZPlane

    # Sum the energy,time, and velocity of the electron
    Object.TEPlanes[MV.IPlane] += EPlane
    Object.TTPlanes[MV.IPlane] += Object.TimeSum + TimeLeft
    Object.TVZPlanes[MV.IPlane] += VZPlane

    Object.NumberOfElectronsPlanes[MV.IPlane] += 1

    return

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(Boltz Object, int ConsoleOuput):
    '''
    Main function to simulate the primary and secondary electrons going through the 7 sequential time planes.
    :param Object: 
    :param ConsoleOuput: 
    :return: 
    '''
    # Some of the simulation variables are put into the MonteVars struct
    cdef MonteVars MV
    cdef double VelBefore, VelXBefore, VelYBefore, VelZBefore, VelBeforeM1, DZCOM, DYCOM, DXCOM
    cdef double PenningTransferTime, TempTime, S2, RandomNum2, ARG1, D, U, ARGZ, TempSinZ, TempCosZ, TempPhi, TempSinPhi, TempCosPhi
    cdef double RandomSeed = 0.3, RandomNum, S1, EI, ESEC, EISTR, CosTheta, SinTheta, Phi, SinPhi, CosPhi, Sign, RandomNum1
    cdef double EAuger
    cdef double GasVelX, GasVelY, GasVelZ, VelocityRatio, VelXAfter, VelYAfter, VelZAfter, COMEnergy, Test1, A, VelocityInCOM, T2
    cdef int IPlaneS[2001], Flag, GasIndex, MaxBoltzNumsUsed, NumCollisions = 0, I, IPT, SecondaryElectronIndex, IAuger, J, NAuger
    cdef int TempPlane, JPrint, J1 = 1

    # Setup for the simulation variables

    # Set a minimum for the number of collisions to be simulated, the minimum is 80000000
    if ConsoleOuput == 0 and Object.MaxNumberOfCollisions > 80000000:
        MV.NumberOfMaxColli = 80000000
    else:
        MV.NumberOfMaxColli = <int> Object.MaxNumberOfCollisions

    # Zero out the variables used check out Boltz.pxd and MonteVars.pxd for more details.
    MV.ID = 0
    MV.I100 = 0
    MV.NumberOfCollision = 0
    MV.NumberOfNullCollision = 0
    MV.NumberOfElectron = 0
    MV.NumberOfElectronAtt = 0
    MV.SecondaryElectronFlag = 0
    Object.X = 0
    Object.Y = 0
    Object.Z = 0
    MV.ElectronStorageTop = 0
    Object.IPrimary = 0
    MV.TotalNumberOfElectrons = 0
    MV.IPlane = 1
    MV.SpaceZStart = 0.0
    Object.TimeSum = 0.0
    MV.TimeSumStart = 0.0
    MV.AbsFakeIoniz = 0.0
    MV.TimeStop = 0.0
    MV.TDash = 0.0
    MV.T = 0.0
    MV.AP = 0.0
    MV.BP = 0.0
    MV.Iterator = 0

    # Zero out all the plane arrays
    for J in range(9):
        Object.TXPlanes[J] = 0.0
        Object.TYPlanes[J] = 0.0
        Object.TZPlanes[J] = 0.0
        Object.TX2Planes[J] = 0.0
        Object.TY2Planes[J] = 0.0
        Object.TZ2Planes[J] = 0.0
        Object.TEPlanes[J] = 0.0
        Object.TTPlanes[J] = 0.0
        Object.TVZPlanes[J] = 0.0
        Object.NumberOfElectronsPlanes[J] = 0.0
        MV.FakeIonisationsTime[J] = 0.0

    Object.TotalSpaceZPrimary = 0.0
    Object.TotalTimePrimary = 0.0
    Object.TotalSpaceZSecondary = 0.0
    Object.TotalTimeSecondary = 0.0
    if ConsoleOuput != 0:
        MV.NumberOfMaxColli = <int> Object.MaxNumberOfCollisions

    MV.StartingEnergy = Object.InitialElectronEnergy

    MV.Sqrt2M = Object.CONST3 * 0.01  # This should be: sqrt(2m)
    MV.TwoM = pow(MV.Sqrt2M, 2)  # This should be: 2m
    MV.TwoPi = 2.0 * np.pi  # This should be: 2 Pi

    # Reset all the counters, this includes the type of collisions per gas per type, etc...
    Object.reset()

    JPrint = MV.NumberOfMaxColli / 10

    # Sum up the collision frequencies into a single array
    cdef int i = 0, K
    cdef double ** TotalCollFreqIncludingNull = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TotalCollFreqIncludingNull[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TotalCollFreqIncludingNull[K][J] = Object.TotalCollisionFrequency[K][J] + \
                                               Object.TotalCollisionFrequencyNull[K][J]
    AbsFakeIoniz = Object.FakeIonisationsEstimate


    for J in range(9):
        MV.FakeIonisationsTime[J] = 0.0
        MV.FakeIonisationsSpace[J] = 0.0

    # Initial direction cosines
    MV.DirCosineZ1 = cos(Object.AngleFromZ)
    MV.DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    MV.DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    # Current 200th electron values, used if there is a need for a new primary in under 200 electrons.
    MV.Energy100 = MV.StartingEnergy
    MV.DirCosineZ100 = MV.DirCosineZ1
    MV.DirCosineX100 = MV.DirCosineX1
    MV.DirCosineY100 = MV.DirCosineY1

    # Generate initial random maxwell boltzman numbers
    GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
    MaxBoltzNumsUsed = 0

    # Here are some constants we will use
    MV.BP = pow(Object.EField, 2) * Object.CONST1  # This should be: 1/2 m e^2 EField^2
    MV.F1 = Object.EField * Object.CONST2
    MV.F2 = Object.EField * Object.CONST3  # This should be: sqrt( m / 2) e EField

    # Create primary electron
    NewPrimary(Object, &MV)
    # register this electron
    NewElectron(Object, &MV)
    while (1):
        # Sample random time to next collision. T is global total time.
        RandomNum = random_uniform(RandomSeed)
        # This is the formula from Skullerud
        MV.T = -log(RandomNum) / Object.MaxCollisionFreqTotal + MV.TDash
        MV.TDash = MV.T
        MV.AP = MV.DirCosineZ1 * MV.F2 * sqrt(MV.StartingEnergy)

        # check if the total time is bigger than the time for the current plane(TimeStop)
        if MV.T + Object.TimeSum >= MV.TimeStop:
            # move to the next plane and update the time for the current plane
            MV.IPlane += 1
            MV.TimeStop += Object.TimeStep
            TimePlanesUpdate(Object, &MV)
            # while the total time is bigger than the time for the current plane
            while MV.T + Object.TimeSum >= MV.TimeStop:

                # move to the next plane and update the time for the current plane
                MV.IPlane += 1
                MV.TimeStop += Object.TimeStep
                TimePlanesUpdate(Object, &MV)

                if MV.T + Object.TimeSum >= MV.TimeStop and MV.TimeStop <= Object.MaxTime:
                    continue
                else:
                    break
            if MV.T + Object.TimeSum >= Object.MaxTime:
                # total time is bigger than the maximum time set for the simulation
                # update the space and time for the two electrons
                Object.TotalSpaceZPrimary += Object.Z
                Object.TotalTimePrimary += Object.TimeSum
                Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart
                Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart
                MV.TimeStop = Object.TimeStep
                if MV.NumberOfElectron == MV.TotalNumberOfElectrons + 1:
                    # Create primary electron
                    Flag = NewPrimary(Object, &MV)
                    NewElectron(Object, &MV)
                    if Flag == 0:
                        break
                    continue
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

                # register this electron
                NewElectron(Object, &MV)
                continue
        # similar to MONTET, calculate the energy of the electron after the collision
        MV.Energy = MV.StartingEnergy + (MV.AP + MV.BP * MV.T) * MV.T
        if MV.Energy < 0.0:
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

        #Update velocity vectors following field acceleration
        VelocityRatio = sqrt(MV.StartingEnergy / MV.Energy)
        VelXAfter = MV.DirCosineX1 * VelocityRatio * MV.Sqrt2M * sqrt(MV.Energy)
        VelYAfter = MV.DirCosineY1 * VelocityRatio * MV.Sqrt2M * sqrt(MV.Energy)
        VelZAfter = (MV.DirCosineZ1 * VelocityRatio + MV.T * MV.F2 / (2.0 * sqrt(MV.Energy))) * MV.Sqrt2M * sqrt(
            MV.Energy)
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
                    MV.FakeIonisationsTime[MV.IPlane + 1] += 1
                    if Object.FakeIonisationsEstimate < 0.0:
                        MV.NumberOfElectronAtt += 1

                        # Fake attachment start a new electron
                        if MV.NumberOfElectron == MV.TotalNumberOfElectrons + 1:
                            # Create primary electron if the number of max cascaded electrons is reached
                            Flag = NewPrimary(Object, &MV)
                            NewElectron(Object, &MV)
                            if Flag == 0:
                                break
                            continue

                        # get a new electron from store
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

                        # register this electron
                        NewElectron(Object, &MV)
                        continue

                    # fake Ionisation add electron to the store (S)

                    MV.TotalNumberOfElectrons += 1
                    MV.ElectronStorageTop += 1

                    if MV.ElectronStorageTop > 2000:
                        raise ValueError("More than 2000 stored electrons")
                    A = MV.T * MV.Sqrt2M * sqrt(MV.StartingEnergy)
                    MV.XS[MV.ElectronStorageTop] = Object.X + MV.DirCosineX1 * A
                    MV.YS[MV.ElectronStorageTop] = Object.Y + MV.DirCosineY1 * A
                    MV.ZS[MV.ElectronStorageTop] = Object.Z + MV.DirCosineZ1 * A + MV.T * MV.T * MV.F1
                    MV.TS[MV.ElectronStorageTop] = Object.TimeSum + MV.T
                    MV.ES[MV.ElectronStorageTop] = MV.Energy
                    MV.IPlaneS[MV.ElectronStorageTop] = MV.IPlane
                    MV.DirCosineX[MV.ElectronStorageTop] = MV.DirCosineX1 * VelocityRatio
                    MV.DirCosineY[MV.ElectronStorageTop] = MV.DirCosineY1 * VelocityRatio
                    MV.DirCosineZ[MV.ElectronStorageTop] = (MV.DirCosineZ1 * VelocityRatio + MV.T * MV.F2 / (2.0 * sqrt(MV.Energy)))
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
            Object.CollisionsPerGasPerType[GasIndex][<int> IPT - 1] += 1
            Object.ICOLN[GasIndex][I] += 1
            Object.CollisionTimes[min(299, int(MV.T + 1))] += 1

            # Add the time for this electron up, as this electron will be taken our of the simulation
            Object.TotalSpaceZPrimary += Object.Z
            Object.TotalTimePrimary += Object.TimeSum
            Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart
            Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart

            if MV.NumberOfElectron == MV.TotalNumberOfElectrons + 1:
                # Electron captured start a new primary
                Flag = NewPrimary(Object, &MV)
                NewElectron(Object, &MV)
                if Flag == 0:
                    break
                continue
            # get a new electron
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

            # register this electron
            NewElectron(Object, &MV)
            continue
        elif Object.ElectronNumChange[GasIndex][I] != 0:
            # An ionisation happened
            EISTR = EI
            RandomNum = random_uniform(RandomSeed)

            # Use Opal Peterson and Beaty splitting factor
            ESEC = Object.WPL[GasIndex][I] * tan(RandomNum * atan((COMEnergy - EI) / (2 * Object.WPL[GasIndex][I])))
            ESEC = Object.WPL[GasIndex][I] * (ESEC / Object.WPL[GasIndex][I]) ** 0.9524

            EI = ESEC + EI

            # Store position, energy, direction, and time of generation of ionisation electron
            MV.TotalNumberOfElectrons += 1
            MV.ElectronStorageTop += 1


            if MV.ElectronStorageTop > 2000:
                raise ValueError("More than 2000 stored electrons")

            # Store the secondary electron
            MV.XS[MV.ElectronStorageTop] = Object.X
            MV.YS[MV.ElectronStorageTop] = Object.Y
            MV.ZS[MV.ElectronStorageTop] = Object.Z
            MV.TS[MV.ElectronStorageTop] = Object.TimeSum
            MV.ES[MV.ElectronStorageTop] = ESEC
            MV.IPlaneS[MV.ElectronStorageTop] = MV.IPlane

            # Set the correct flag up to indicate that there is a secondary electron being simulated.
            # This is done so that the simulation can calculate its cosines later on.
            MV.SecondaryElectronFlag = 1
            SecondaryElectronIndex = MV.ElectronStorageTop


            # Store possible shell emissions auger or fluorescence, update the angles and cosines
            if EISTR > 30.0:
                # Auger Emission without fluorescence
                NAuger = <int> Object.NC0[GasIndex][I]
                EAuger = Object.EC0[GasIndex][I] / NAuger
                for J in range(NAuger):
                    # Register each electron, and its cosines
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
                    MV.IPlaneS[MV.ElectronStorageTop] = MV.IPlane

        # Generate scattering angles and update laboratory cosines after collision also update energy of electron
        IPT = <long long> Object.InteractionType[GasIndex][I]
        Object.CollisionsPerGasPerType[GasIndex][<int> IPT - 1] += 1
        MV.ID += 1
        MV.Iterator += 1

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
                        MV.XS[MV.ElectronStorageTop] = Object.X - log(RandomNum) * Object.PenningFraction[GasIndex][1][I] * Sign

                        RandomNum = random_uniform(RandomSeed)
                        if RandomNum < 0.5:
                            Sign = -1 * Sign
                        RandomNum = random_uniform(RandomSeed)
                        MV.YS[MV.ElectronStorageTop] = Object.Y - log(RandomNum) * Object.PenningFraction[GasIndex][1][I] * Sign

                        RandomNum = random_uniform(RandomSeed)
                        if RandomNum < 0.5:
                            Sign = -1 * Sign
                        RandomNum = random_uniform(RandomSeed)
                        MV.ZS[MV.ElectronStorageTop] = Object.Z - log(RandomNum) * Object.PenningFraction[GasIndex][1][I] * Sign

                    # Possible penning transfer time
                    PenningTransferTime = Object.TimeSum
                    if Object.PenningFraction[GasIndex][2][I] != 0.0:
                        RandomNum = random_uniform(RandomSeed)
                        PenningTransferTime = Object.TimeSum - log(RandomNum) * Object.PenningFraction[GasIndex][2][I]
                    MV.TS[MV.ElectronStorageTop] = PenningTransferTime
                    MV.ES[MV.ElectronStorageTop] = 1.0
                    MV.DirCosineX[MV.ElectronStorageTop] = MV.DirCosineX1
                    MV.DirCosineY[MV.ElectronStorageTop] = MV.DirCosineY1
                    MV.DirCosineZ[MV.ElectronStorageTop] = MV.DirCosineZ1
                    TempTime = 0.0
                    TempPlane = 0
                    # Last time before the penning transfer time
                    for J in range(Object.NumberOfTimeSteps):
                        TempTime += Object.TimeStep
                        if PenningTransferTime < TempTime:
                            break
                        TempPlane += 1

                    if PenningTransferTime >= Object.MaxTime:
                        # Penning happens after final time plane, dont store the electron
                        MV.ElectronStorageTop -= 1
                        MV.TotalNumberOfElectrons -= 1
                    else:
                        # Store the electron
                        MV.IPlaneS[MV.ElectronStorageTop] = TempPlane


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
            EI = COMEnergy - 0.0001

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
        #And go around again to the next collision!

        MV.I100 += 1
        if MV.I100 == 200:
            MV.DirCosineZ100 = MV.DirCosineZ1
            MV.DirCosineX100 = MV.DirCosineX1
            MV.DirCosineY100 = MV.DirCosineY1
            MV.Energy100 = MV.StartingEnergy
            MV.I100 = 0


    # Get the ratios of attachment and ionisation into the desired forms for the later functions
    if MV.NumberOfElectron > Object.IPrimary:
        Object.AttachmentOverIonisationPT = MV.NumberOfElectronAtt / (MV.NumberOfElectron - Object.IPrimary)
        Object.AttachmentErrPT = sqrt(MV.NumberOfElectronAtt) / MV.NumberOfElectronAtt
        Object.AttachmentOverIonisationErrPT = sqrt(MV.NumberOfElectron - Object.IPrimary) / (MV.NumberOfElectron - Object.IPrimary)
    else:
        Object.AttachmentOverIonisationPT = -1
        Object.AttachmentErrPT = sqrt(MV.NumberOfElectronAtt) / MV.NumberOfElectronAtt

    if ConsoleOuput:
        print(
            '\nSimulation through {} time planes:\n Total number of Electrons: {:10.1f}\n Number of Negative Ions: {:10.1f}\n Number of primaries: {:10.1f}\n'.format(
                Object.NumberOfTimeSteps,
                MV.NumberOfElectron,
                MV.NumberOfElectronAtt, Object.IPrimary))

    for i in range(6):
        free(TotalCollFreqIncludingNull[i])
    free(TotalCollFreqIncludingNull)
