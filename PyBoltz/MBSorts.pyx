from PyBoltz.Boltz cimport Boltz
import sys
import cython
cimport numpy as np
import  numpy as np

cdef long long MBSort(int I, double R2, int IE,Boltz Object):
    """
    This function selects collision type from collision array by binary step sampling reduces sampling range to within 4
    positions in array output =  i ( position within 4 of correct value).
    :param I:
    :param R2:
    :param IE:
    :param Object:
    :return: a new index
    """
    cdef long long iStep,Increment
    cdef int K
    iStep = long(Object.ISIZENT)
    Increment = 0

    for K in range(12):
        I = Increment
        if iStep == 2:
            if I==0:
                return I
            return I -1

        I = Increment + iStep
        if I <= Object.NumMomCrossSectionPointsNT:
            if Object.CollisionFrequencyNT[IE][I - 1] < R2:
                Increment = Increment + iStep
        iStep =  <long long>(iStep / 2)
    if I==0:
        return I
    return I - 1


cdef long long MBSortT(int GasIndex, int I, double R2, int IE,Boltz Object):
    """
    This function selects collision type from collision array by binary step sampling reduces sampling range to within 4
    positions in array output =  i ( position within 4 of correct value).
    :param I:
    :param R2:
    :param IE:
    :param Object:
    :return: a new index
    """

    cdef long long iStep,Increment
    cdef int K
    iStep = long(Object.ISIZE[GasIndex])

    Increment = 0
    for K in range(12):
        I = Increment
        if iStep == 2:
            if I==0:
                return I
            return I -1
        I = Increment + iStep
        if I <= Object.NumMomCrossSectionPoints[GasIndex]-1:
            if Object.CollisionFrequency[GasIndex][IE][I-1] < R2:
                Increment = Increment + iStep
        iStep = <long long>(iStep / 2)
    if I==0:
        return I
    return I -1
