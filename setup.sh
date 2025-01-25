#!/usr/bin/env bash
# setup the enviorment
temp="/PyBoltz/"
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/PyBoltz/:$PWD/PyBoltz/C:$PWD/PyBoltz/:$PWD/PyBoltz/MonteFuncs:$PWD/PyBoltz/Townsend:$PWD/PyBoltz/Townsend/CollisionFrequencyCalc:$PWD/PyBoltz/Townsend/Friedland:$PWD/PyBoltz/Townsend/PulsedTownsend:$PWD/PyBoltz/Townsend/Monte:$PWD/PyBoltz/Townsend/SteadyStateTownsend:$PWD/PyBoltz/Townsend/TimeOfFlight

export PATH=$PATH:$PWD
echo $PYTHONPATH

# build the code
python3 setup_build.py clean
export CFLAGS="-I /usr/local/lib/python3.7/site-packages/numpy/core/include $CFLAGS"
python3 setup_build.py build_ext --inplace


