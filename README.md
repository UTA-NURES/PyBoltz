![PyBoltz_Tests](https://github.com/UTA-REST/PyBoltz/workflows/PyBoltz_Tests/badge.svg)
# PyBoltz
This software package is a translation of the Fortran based Magboltz code (Biagi, 2001) into Cython. This project was built to allow for more productive work to be done with magboltz.

## General information.

### About PyBoltz ###

PyBoltz is an electron swarm Monte Carlo package for calcualting transport parameters of electrons in different gas mixtures that are relevant to particle detectors.

To cite PyBoltz, and for more information:

```
Electron transport in gaseous detectors with a Python-based Monte Carlo simulation code
B. Al Atoum et al, Comp Phys Comm 254 (2020) 107357
https://www.sciencedirect.com/science/article/abs/pii/S0010465520301533
```

### About Magboltz.
The Magboltz program computes drift gas properties by "numerically integrating the Boltzmann transport equation"-- i.e., simulating an electron bouncing around inside a gas. By tracking how far the virtual electron propagates, the program can compute the drift velocity. By including a magnetic field, the program can also calculate the Lorentz angle. [Read more](http://cyclo.mit.edu/drift/www/aboutMagboltz.html).

### Why Cython?
Cython's static typing improves the speed of python code by about a hundred times. In other words, Cython provides us with the simplicity of python and the speed of Fortran/C. [Read more](https://cython.org/).

## Setting up and running instructions. 

### Setting up.
To be able to run this project you will need python3+, cython, and numpy installed. Our setup has python 3.6.7, Cython 0.29.3, and numpy 1.16.1. 

### First method.
For simple running purposes, one can simply install PyBoltz using the following command. **Note: This method needs a new version of PIP.**

```
python3 -m pip install --upgrade  git+https://github.com/UTA-REST/PyBoltz.git --user
```

### Second method.
For developing purposes, follow the steps below.

#### Cloning.
Start off by simply cloning this repository. 

#### Gasmix module.
Before building the code make sure to run the following command to install the PyGasMix module.

```
$ sudo pip3 install --upgrade  git+https://github.com/UTA-REST/PyGasMix --user
```

**Notice** If you are planning to develop upon the Gasmix module as well, you should be installing the PyGasMix module by following the steps in the repository link below. (steps detailed in the readme).
[PyGasMix](https://github.com/UTA-REST/GasMix).


#### Building.
Finally, to build the code, run the following command. This should compile all of the Cython files and add the path of the repository directory to your PYTHONPATH so you can access the libraries from anywhere. This will take a few minutes the first time.

```
$ source setup.sh
```

Please note that you might need to change the commands inside the setup.sh file to match your python version.

### Running PyBoltz.
To run the code, you will need to import PyBoltz and instantiate an instance of the PyBoltz object, fill in the input parameters and call the PyBoltz.Start() function. There are examples in the Examples directory on to how to use PyBoltz. The main example is the Test_PyBoltz_NoWrapper.py code. This example also has a list of the gases in PyBoltz.

#### Input parameters.
* **PyBoltz.NumberOfGases** - The number of gases in the mixture (goes up to 6).
* **PyBoltz.MaxNumberOfCollisions** - The number of simulated events / 2*10E7.
* **PyBoltz.Enable_Penning** - Penning effects included (0 or 1).
* **PyBoltz.Enable_Thermal_Motion** - Thermal motion included (0 or 1).
* **PyBoltz.Max_Electron_Energy** - Upper limit of electron energy integration (0.0 to automatically calculate this value).
* **PyBoltz.GasIDs** - Array of six elements that has the number of each gas in the mixture.
* **PyBoltz.GasFractions** - Array of six elements that has the percentage of each gas in the mixture.
* **PyBoltz.TemperatureCentigrade** - The tempreture in degrees centigrade.
* **PyBoltz.Pressure_Torr** - The pressure \[torr\].
* **PyBoltz.EField** - The electric field in the chamber \[Volts/Cm\].
* **PyBoltz.BField_Mag** - The magnitude of the magentic field \[KiloGauss\].
* **PyBoltz.BField_Angle** - The angle between the magentic field and the electric field. 
* **PyBoltz.Which_Angular_Model** - This variable is used to fix the angular distrubtions to one of the following types. 
  - Okhrimvoskky Type - PyBoltz.WhichAngularModel = 2 (default value).
  - Capitelli Longo Type - PyBoltz.WhichAngularModel = 1.
  - Isotropic Scattering - PyBoltz.WhichAngularModel = 0.
* **PyBoltz.Console_Output_Flag** - This variable is used to tell PyBoltz to print to the console.
  - Print to the console - PyBoltz.ConsoleOutputFlag = 1.
  - Avoid printing to the console - PyBoltz.ConsoleOutputFlag = 0.
* **PyBoltz.Random_Seed** - This variable is used to set the seed for the random number geenerator used by the simulation. 

#### Output parameters.
Please note that the following are only the main output parameters. One can still get any value from the parameters within the Magboltz class.

* **PyBoltz.VelocityZ** - Drift velocity in the Z direction \[mm/mus\].
* **PyBoltz.VelocityY** - Drift velocity in the Y direction \[mm/mus\].
* **PyBoltz.VelocityX** - Drift velocity in the X direction \[mm/mus\].
* **PyBoltz.VelocityErrorZ** - Error for the Magboltz.WZ value (+- Magboltz.DWZ * Magboltz.WZ).
* **PyBoltz.VelocityErrorY** - Error for the Magboltz.WY value (+- Magboltz.DWY * Magboltz.WY).
* **PyBoltz.VelocityErrorX** - Error for the Magboltz.WX value (+- Magboltz.DWX * Magboltz.WX).
* **PyBoltz.TransverseDiffusion** - Transverse diffusion \[cm^2/s\].
* **PyBoltz.TransverseDiffusionError** - Error for the Magboltz.DIFTR value (+- Magboltz.DFTER * Magboltz.DIFTR).
* **PyBoltz.LongitudinalDiffusion** - Longitudinal diffusion \[cm^2/s\]..
* **PyBoltz.LongitudinalDiffusionError** - Error for the Magboltz.DIFLN value (+- Magboltz.DFLER * Magboltz.DIFLN).
* **PyBoltz.TransverseDiffusion1** - Transverse diffusion \[mum/cm^0.5\].
* **PyBoltz.TransverseDiffusion1Error** - Error for the Magboltz.DTMN value (+- Magboltz.DTMN * Magboltz.DFTER1).
* **PyBoltz.LongitudinalDiffusion1** - Longitudinal diffusion \[mum/cm^0.5\].
* **PyBoltz.LongitudinalDiffusion1Error** - Error for the Magboltz.DLMN vlaue (+- Magboltz.DLMN * Magboltz.DFLER1).
* **PyBoltz.MeanElectronEnergy** - Mean electron energy \[eV\].
* **PyBoltz.MeanElectronEnergyError** - Error for the Magboltz.AVE value (+- Magboltz.AVE * Magboltz.DEN).
* **PyBoltz.DiffusionX** - Diffusion in the X plane \[cm^2/s\].
* **PyBoltz.ErrorDiffusionX** - Error for the Magboltz.DIFXX value (+- Magboltz.DIFXX * Magboltz.DXXER).
* **PyBoltz.DiffusionY** - Diffusion in the Y plane \[cm^2/s\].
* **PyBoltz.ErrorDiffusionY** - Error for the Magboltz.DIFYY value (+- Magboltz.DIFYY * Magboltz.DYYER).
* **PyBoltz.DiffusionZ** - Diffusion in the Z plane \[cm^2/s\].
* **PyBoltz.ErrorDiffusionZ** - Error for the Magboltz.DIFZZ value (+- Magboltz.DIFZZ * Magboltz.DZZER).
* **PyBoltz.DiffusionYZ** - Diffusion in the YZ plane \[cm^2/s\].
* **PyBoltz.ErrorDiffusionYZ** - Error for the Magboltz.DIFYZ value (+- Magboltz.DIFYZ * Magboltz.DYZER).
* **PyBoltz.DiffusionXY** - Diffusion in the XY plane \[cm^2/s\].
* **PyBoltz.ErrorDiffusionXY** - Error for the Magboltz.DIFXY value (+- Magboltz.DIFXY * Magboltz.DXYER).
* **PyBoltz.DiffusionXZ** - Diffusion in the XZ plane \[cm^2/s\].
* **PyBoltz.ErrorDiffusionXZ** - Error for the Magboltz.DIFXZ value (+- Magboltz.DIFXZ * Magboltz.DXZER).
* **PyBoltz.MeanCollisionTime** - Mean Collision Time.
* **TOF Outputs** - Those outputs include townsend coeffiecents, diffusion and energy values. Those outputs are calculated from the time of flight simulation. Check the PyBoltz object documentation for more details.
* **SST Outputs** - Those outputs include townsend coeffiecents, diffusion and energy values. Those outputs are calculated from the steady state simulation. Check the PyBoltz object documentation for more details.
* **Collision type counters** - Six elements arraies that houses the number of collisions of each gas for each types. The types are elastic, inelastic, super-elastic, ionisation, and attachment. Check the PyBoltz object documentaion for more details.

#### Compilation issues.
This sections is written here to help troubleshoot compilation issues. The following are links to the two main issues:

* [First Issue](https://github.com/UTA-REST/PyBoltz/issues/1).
* [Second Issue](https://github.com/UTA-REST/PyBoltz/issues/2).

## Gas list.
The current PyBoltz version has the following gases. Please note that the number of the gas is used as an indicator to that gas in the code. 

* **CF4** - Gas # 1.
* **Argon** Gas # 2.
* **Helium-4** Gas # 3.
* **Helium-3** Gas # 4.
* **Neon** Gas # 5.
* **Krypton** Gas #6.
* **Xenon** Gas # 7.
* **CH4** Gas # 8.
* **Ethane** Gas # 9.
* **Propane** Gas # 10.
* **Isobutane** Gas # 11.
* **CO2** Gas # 12.
* **H2O** Gas # 14.
* **Oxygen** Gas # 15.
* **Nitrogen** Gas # 16.
* **Hydrogen** Gas # 21.
* **Deuterium** Gas # 22.
* **DME** Gas # 25.
* **XenonMert** Gas # 61 (This gas requires extra parameters, check /Examples/Test_PyBoltz_mert.py).

## Testing
To be able to run the tests for this module, you will need to have pytest installed on your machine. Also, you need to run the following to install the testing data package. 
```
$ sudo pip3 install --upgrade  git+https://github.com/UTA-REST/PyBoltz_Test_Data --user
```


After doing so, go to the test directory and run the following. 
```
$ pytest
```
This will run all the tests. If you are intrested in a single test, add the name of the testing python file to the end of the above command. 

For more information on the testing data package, check the following repository. 
[Testing Data Package](https://github.com/UTA-REST/PyBoltz_Test_Data).


## Documentaion link
[Documentaion...](https://uta-rest.github.io/PyBoltz-Documentation/html/).
