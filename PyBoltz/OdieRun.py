import json
import time
import numpy as np
from PyBoltz.Boltz import Boltz
from PyBoltz.PyBoltzRun import PBRes

class OdieRun:
    '''Class to run PyBoltz and provide output for the Garfield++ package'''

    OdieSettings = {
        'Gases': ['ARGON', 'CH4'],
        'Fractions': [90.0, 10.0],
        'Max_collisions': 4e7,
        'EField_Vcm': 100,
        'Max_electron_energy': 0,
        'Temperature_C': 23,
        'Pressure_Torr': 750.062,
        'BField_Tesla': 0,
        'BField_angle': 0,
        'Angular_dist_model': 2,
        'Enable_penning': 0,
        'Enable_thermal_motion': 1,
        'ConsoleOutputFlag': 0,
        'Decor_Colls': 0,
        'Decor_LookBacks': 0,
        'Decor_Step': 0,
        'NumSamples': 10
    }

    Gases = [
        [], ['CF4'], ['ARGON', 'AR'], ['HELIUM4', 'HE4'], ['HELIUM3', 'HE3'], ['NEON', 'NE'],
        ['KRYPTON', 'KR'], ['XENON', 'XE'], ['METHANE', 'CH4'], ['ETHANE', 'C2H6'], ['PROPANE', 'C3H8'],
        ['ISOBUTANE', 'C4H10'], ['CO2'], [], ['WATER', 'H2O'], ['OXYGEN', 'O2'], ['NITROGEN', 'N2'],
        [], [], [], [], ['HYDROGEN', 'H2'], ['DEUTERIUM', 'D2'], [], [], ['DME']
    ]

    GridSettings = {
        'nE': 1,
        'minE': 100,
        'maxE': 100,
        'EFields': [],
        'nB': 1,
        'minB': 1,
        'maxB': 1,
        'BFields': [],
        'nA': 1,
        'minA': 0,
        'maxA': 0,
        'EBAngles': []
    }

    def ListGases(self):
        for idx, gas in enumerate(self.Gases):
            if gas:
                print("{} {}".format(idx, gas))

    def GasCode(self, GasName):
        for idx, names in enumerate(self.Gases):
            if GasName.upper() in names:
                return idx

    def GasName(self, Code):
        return Gases[Code]

    def ProcessInputs(self, MBObject, Inputs, PrintSettings=False):
        for key in self.OdieSettings.keys():
            if not key in Inputs:
                print("{} not set, using default value: {}".format(key, self.OdieSettings[key]))
                Inputs[key] = self.OdieSettings[key]

        for key in Inputs.keys():
            if not key in self.OdieSettings.keys():
                print("{} not a valid setting. Ignoring option.".format(key))

        if len(Inputs['Gases']) != len(Inputs['Fractions']):
            print(
                "Error: Gases and fractions lists are not the same length. Not running simulation."
            )
            return False

        if len(Inputs['Gases']) > 6:
            print("Error: Too many gases in mixture. Maximum is 6. Not running simulation.")
            return False

        if abs(sum(Inputs['Fractions']) - 100.0) > 1e-6:
            print("Error: Gas fractions do not add up to 100%. Not running simulation.")
            return False

        MBObject.EField = Inputs['EField_Vcm']
        MBObject.NumberOfGases = len(Inputs['Gases'])
        GasIDs = np.zeros(6, dtype='int')
        GasFractions = np.zeros(6, dtype='float')
        for i in range(len(Inputs['Gases'])):
            GasIDs[i] = self.GasCode(Inputs['Gases'][i])
            GasFractions[i] = Inputs['Fractions'][i]
        MBObject.GasIDs = GasIDs
        MBObject.GasFractions = GasFractions
        MBObject.MaxNumberOfCollisions = Inputs['Max_collisions']
        MBObject.Enable_Penning = Inputs['Enable_penning']
        MBObject.Enable_Thermal_Motion = Inputs['Enable_thermal_motion']
        MBObject.Max_Electron_Energy = Inputs['Max_electron_energy']
        MBObject.TemperatureCentigrade = Inputs['Temperature_C']
        MBObject.Pressure_Torr = Inputs['Pressure_Torr']
        MBObject.BField_Mag = Inputs['BField_Tesla']
        MBObject.BField_Angle = Inputs['BField_angle']
        MBObject.Console_Output_Flag = Inputs['ConsoleOutputFlag']
        MBObject.Which_Angular_Model = Inputs['Angular_dist_model']
        MBObject.Decor_Colls = Inputs['Decor_Colls']
        MBObject.Decor_Lookbacks = Inputs['Decor_LookBacks']
        MBObject.Decor_Step = Inputs['Decor_Step']
        MBObject.Num_Samples = Inputs['NumSamples']

        if PrintSettings:
            print("Simulation settings...")
            print(json.dumps(Inputs, indent=4))

        return True

    def ProcessOutputs(self, MBObject):
        Outputs = {}
        Outputs['Drift_vel'] = PBRes(
            np.array([MBObject.VelocityX, MBObject.VelocityY, MBObject.VelocityZ]),
            np.array([MBObject.VelocityErrorX, MBObject.VelocityErrorY, MBObject.VelocityErrorZ])
        )
        Outputs['DT'] = PBRes(MBObject.TransverseDiffusion, MBObject.TransverseDiffusionError)
        Outputs['DL'] = PBRes(MBObject.LongitudinalDiffusion, MBObject.LongitudinalDiffusionError)
        Outputs['DT1'] = PBRes(MBObject.TransverseDiffusion1, MBObject.TransverseDiffusion1Error)
        Outputs['DL1'] = PBRes(
            MBObject.LongitudinalDiffusion1, MBObject.LongitudinalDiffusion1Error
        )

        Outputs['MeanEnergy'] = PBRes(MBObject.MeanElectronEnergy, MBObject.MeanElectronEnergyError)
        DTensor = [
            [MBObject.DiffusionX, MBObject.DiffusionXY, MBObject.DiffusionXZ],
            [MBObject.DiffusionXY, MBObject.DiffusionY, MBObject.DiffusionYZ],
            [MBObject.DiffusionXZ, MBObject.DiffusionYZ, MBObject.DiffusionZ]
        ]
        DTensorErr = [
            [MBObject.ErrorDiffusionX, MBObject.ErrorDiffusionXY, MBObject.ErrorDiffusionXZ],
            [MBObject.ErrorDiffusionXY, MBObject.ErrorDiffusionY, MBObject.ErrorDiffusionYZ],
            [MBObject.ErrorDiffusionXZ, MBObject.ErrorDiffusionYZ, MBObject.ErrorDiffusionZ]
        ]
        Outputs['DTensor'] = PBRes(np.array(DTensor), np.array(DTensorErr))

        Outputs['AttachmentRate'] = PBRes(MBObject.AttachmentRate, MBObject.AttachmentRateError)
        Outputs['IonisationRate'] = PBRes(MBObject.IonisationRate, MBObject.IonisationRateError)
        Outputs['AttachmentSST'] = PBRes(MBObject.AttachmentSST, MBObject.AttachmentSSTErr)
        Outputs['IonisationSST'] = PBRes(MBObject.AlphaSST, MBObject.AlphaSSTErr)
        Outputs['Crossed_SST'] = MBObject.Crossed_SST

        lor_angle, lor_error = self.CalcLorentzAngle(MBObject)
        Outputs['LorentzAngle'] = PBRes(lor_angle, lor_error)

        return Outputs

    def CalcLorentzAngle(self, MBObject):

        vx = MBObject.VelocityX
        vy = MBObject.VelocityY
        vz = MBObject.VelocityZ

        vt = np.sqrt(vx * vx + vy * vy)
        v2 = vx * vx + vy * vy + vz * vz

        lor_angle = np.arctan2(vt, vz)
        lor_error = 0.0
        if vt > 0.0 and v2 > 0.0 and abs(lor_angle) > 0.0:
            dvx = vx * MBObject.VelocityErrorX
            dvy = vy * MBObject.VelocityErrorY
            dvz = vz * MBObject.VelocityErrorZ
            a = vz / vt
            lor_error = np.sqrt(
                a * a * (vx * vx * dvx * dvx + vy * vy * dvy * dvy) + vt * vt * dvz * dvz
            ) / v2
            lor_error = lor_error / lor_angle

        return (lor_angle, lor_error)

    def Run(self, InputSettings=None, JSONFileName=None, PrintSettings=False):
        MBObject = Boltz()

        if JSONFileName:
            with open(JSONFileName, 'r') as File:
                print("Reading {} ...".format(JSONFileName))
                JSONSettings = json.load(File)
                json.dumps(JSONSettings, indent=4)
            Status = self.ProcessInputs(MBObject, JSONSettings, PrintSettings)
        elif InputSettings:
            Status = self.ProcessInputs(MBObject, InputSettings, PrintSettings)
        else:
            print("Using stored inputs.")
            Status = self.ProcessInputs(MBObject, self.OdieSettings, PrintSettings)

        if Status:
            MBObject.Start()
        else:
            print("Error detected. Not running simulation.")

        return self.ProcessOutputs(MBObject)

    def LoadSettings(self, InputSettings=None, JSONFileName=None, PrintSettings=False):

        if JSONFileName:
            with open(JSONFileName, 'r') as File:
                print("Reading {} ...".format(JSONFileName))
                NewSettings = json.load(File)
        elif InputSettings:
            NewSettings = InputSettings
        else:
            print("Error. No inputs provided.")
            return False

        for key in self.OdieSettings.keys():
            if not key in NewSettings:
                print("{} not set, using default value: {}".format(key, self.OdieSettings[key]))
                NewSettings[key] = self.OdieSettings[key]

        for key in NewSettings.keys():
            if not key in self.OdieSettings.keys():
                print("{} not a valid setting. Ignoring option.".format(key))

        if len(NewSettings['Gases']) != len(NewSettings['Fractions']):
            print(
                "Error: Gases and fractions lists are not the same length. Not running simulation."
            )
            return False

        if len(NewSettings['Gases']) > 6:
            print("Error: Too many gases in mixture. Maximum is 6. Not running simulation.")
            return False

        if abs(sum(NewSettings['Fractions']) - 100.0) > 1e-6:
            print("Error: Gas fractions do not add up to 100%. Not running simulation.")
            return False

        self.OdieSettings = NewSettings

        if PrintSettings:
            print(self.OdieSettings)

        print("Loaded new simulation settings.")
        return True

    def GenerateGasGrid(
        self, minE, maxE, nE, LogScale=False, minB=0, maxB=0, nB=1, minA=0, maxA=0, nA=1
    ):
        """
        Automate the process of generating the properties of a given gas mixture at a grid
        of EFields, BFields, and E-B Angles.
        """

        #This format of generating the grid is the same as Garfield++.
        if LogScale:
            EFields = np.geomspace(minE, maxE, nE)
        else:
            EFields = np.linspace(minE, maxE, nE)

        BFields = np.linspace(minB, maxB, nB)
        EBAngles = np.linspace(minA, maxA, nA)

        #Store grid settings for formatting the file output
        self.GridSettings['nE'] = nE
        self.GridSettings['minE'] = minE
        self.GridSettings['maxE'] = maxE
        self.GridSettings['EFields'] = EFields
        self.GridSettings['nB'] = nB
        self.GridSettings['minB'] = minB
        self.GridSettings['maxB'] = maxB
        self.GridSettings['BFields'] = BFields
        self.GridSettings['nA'] = nA
        self.GridSettings['minA'] = minA
        self.GridSettings['maxA'] = maxA
        self.GridSettings['EBAngles'] = EBAngles

        GridOutput = {}
        for E in EFields:
            for B in BFields:
                for A in EBAngles:
                    self.OdieSettings['EField_Vcm'] = E
                    self.OdieSettings['BField_Tesla'] = B
                    self.OdieSettings['BField_angle'] = A

                    print("-------------------------------")
                    print("Running E={} V/cm, B={} T, A={}".format(E, B, A))
                    GridOutput[(E, B, A)] = self.Run(PrintSettings=False)

        print("Finished generating gas grid.")
        return GridOutput

    def WriteGasFile(self, FileName, GridOutput):
        """
        Function to write the output from GenerateGasGrid() in the format defined
        by the Garfield++ simulation package. The output file is a text file containing
        the simulation parameters of the grid and the gas properties at each grid point.

        The output file matches the Garfield++ format as closely as possible with the
        exception of some whitespace between values. A number of output lines are included
        just to match the Garfield++ output even though PyBoltz does not produce them.

        The function needs two required parameters:
            FileName: name of the output gas file, e.g. ar_90_ch4_10.gas
            GridOutput: a dictionary of the gas properties defined by the ProcessOuputs
            function indexed by a tuple containing the EField, E-B Angle, and BField used.
            The GenerateGasGrid() returns this object in the correct format, but this
            could be constructed using other functions.

        NB: Garfield and Garfield++ are used interchangably, but this output is specifically
        for Garfield++.
        """
        print("Writing gas properties to {}".format(FileName))

        #MagBoltz defines a list of 60 available gases and Garfield++ expects a table
        #with 60 entries defining the mixture used. The values are listed as percents.
        MixtureList = [0 for i in range(60)]
        MixtureStr = ""
        for gas, frac in zip(self.OdieSettings['Gases'], self.OdieSettings['Fractions']):
            MixtureList[self.GasCode(gas) - 1] = frac
            MixtureStr += "{} {}%, ".format(gas, frac)

        Temperature_K = self.OdieSettings['Temperature_C'] + 273.15
        Pressure_ATM = self.OdieSettings['Pressure_Torr'] / 760

        #Garfield (currently) expects the version to be 10, 11, or 12 when reading this file.
        Version = "12"

        #Garfield makes a difference between a 1D or 2D table when generating with more than
        #one BField or E-B angle. Define a flag for later table formatting.
        Tab2D = True if self.GridSettings['nA'] > 1 or self.GridSettings['nB'] > 1 else False
        Tab2DStr = "T" if Tab2D else "F"

        #The GasBits define which values are included in this file (e.g. the drift velocity).
        #This string is hardcoded based on including all the standard PyBoltz output, along
        #with ignoring a few values which are skipped in Garfield++.
        GasBits = "TFTTFTTTTTTFFFFFFFFF"

        #Store the creation time of the file. Garfield++ does not parse the date/time,
        #so this can be in any format.
        DateStr = time.strftime("%Y/%m/%d at %H:%M:%S %z", time.localtime())

        #Including these to simply match the style of the Garfield++ output, and to
        #indicate that this file was created using PyBoltz.
        MemberName = "< PyBoltz Ouput >"
        HeaderStr = "*----.----1----.----2----.----3----.----4----.----5----.----6----.----7----.----8----.----9----.---10----.---11----.---12----.---13--\n"
        IdentifierStr = MixtureStr + "T={} K, ".format(Temperature_K
                                                       ) + "p={} atm".format(Pressure_ATM)

        #Open file and start writing the formatted output. The file uses % for comment lines
        #which are found in the header (before the gas table). They are not valid after the
        #header.
        File = open(FileName, 'w')
        File.write(HeaderStr)
        File.write("% Created {} {} GAS\n".format(DateStr, MemberName))
        File.write("% PyBoltz to Garfield++ converter, Odie, written by Andrew.Cudd@colorado.edu\n")
        File.write(" Version   : {}\n".format(Version))
        File.write(" GASOK bits: {}\n".format(GasBits))
        File.write(" Identifier: {}\n".format(IdentifierStr))
        File.write(" Clusters  : \n")

        #The dimensions are for the grid points included in the gas tables. The dimensions
        #are EFields, E-B Angles, BFields, Excitiation, and Ionisation levels. The last two are
        #set to zero as PyBoltz does not include them in the output.
        File.write(
            " Dimension : {} {:8d} {:8d} {:8d} {:8d} {:8d}\n".format(
                Tab2DStr, self.GridSettings['nE'], self.GridSettings['nA'], self.GridSettings['nB'],
                0, 0
            )
        )

        #The EFields, E-B Angles, BFields, and Gas Mixture tables are all no more
        #than five columns wide.
        #List EFields used as the reduced EFields: E/P (where the pressure is in torr)
        File.write(" E fields \n")
        for idx, e in enumerate(self.GridSettings['EFields']):
            reduced_e = e / self.OdieSettings['Pressure_Torr']
            File.write(" {:.8E}".format(reduced_e))
            if (idx + 1) % 5 == 0:
                File.write("\n")
        File.write("\n")

        #List E-B angles used in radians.
        File.write(" E-B angles \n")
        for idx, a in enumerate(self.GridSettings['EBAngles']):
            a_radians = np.radians(a)
            File.write(" {:.8E}".format(a_radians))
            if (idx + 1) % 5 == 0:
                File.write("\n")
        File.write("\n")

        #List the BFields used with factor of 100 to match the Garfield format.
        File.write(" B fields \n")
        for idx, b in enumerate(self.GridSettings['BFields']):
            b_scaled = b * 100
            File.write(" {:.8E}".format(b_scaled))
            if (idx + 1) % 5 == 0:
                File.write("\n")
        File.write("\n")

        #List the gas mixture of a table containing all possible gases. Each
        #fraction is listed as a percent e.g. 50.0 or 75.0
        File.write(" Mixture: \n")
        for idx, frac in enumerate(MixtureList):
            File.write(" {:.8E}".format(frac))
            if (idx + 1) % 5 == 0:
                File.write("\n")

        SQRTP = np.sqrt(self.OdieSettings['Pressure_Torr'])
        LOGP = np.log(self.OdieSettings['Pressure_Torr'])
        ROWS = 0

        #Write the gas tables containing the gas properties for each grid point.
        #Note that Garfield++ explicitly looks for this exact string to start
        #parsing the gas tables.
        File.write(" The gas tables follow:\n")
        for e in self.GridSettings['EFields']:
            for a in self.GridSettings['EBAngles']:
                for b in self.GridSettings['BFields']:
                    Output = GridOutput[(e, b, a)]
                    TableVal = []

                    #In the Garfield++ code the variables are named corresponding to the
                    #the drift along E, B, and ExB, which are oriented in Z, X, and Y.
                    #Convert from mm/us to cm/us
                    vz = Output['Drift_vel'].val[2] / 10.0
                    vx = Output['Drift_vel'].val[0] / 10.0
                    vy = Output['Drift_vel'].val[1] / 10.0

                    if Tab2D:
                        TableVal.extend([vz, vx, vy])
                    else:
                        TableVal.extend([vz, 0.0, vx, 0.0, vy, 0.0])

                    #Match scaling that Garfield++ expects.
                    dl = Output['DL1'].val * SQRTP * 1E-4
                    dt = Output['DT1'].val * SQRTP * 1E-4

                    #Yes, alpha and alpha0 are supposed to be the same value.
                    alpha  = Output['IonisationRate'].val
                    alpha0 = Output['IonisationRate'].val
                    eta    = Output['AttachmentRate'].val

                    #If the steady state threshold was crossed, use the SST calculation of the ionisation
                    #and attachment coefficients as the default values.
                    if Output['Crossed_SST'] is True:
                        alpha  = Output['IonisationSST'].val
                        alpha0 = Output['IonisationSST'].val
                        eta    = Output['AttachmentSST'].val

                    #If the coefficients are zero, set to -30 as a sufficiently small power; log(-30) is basically zero.
                    #Otherwise store the logarithm of the reduced coefficients.
                    alpha = np.log(alpha) - LOGP if alpha > 0 else -30 - LOGP
                    alpha0 = np.log(alpha0) - LOGP if alpha0 > 0 else -30 - LOGP
                    eta = np.log(eta) - LOGP if eta > 0 else -30 - LOGP

                    if Tab2D:
                        TableVal.extend([dl, dt, alpha, alpha0, eta])
                    else:
                        TableVal.extend([dl, 0.0, dt, 0.0, alpha, 0.0, alpha0, eta, 0.0])

                    lor = Output['LorentzAngle'].val
                    #Ion mobility. Placeholder of zero since it is not included.
                    mu = 0.0
                    #Dissociation coefficient. Set to -30 as a placeholder since it is not included,
                    #and this value is stored as a log, so it can't be zero.
                    diss = -30

                    if Tab2D:
                        TableVal.extend([mu, lor, diss])
                    else:
                        TableVal.extend([mu, 0.0, lor, 0.0, diss, 0.0])

                    #Store diffusion tensor components with scaling and ordering to match the
                    #Garfield++ format.
                    difxx = Output['DTensor'].val[0][0] * self.OdieSettings['Pressure_Torr'
                                                                            ] * 0.2E-4 / (vz / 1E3)
                    difyy = Output['DTensor'].val[1][1] * self.OdieSettings['Pressure_Torr'
                                                                            ] * 0.2E-4 / (vz / 1E3)
                    difzz = Output['DTensor'].val[2][2] * self.OdieSettings['Pressure_Torr'
                                                                            ] * 0.2E-4 / (vz / 1E3)
                    difxz = Output['DTensor'].val[0][2] * self.OdieSettings['Pressure_Torr'
                                                                            ] * 0.2E-4 / (vz / 1E3)
                    difyz = Output['DTensor'].val[1][2] * self.OdieSettings['Pressure_Torr'
                                                                            ] * 0.2E-4 / (vz / 1E3)
                    difxy = Output['DTensor'].val[0][1] * self.OdieSettings['Pressure_Torr'
                                                                            ] * 0.2E-4 / (vz / 1E3)

                    if Tab2D:
                        TableVal.extend([difzz, difxx, difyy, difxz, difyz, difxy])
                    else:
                        TableVal.extend(
                            [
                                difzz, 0.0, difxx, 0.0, difyy, 0.0, difxz, 0.0, difyz, 0.0, difxy,
                                0.0
                            ]
                        )

                    #Write values in a table which contains at most eight columns.
                    for val in TableVal:
                        File.write(" {: .8E}".format(val))
                        if (ROWS + 1) % 8 == 0:
                            File.write("\n")
                        ROWS += 1

                if ROWS % 8 != 0:
                    File.write("\n")
                ROWS = 0

        #Extrapolation methods. This is Garfield specific information and is hardcoded to match how Garfield sets the values.
        if not Tab2D:
            ExtrapolateHigh = [1 for i in range(13)]
            ExtrapolateLow = [0 for i in range(13)]

            File.write(" H Extr: ")
            for flag in ExtrapolateHigh:
                File.write("{:4d} ".format(flag))
            File.write("\n")

            File.write(" L Extr: ")
            for flag in ExtrapolateLow:
                File.write("{:4d} ".format(flag))
            File.write("\n")

        File.write(" Thresholds: {:4d} {:4d} {:4d}\n".format(1, 1, 1))

        #Interpolation methods. This is Garfield specific information and is hardcoded to match how Garfield sets the values.
        InterpolationFlags = [2 for i in range(13)]
        File.write(" Interp: ")
        for flag in InterpolationFlags:
            File.write("{:4d} ".format(flag))
        File.write("\n")

        #More Garfield++ specific information included to simply match the output.
        File.write(
            " A     = {:.8E}, Z   = {:.8E}, EMPROB= {:.8E}, EPAIR = {:.8E}\n".format(
                0.0, 0.0, 0.0, 0.0
            )
        )
        File.write(" Ion diffusion: {:.8E} {:.8E}\n".format(0.0, 0.0))
        File.write(
            " CMEAN = {:.8E}, RHO = {:.8E}, PGAS  = {:.8E}, TGAS  = {:.8E}\n".format(
                0.0, 0.0, self.OdieSettings['Pressure_Torr'], Temperature_K
            )
        )
        File.write(" CLSTYP   : NOT SET\n")
        File.write(" FCNCLS   : \n")
        File.write(" NCLS     : {:d}\n".format(0))
        File.write(" Average  : {:.18E}\n".format(0.0))
        File.write("  Heed initialisation done: F\n")
        File.write("  SRIM initialisation done: F\n")
        File.close()

        print("Finished writing data to file.")
        return True
