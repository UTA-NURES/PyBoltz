import numpy as np
from PyBoltz.Boltz import Boltz

#Data type to store results with uncertainties
# Initialized with val and % error
class PBRes:
    '''Class used to store the output of PyBoltz with a val and error.'''
    err=0
    '''The error variable.'''
    val=0
    '''The value variable.'''
    def __init__(self, a=0,b=0):
        self.val=a
        self.err=b*a/100.
    def __str__(self):
        return "value: " + str(self.val)+ "; error: "+ str(self.err)

# PyBoltzRun helper class
class PyBoltzRun:
    '''Class used to be the wrapper object of PyBoltz.'''
    #Default settings for running PyBolz
    PBSettings   ={'Gases'                 :['NEON','CO2'],
                   'Fractions'             :[90,10],
                   'Max_collisions'        :4e7,
                   'EField_Vcm'            :100,
                   'Max_electron_energy'   :0,
                   'Temperature_C'         :23,
                   'Pressure_Torr'         :750.062,
                   'BField_Tesla'          :0,
                   'BField_angle'          :0,
                   'Angular_dist_model'    :1,
                   'Enable_penning'        :0,
                   'Enable_thermal_motion' :1,
                   'ConsoleOutputFlag'     :0,
                   'Decor_Colls'           :0,
                   'Decor_LookBacks'       :0,
                   'Decor_Step'            :0,
                   'NumSamples'            :10}
    '''Dictionary used to store the inputs/settings for the PyBoltz simulation.'''

    # Available Gases
    Gases = [
        [], ['CF4'], ['ARGON', 'AR'], ['HELIUM4', 'HE4'], ['HELIUM3', 'HE3'], ['NEON', 'NE'],
        ['KRYPTON', 'KR'], ['XENON', 'XE'], ['METHANE', 'CH4'], ['ETHANE', 'C2H6'], ['PROPANE', 'C3H8'],
        ['ISOBUTANE', 'C4H10'], ['CO2'], [], ['WATER', 'H2O'], ['OXYGEN', 'O2'], ['NITROGEN', 'N2'],
        [], [], [], [], ['HYDROGEN', 'H2'], ['DEUTERIUM', 'D2'], [], [], ['DME']
    ]
    '''Array of available gases in PyBoltz.'''

    # Print list of available gases
    def ListGases(self):
        '''Function used to print all the gas names in PyBoltz.'''
        for idx, gas in enumerate(self.Gases):
            if gas:
                print("{} {}".format(idx, gas))

    # Convert GasName into MagBoltz GasCode
    def GasCode(self, GasName):
        '''Function used to get the ID of the gas. The ID is simply the index of that gas in the array.'''
        for idx, names in enumerate(self.Gases):
            if GasName.upper() in names:
                return idx

    # Convert MagBoltz GasCode in GasName
    def GasName(self, Code):
        '''Function used to return the name(s) of the given Gas ID.'''
        return Gases[Code]

    # Load Input Dictionary into MagBoltz object
    def ProcessInputs(self,MBObject, Inputs):
        '''Function used to setup the PyBoltz Object with the given inputs in the PBSettings dictionary.'''
        for key in self.PBSettings.keys():
            if not key in Inputs:
                print("Input "+str(key)+ " not set, using default "+str(self.PBSettings[key]))
                Inputs[key]=self.PBSettings[key]
        if(len(Inputs['Gases'])!=len(Inputs['Fractions'])):
            print("Error! Gas and fraction lists not the same length")
            return False
        if(len(Inputs['Gases'])>6):
            print("Error! Too many gases. Max is 6.")
            return False
        if(abs(sum(Inputs['Fractions'])-100)>1e-6):
            print("Error! Gas fractions don't add to 100%")
            return False
        MBObject.EField=Inputs['EField_Vcm']
        MBObject.NumberOfGases=len(Inputs['Gases'])
        GasIDs=np.zeros(6,dtype='int')
        GasFractions=np.zeros(6,dtype='float')
        for i in range(len(Inputs['Gases'])):
            GasIDs[i] = self.GasCode(Inputs['Gases'][i])
            GasFractions[i]  = Inputs['Fractions'][i]
        MBObject.GasIDs  = GasIDs
        MBObject.GasFractions   = GasFractions
        MBObject.MaxNumberOfCollisions   = Inputs['Max_collisions']
        MBObject.Enable_Penning   = Inputs['Enable_penning']
        MBObject.Enable_Thermal_Motion  = Inputs['Enable_thermal_motion']
        MBObject.Max_Electron_Energy = Inputs['Max_electron_energy']
        MBObject.TemperatureCentigrade  = Inputs['Temperature_C']
        MBObject.Pressure_Torr   = Inputs['Pressure_Torr']
        MBObject.BField_Mag   = Inputs['BField_Tesla']
        MBObject.BField_Angle = Inputs['BField_angle']
        MBObject.Console_Output_Flag     = Inputs['ConsoleOutputFlag']
        MBObject.Which_Angular_Model = Inputs['Angular_dist_model']
        MBObject.Decor_Colls = Inputs['Decor_Colls']
        MBObject.Decor_Lookbacks = Inputs['Decor_LookBacks']
        MBObject.Decor_Step = Inputs['Decor_Step']
        MBObject.Num_Samples = Inputs['NumSamples']

        return True

    # Extract Outputs into Output Dictionary
    def ProcessOutputs(self, MBObject):
        '''Function used to fill the Outputs dictionary with the results from the PyBoltz object.'''
        Outputs={}
        Outputs['Drift_vel']      = PBRes(
            np.array([MBObject.VelocityX,MBObject.VelocityY,MBObject.VelocityZ]),
            np.array([MBObject.VelocityErrorX,MBObject.VelocityErrorY,MBObject.VelocityErrorZ]))
        Outputs['DT']             = PBRes(MBObject.TransverseDiffusion, MBObject.TransverseDiffusionError)
        Outputs['DL']             = PBRes(MBObject.LongitudinalDiffusion, MBObject.LongitudinalDiffusionError)
        Outputs['DT1']            = PBRes(MBObject.TransverseDiffusion1,MBObject.TransverseDiffusion1Error)
        Outputs['DL1']            = PBRes(MBObject.LongitudinalDiffusion1,MBObject.LongitudinalDiffusion1Error)

        Outputs['MeanEnergy']    = PBRes(MBObject.MeanElectronEnergy,MBObject.MeanElectronEnergyError)
        DTensor     = [[MBObject.DiffusionX, MBObject.DiffusionXY, MBObject.DiffusionXZ],
                       [MBObject.DiffusionXY, MBObject.DiffusionY, MBObject.DiffusionYZ],
                       [MBObject.DiffusionXZ, MBObject.DiffusionYZ, MBObject.DiffusionZ]]
        DTensorErr  = [[MBObject.ErrorDiffusionX, MBObject.ErrorDiffusionXY, MBObject.ErrorDiffusionXZ],
                       [MBObject.ErrorDiffusionXY, MBObject.ErrorDiffusionY, MBObject.ErrorDiffusionYZ],
                       [MBObject.ErrorDiffusionXZ, MBObject.ErrorDiffusionYZ, MBObject.ErrorDiffusionZ]]
        Outputs['DTensor']       = PBRes(np.array(DTensor), np.array(DTensorErr))
        Outputs['ReducedIonization'] =  PBRes(MBObject.ReducedIonization, MBObject.ReducedIonizationErr)
        Outputs['ReducedAttachment'] =  PBRes(MBObject.ReducedAttachment, MBObject.ReducedAttachmentErr)
        Outputs['AlphaSST']    =  PBRes(MBObject.AlphaSST, MBObject.AlphaSSTErr)

        return Outputs

    # Run PyBoltz with chosen settings
    def Run(self,MySettings):
        '''Function used to run the PyBoltz simulation. Note that the PBSettings dictionary needs to be set up.'''
        MBObject = Boltz()
        Status=self.ProcessInputs(MBObject,MySettings)
        if(Status):
            MBObject.Start()
        Outputs = self.ProcessOutputs(MBObject)
        return Outputs
