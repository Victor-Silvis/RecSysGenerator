from Modules.CTGAN import CTGAN
from Modules.MMPP import MMPP
from SimEnvironment.Simulation import Simulation
import time


# COMPILER SCRIPT
# -----------------------------------------------------------------------------------------------
# This script contains the compiler. The Compiler is to improve usability of the simulator.
# The compiler takes all the user set params set in the main.py script and uses those params
# to train all the modules of the simulator, and initiate the simulation.
# -----------------------------------------------------------------------------------------------


class Compiler:

    def __init__(self, DataFields=None, Userdata=None, Itemdata=None, Eventdata=None, CTGAN_params=None, CTGAN_Fit_Params=None,
                 MMPP_params=None, SimParams=None, database_info=None):
        
        #Get all params
        self.DataFields = DataFields
        self.Userdata = Userdata
        self.Itemdata = Itemdata
        self.Eventdata = Eventdata
        self.CTGAN_params = CTGAN_params
        self.CTGAN_Fit_Params = CTGAN_Fit_Params
        self.MMPP_params = MMPP_params
        self.SimParams = SimParams
        self.database_info = database_info

    def init_ctgan_users(self):
        #Init CTGAN and Fit for users
        disc_model = self.CTGAN_Fit_Params.get('trained_disc_user_model')
        gen_model = self.CTGAN_Fit_Params.get('trained_gen_user_model')

        self.ctgan_users = CTGAN(**self.CTGAN_params, type_data='user')
        self.user_train_data = self.Userdata.get('train_data')
        self.discrete_columns = self.Userdata.get('discrete_columns')
        self.userfield = self.DataFields.get('user_id')
        self.ctgan_users.fit(self.user_train_data, self.discrete_columns, id_column=self.userfield, trained_disc_model=disc_model, trained_gen_model=gen_model)

    def init_ctgan_items(self):
        #Init CTGAN and Fit for items
        disc_model = self.CTGAN_Fit_Params.get('trained_disc_item_model')
        gen_model = self.CTGAN_Fit_Params.get('trained_gen_item_model')

        self.ctgan_items = CTGAN(**self.CTGAN_params, type_data='item')
        self.item_train_data = self.Itemdata.get('train_data')
        self.itemfield = self.DataFields.get('item_id')
        self.discrete_columns_items = self.Itemdata.get('discrete_columns')
        self.ctgan_items.fit(self.item_train_data, self.discrete_columns_items, id_column=self.itemfield, trained_disc_model=disc_model, trained_gen_model=gen_model)


    def init_mmpp(self):
        n_states = self.MMPP_params.get('n_states')
        n_init = self.MMPP_params.get('kmeans_n_init')
        interval = self.MMPP_params.get('interval')
        transmatrix = self.MMPP_params.get('input_transition_matrix')
        arrivalrates = self.MMPP_params.get('input_arrival_rates')
        arrival_data = self.MMPP_params.get('train_data')
        self.mmpp = MMPP(n_states, n_init)

        if transmatrix is not None and arrivalrates is not None:
            print('MMPP fitting on Manual Input\n')
            self.mmpp.manual_input(transmatrix, arrivalrates)
        elif arrival_data is not None:
            print('MMPP fitting on Data\n')
            timestampfield = self.DataFields.get('timestamp')
            timestamps = arrival_data[timestampfield]
            self.mmpp.fit(timestamps, interval)
        else:
            raise ValueError('Either no data provided or transmatrix or arrival data missing')

    def init_simulation(self):
        self.event_train_data = self.Eventdata.get('train_data')
        continuous_colums = self.Userdata.get('continuous_columns')
        continuous_colums_items = self.Itemdata.get('continuous_columns')
        self.sim = Simulation(eventdata=self.event_train_data,
                              CTGAN_users= self.ctgan_users,
                              CTGAN_item=self.ctgan_items,
                              MMPP=self.mmpp,
                              DB=self.database_info,
                              sim_params=self.SimParams,
                              continuous_columns=continuous_colums,
                              discrete_columns=self.discrete_columns,
                              continuous_columns_items=continuous_colums_items,
                              discrete_columns_items=self.discrete_columns_items,
                              userdata=self.user_train_data,
                              itemdata=self.item_train_data,
                              DataFields=self.DataFields)

    def simulate(self):
        self.sim.simulate()

    def compile_and_run(self):
        print('User Model:')
        self.init_ctgan_users()
        print('CTGAN for new users Fitted\n')
        print('Item Model:')
        self.init_ctgan_items()
        print('CTGAN for new items Fitted\n')
        print('Fitting MMPP: ')
        self.init_mmpp()
        self.init_simulation()
        print('Simulation Params Set\n')
        print('Simulation Starting...\n')
        time.sleep(2)
        self.simulate()