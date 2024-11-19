

# MAIN GENERATOR/SIMULATOR SCRIPT
# -----------------------------------------------------------------------------------------------
# This script controls the simulation. Please set your parameters below, from which the
# simulator can be compiled. Make sure your server is running and then run main.py to
# start the simulator with the set parameters
# -----------------------------------------------------------------------------------------------

# Packages
import pandas as pd
import numpy as np

# Import Compiler Script
from Modules.Compiler import Compiler



## -------------------------------- GENERAL INFO/PARAMETERS --------------------------------------

DataFields = {
'user_id'               : 'visitorid',      # Name of user id column
'item_id'               : 'itemid',         # Name of item id column
'event'                 : 'event',          # Name of value/event/rating column
'timestamp'             : 'timestamp'       # Name of time column (in seconds)
}

Userdata = {
'train_data'            : None,             # Name of train user data dataframe
'discrete_columns'      : None,             # List of discrete features in user dataframe
'continuous_columns'    : None              # List of continuous features in user dataframe
}

Itemdata = {    
'train_data'            : None,             # Name of train item data dataframe
'discrete_columns'      : None,             # List of discrete features in item dataframe
'continuous_columns'    : None              # list of continuous features in item dataframe
}

Eventdata = {
'train_data'            : None              # Name of dataframe containing training event data
}


## -------------------------------- CTGAN (GENERATOR) PARAMETERS -------------------------------------------------

CTGAN_params = {
'epochs'                : 5,                # Number of Epochs for training
'batch_size'            : 500,              # Batchsize (always multiple of 10)
'noise_dim'             : 128,              # Dimension of latent space, (higher could help capture complex patterns in data)
'generator_dim'         : (256, 256),       # Dimensions and layers of generator NN model, e.g. (x, x, x) = three layers
'discriminator_dim'     : (256, 256),       # Dimensions and layers of Discriminator NN model, e.g. (x, x) = two layers
'generator_lr'          : 2e-4,             # Generator NN model learning rate
'generator_decay'       : 1e-6,             # Generator Weight for optimizer
'discriminator_lr'      : 2e-4,             # Discriminator learning rate
'discriminator_decay'   : 1e-6,             # Discriminator Weight for optimizer
'discriminator_steps'   : 1,                # Number of discriminator steps for each generator update, 5 could be result in better outcomes https://arxiv.org/abs/1701.07875
'pac'                   : 10,               # Number of samples to group together when applying discriminator
'log_frequency'         : True,             # Whether to use log frequency of categorical levels in conditinal sampling
'verbose'               : True,             # Wheter to have print statements for training progress results
'auto_save_models'      : False             # Wheter to auto save the trained models. Defaul 'Models' folder, saves in .pth
#.....                                      # For more parameters (mainly path related) see script or document in folder
}

CTGAN_Fit_Params = {
'trained_gen_user_model'     : None,        # Name of already trained user generator model (.pt or .pth) (default is in 'Models' folder)     
'trained_disc_user_model'    : None,        # Name of already trained user discriminator model (.pt or .pth) (default is in 'Models' folder)
'trained_gen_item_model'     : None,        # Name of already trained item generator model (.pt or .pth) (default is in 'Models' folder)
'trained_disc_item_model'    : None         # Name of already trained item discriminator model (.pt or .pth) (default is in 'Models' folder)
}

## -------------------------------- MMPP (ARRIVAL RATE) PARAMETERS ----------------------------------------------

MMPP_params = {
'train_data'                : None,         # Name of arrival dataframe that includes a timestamp column (often same as event data)
'n_states'                  : 3,            # Name of Markov Transition States to define (e.g. High, Med, Low Traffic = 3)
'kmeans_n_init'             : 20,           # Number of random init posititions to user the KMEANS algorithm
'interval'                  : 60,           # Interval in seconds that gets used to estimate arrival rate
'input_transition_matrix'   : None,         # Manual input Numpy transition matrix with probs for states, if input it will not get trained on data
'input_arrival_rates'       : None,         # Manual input dictonary of arrival rates for each state. Format for e.g 3 states: {0:x, 1:x, 2:x} it should match the transition matrix above
}

## -------------------------------- SIMULATION PARAMETERS -------------------------------------------------------

SimParams = {
'simulation_length' : 180,                  # Length of simulation in seconds
'p_new_item'        : 0.15,                 # Probability of new item added
'p_old_user'        : 0.7,                  # Probability of existing user arriving
'p_new_user'        : 0.3,                  # Probability of new user arriving
'use_database'      : True,                 # Wheter to use the database
'prints'            : True,                 # Wheter to allow prints
'train_interval'    : 20,                   # Training interval of recsys and interaction model (amount in arrivals)
'report_interval'   : 60,                   # Interval of outputting reports (amount in arrivals)
'API'               :'http://127.0.0.1:5001'# Server Adress on which recsys is located (default is local: in SimEnvironment Run 'python server.py')
}

database_info = {
'dbname'            : 'Generator',          # Database Name
'user'              : 'postgres',           # Database username
'password'          : '0000',               # Password of Database
'host'              : 'localhost',          # Host name
'port'              : '5432',               # Port to which to connect
'table_name_users'  : 'users',              # Name of table in which users will be stored
'table_name_items'  : 'items'               # Name of table in which items will be stored
}


## -------------------------------- COMPILE AND RUN SIMULATION  -----------------------------------------------------


#Compile all parameters
compiler = Compiler(DataFields=DataFields,
                    Userdata=Userdata,
                    Itemdata=Itemdata,
                    Eventdata=Eventdata,
                    CTGAN_params=CTGAN_params,
                    CTGAN_Fit_Params=CTGAN_Fit_Params,
                    MMPP_params=MMPP_params,
                    SimParams=SimParams,
                    database_info=database_info)

#Run simulation
compiler.compile_and_run()