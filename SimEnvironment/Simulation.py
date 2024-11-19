#Packages
import psycopg2
import numpy as np
import pandas as pd
import requests
import random
import time
from collections import deque

#Treading
from threading import Thread, Lock

#Modules
from Modules.Evaluation import DataQuality, DataQualityItems, SimulationQuality


### DEMONSTRATION INTERACTION SCRIPT (PLACE HERE) ###
from SimEnvironment.interactionmodel import svd_history


# SIMULATION SCRIPT
# -----------------------------------------------------------------------------------------------
# This script contains the actual simulation. The purpose of this script is modeling
# the flow of the simulation. So getting new users/item from the CTGAN, sending users
# to the server at a rate retreived from the MMPP. While sending data to databases, and calling 
# the report outputs.
# -----------------------------------------------------------------------------------------------


class Simulation:

    def __init__(self, eventdata=None, CTGAN_users=None, CTGAN_item=None, MMPP=None, DB=None, sim_params=None,
                continuous_columns = None, discrete_columns = None, continuous_columns_items = None, discrete_columns_items = None,
                userdata= None, DataFields= None, itemdata=None
                 ):

        '''
        Args (provided by main script)

        eventdata   :   Training Event Data, Used for training Interaction, demo Recommender System
        CTGAN       :   A trained CTGAN
        MMPP        :   A trained MMPP
        Evaluator   :   Inititialised Evaluator Script
        DB          :   (Dict) with Database information to store Userdata (optional)
        simparams   :   (Dict) with simulation Parameters
        '''

        #Variables
        self.eventdata = eventdata
        self.userfield = DataFields.get('user_id')
        self.itemfield = DataFields.get('item_id')
        self.valuefield = DataFields.get('event')
        self.known_users = set(self.eventdata[self.userfield])
        self.known_items = set(self.eventdata[self.itemfield])

        #Modules
        self.CTGAN_user = CTGAN_users
        self.CTGAN_item = CTGAN_item
        self.MMPP = MMPP

        #Sim Params
        self.sim_params = sim_params
        self.API = sim_params.get('API')
        self.train_interval = sim_params.get('train_interval')
        self.database = sim_params.get('use_database')
        self.p_new_item = sim_params.get('p_new_item')
        self.simulation_length = self.sim_params.pop('simulation_length')
        self.reporting_batch = self.sim_params.pop('report_interval')
        self.prints = self.sim_params.get('prints')

        #Database Params
        self.DB = DB
        self.table_name_users = self.DB.pop('table_name_users')
        self.table_name_items = self.DB.pop('table_name_items')
        
        #Init Queue Essentials
        self.server_busy = False
        self.queue = deque()
        self.lock = Lock()
        
        #Log Variables
        self.users_processed = 1
        self.new_items_added = 0
        self.types_of_user = []
        self.arrival_times = []
        self.queue_lengths = []
        self.session_lengths = []
        self.user_ids_entered = []
        self.event_log = []
        self.recommendations_log = []
        self.new_new_users = []
        self.new_items = []

        #reporting variables
        self.continuous_columns = continuous_columns
        self.discrete_columns = discrete_columns
        self.continuous_columns_items = continuous_columns_items
        self.discrete_columns_items = discrete_columns_items
        self.original_user_data = userdata
        self.original_item_data = itemdata
        self.report_counter = 0
        self.item_report_counter = 0
        self.type_events = list(self.eventdata[self.valuefield].unique())
        self.orginal_avg_arrival_rate = self.MMPP._calc_avg_arrival_rate()


        self.init_interaction_model(self.eventdata)
        self.init_recsys(self.eventdata)
        self.init_reporting()


#-------------------------------- SIMULATION LOGIC --------------------------------------------------------

    #Phase 1:  Generate A User
    def generate_user(self):
        types = ['new', 'old']
        p = [self.sim_params.get('p_new_user'), self.sim_params.get('p_old_user')]
        type_of_arrival = np.random.choice(types, p=p)

        # Existing User
        if type_of_arrival == 'old':
            self.types_of_user.append('existing user')
            user_id = np.random.choice(list(self.known_users))
            return user_id

        # New User & add to known users
        elif type_of_arrival == 'new':
            self.types_of_user.append('new user')
            data = self._new_user()
            user_id = data[self.userfield][0]
            self.known_users.add(user_id)
            self.new_new_users.append(data)
            return user_id


    #Phase 2: Define an Interaction for the user
    def _define_interaction(self, userid):
        session_length = random.randint(0, 5)
        self.session_lengths.append(session_length)
        potential_items = self.interaction_model.likely_to_interact([userid])
        interacted_items = random.sample(potential_items, session_length)
        events = [self.interaction_model.get_event() for _ in range(session_length)]
        return interacted_items, events
    

    #Phase 3: Add user to the queue
    def enqueue(self, user_id):
        with self.lock:
            self.queue.append(user_id)
            self.queue_lengths.append(len(self.queue))


    #Phase 4: Process the Queue
    def process_queue(self):
        while self.queue:
            with self.lock:
                user_id = self.queue.popleft()
                self.server_busy = True
            interacted_items, events = self._define_interaction(user_id)
            for item, event in zip(interacted_items, events):
                self.get_recommendations(user_id=user_id, item_id=item, event=event)
            with self.lock:
                self.server_busy = False


    #Phase 5: Send user to recsys server
    def get_recommendations(self, user_id, item_id, event):
        data = {'user_id' : int(user_id),
                'item_id' : int(item_id),
                'event' : int(event),
                'timestamp' : int(time.time())} 
        
        #Send user to server (for this recsys only userid, itemid, and event)
        response = requests.post(f"{self.API}/recommend", json=data)

        if response.status_code == 200:
            self._parse_response_data(response, user_id, item_id, event)
        
            


    #Phase 6: Simulate everything for a certain amount of time
    def simulate(self):
        start_time = time.time()

        while time.time() - start_time < self.simulation_length:
            if self.new_item():
                continue
            user_id = self.generate_user()
            self.users_processed += 1
            self.user_ids_entered.append(user_id)
            self.arrival_times.append(time.time())
            self.enqueue(user_id)
            if not self.server_busy:
                t = Thread(target=self.process_queue)
                t.start()

            if (self.users_processed % self.train_interval) == 0:
                self.retraining()
            if (self.users_processed % self.reporting_batch) == 0:
                self._call_interval_functions()
            if len(self.new_items) > self.reporting_batch:
                self._interval_item_logic()

            interarrival = np.clip(self._arrival_rate(), 0.01,100)
            time.sleep(interarrival)
        self.queue.clear()
        print("\nSimulation Ended")

    #Phase 7: New Item Added
    def new_item(self):
        add_item = np.random.choice([True, False], p = [self.p_new_item, (1-self.p_new_item)])
        if add_item:
            new_item_data = self._new_item()
            new_item_id = new_item_data[self.itemfield][0]
            self.new_items_added += 1
            self.known_items.add(new_item_id)
            if self.database:
                self.new_items.append(new_item_data)
            user = np.random.choice(list(self.known_users))
            self.event_log.append({
             self.userfield : user,
             self.itemfield : new_item_id,
             self.valuefield : np.random.choice(self.type_events)})
            if self.prints:
                print(f'new item added with id: {new_item_id}')
            return True
        else:
            return False

    #Helper Function to get arrival rate from MMPP
    def _arrival_rate(self):
        arrival_rate = self.MMPP.get_arrival_rate()
        return 1 / arrival_rate

    #Helper Function to get new user from CTGAN
    def _new_user(self):
        return self.CTGAN_user.sample(1, new_ids=True)
    
    #Helper Function to get new item from CTGAN
    def _new_item(self):
        return self.CTGAN_item.sample(1, new_ids=True)


#-------------------------------- INTERVAL LOGIC --------------------------------------------------------
    
    ''' This section contains the code that executes when certain intervals are met. It saves important data to databases
    generates report and clears logging data again to save memory'''

    def _call_interval_functions(self):
        self.report_counter += 1 
        self.user_report()
        self.simulation_report()
        self._store_users_in_db()
        self.new_new_users.clear()
        self.recommendations_log.clear()
    
    def _interval_item_logic(self):
        self.item_report_counter += 1
        self.item_report()
        self._store_items_in_db()
        self.new_items.clear()


#-------------------------------- LOG DATA LOGIC --------------------------------------------------------

    '''
    This section includes the code for logging, saving and sending data to the database. Data gets
    logged for each batch, after which recsys and interaction system will be retrained with the new data.
    From this data evaluation reports will be made (next section). Finally the logs will be cleared to
    save memory.
    
    '''

    # 1.1. Store new user data into database (e.g. connected to recsys)
    def _store_users_in_db(self):
        '''Code to store new users in the database'''
        if self.database:
            if len(self.new_new_users) > 0:
                conn = psycopg2.connect(**self.DB)
                cursor = conn.cursor()
                df = pd.concat(self.new_new_users)
                df = df.where(pd.notnull(df), None)
                data_tuples = [tuple(x) for x in df.to_numpy()]
                columns = ', '.join(df.columns)
                placeholders = ', '.join(['%s'] * len(df.columns))
                query = f"INSERT INTO {self.table_name_users} ({columns}) VALUES ({placeholders})"
                cursor.executemany(query, data_tuples)
                print('New User data Added to DataBase')
                conn.commit()
                cursor.close()
                conn.close()

    # 1.2. Store new Item data into database (e.g. connected to recsys)
    def _store_items_in_db(self):
        '''Code to store new Items in the database'''
        if self.database:
            if len(self.new_item)>0:
                conn = psycopg2.connect(**self.DB)
                cursor = conn.cursor()
                df = pd.concat(self.new_items)
                df = df.where(pd.notnull(df), None)
                data_tuples = [tuple(x) for x in df.to_numpy()]
                columns = ', '.join(df.columns)
                placeholders = ', '.join(['%s'] * len(df.columns))
                query = f"INSERT INTO {self.table_name_items} ({columns}) VALUES ({placeholders})"
                cursor.executemany(query, data_tuples)
                print('New Item data Added to DataBase')
                conn.commit()
                cursor.close()
                conn.close()

    # 2. Parse response data of the recsys (save recommendation data e.g. simulated event data)
    def _parse_response_data(self, response, user, item, event):
        response_data = response.json()
        recommendations = response_data.get('recommendations')
        self.event_log.append({
             self.userfield : user,
             self.itemfield : item,
             self.valuefield : event})
        self.recommendations_log.append({
            'user-id': user,
            'recommendations': recommendations
        })
        if self.prints:
            print(f'user: {user}, item: {item} recommendations: {recommendations}')
    
    #3. Retreive the logs and output then in a dict, Clear the log lists for memory
    def _get_logs(self):
        logs = {
            'user_id': self.user_ids_entered,
            'arrival_times': self.arrival_times,
            'queue_lengths': self.queue_lengths,
            'user_types': self.types_of_user,
            'session_lengths':self.session_lengths,
            'new_items_added':self.new_items_added
        }

        self.user_ids_entered = []
        self.arrival_times = []
        self.queue_lengths = []
        self.types_of_user = []
        self.session_lengths = []
        self.new_items_added = 0
        return logs

    
#-------------------------------- TRAIN / INIT LOGIC --------------------------------------------------------

    # 1. Initializing Local Interaction model
    def init_interaction_model(self, data=None):
        self.interaction_model = svd_history(self.userfield,self.itemfield,self.valuefield, data=data)
        self.interaction_model.create_matrix(data)
        self.interaction_model.fit_svd()
        print('Interaction Model Trained')

    # 2. Initializing Recsys on server
    def init_recsys(self, data=None):
        if data is not None:
            data = data.to_json(orient='split')
            payload = {'data' : data, 'userfield':self.userfield, 'itemfield':self.itemfield, 'valuefield':self.valuefield}
        response = requests.post(f"{self.API}/train", json=payload)
        if response.status_code == 200:
            response_data = response.json()
            print(f'{response_data.get("status")}')

    # 3. Retraining of Interaction & Recsys (optionally: adjust for own recsys logic)
    def retraining(self):       
        new_events = pd.DataFrame(self.event_log)
        self.eventdata = pd.concat([self.eventdata, new_events])
        self.init_interaction_model(self.eventdata)
        self.init_recsys(self.eventdata)
        self.event_log.clear()


#------------------------------ EVALUATION LOGIC -------------------------------------------------------

    
    '''USER, ITEM & Simulation QUALITY REPORT
    At every specified interval an report will be generated to evaluate if the generated users, items and simulation parameters
    are similar to the original training data. After this, data will be cleared to save memory. Note:
    if database is True, user and item data will be saved in the database (which can be used by the recsys if needed).
    The event data is also used to train the recsys and interaction model from previous section before cleared.'''

    def user_report(self):
        if len(self.new_new_users) > 0:
            generated_users = pd.concat(self.new_new_users)
            real_data = self.original_user_data.sample(len(generated_users))
            self.userreport.evaluate(real_data, generated_users, number=self.report_counter)
            print('\nNew user report Generated!\n')
    
    def item_report(self):
        generated_items = pd.concat(self.new_items)
        real_data = self.original_item_data.sample(len(generated_items))
        self.itemreport.evaluate(real_data, generated_items, number=self.item_report_counter)
        print('\nNew Item report Generated!\n')
    
    def simulation_report(self):
        logs = self._get_logs()
        self.simulationreport.evaluate(logs, self.recommendations_log, number=self.report_counter)
        print('\nNew Simulation report Generated!\n')

    def init_reporting(self):
        self.userreport = DataQuality(self.discrete_columns, self.continuous_columns, id_field=self.userfield)
        self.itemreport = DataQualityItems(self.discrete_columns_items, self.continuous_columns_items, id_field=self.itemfield)
        self.simulationreport = SimulationQuality(avg_rate_real=self.orginal_avg_arrival_rate)
    