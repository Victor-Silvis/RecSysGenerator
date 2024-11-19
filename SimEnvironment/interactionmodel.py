

'''
##### PLACEHOLDER INTERACTION SCRIPT ####

place here your own interaction script. Current basic script
is for demonstration purposes.

This script includes:
    - SVD based system to get user most prefered items, with they would likely interact
    - Random item selection for new users with no history
    - Randomness in item selection. E.G. Sometimes users will view, items that are not necessary items they normally like.

input: user-id

'''

#packages
import numpy as np
from scipy.linalg import sqrtm
import pandas as pd
import random


class svd_history:

    def __init__(self, userField, itemField, valueField, data):
        self.param = {}
        self.userField = userField
        self.itemField = itemField
        self.valueField = valueField
        self.update_counter = 0
        self.type_events = list(data[self.valueField].unique())

    ''' 
    args for SVD:
    userField   : The name of the column that contains user_id's
    itemField   : The name of the column that contains item_id's
    valueField  : the name of the column that contains the values (ratings)
    '''

    def create_matrix(self, data):
        self.data = data



        utility_matrix = data.pivot_table(index=self.userField, 
                                    columns=self.itemField,
                                    values=self.valueField,
                                    aggfunc='sum')
                         
        #Create mapping
        N = data[self.userField].nunique()
        M = data[self.itemField].nunique()
        user_list = np.unique(data[self.userField])
        item_list = np.unique(data[self.itemField])
        self.user_to_index = dict(zip(user_list, range(0, N)))
        self.index_to_item = dict(zip(range(0,M), item_list))
        self.items = item_list.tolist()

        #Create Matrix Operations
        matrix_array = utility_matrix.values #To array
        mask = np.isnan(matrix_array) #Get nan value mask True/False
        masked_arr = np.ma.masked_array(matrix_array, mask) #mask based on True/False
        self.predMask = ~mask #store the inverse of the mask for later recommendation (for what the user already has watched)
        item_means = np.mean(masked_arr, axis=0) #get mean of each item
        matrix = masked_arr.filled(item_means) #Replace NaN with item means
        self.item_means_tiled = np.tile(item_means, (matrix.shape[0], 1)) #2D array storing means of all items, same size

        #Remove the per item average from all entries, Nan will essentially be zero
        self.matrix = matrix - self.item_means_tiled

#SVD fit
    def fit_svd(self, k = 20):
        #SVD magic, U and V are user and item features
        U, s, V = np.linalg.svd(self.matrix, full_matrices=False)
        s = np.diag(s)
        #next we take only K most significant features
        s = s[0:k,0:k] #Select the top K diagonal elements
        U = U[:,0:k] #Keep the first K columns of U
        V = V[0:k,:] #Keep the first K rows of V
        s_root = sqrtm(s) #Compute the square root of the diagonal matrix s
        Usk = np.dot(U, s_root) # Multiply U by the square root of s
        skV = np.dot(s_root, V) # Multiply the square root of s by V
        UsV = np.dot(Usk, skV) # Compute the reconstructed matrix UsV by multiplying Usk and skV
        self.UsV = UsV + self.item_means_tiled #we add the means back in to get final predicted ratings
        #UsV now ofcourse contains the final predicted ratings for each user-item combination
    
#Recommender   
    def likely_to_interact(self, users_list, N=10, values=True):
        # self.predmask is a mask that has True if already seen, This was a False for the Nan Mask just inversely stored
        #predMat consists of items that are not yet discovered by the user, the ones that are, are set very low
        predMat = np.ma.masked_where(self.predMask, self.UsV).filled(fill_value=-999)
        recommendations = [] #init list in which recommendations will be stored

        randomness = np.random.choice([True, False], p=[0.05, 0.95])
        if randomness:
            return random.sample(self.items,N)

        if values == True:
            for user in users_list:
                try:
                    user_idx = self.user_to_index[user]
                except:
                    return self.unkown_user(N)
                top_indeces = predMat[user_idx,:].argsort()[-N:][::-1] #access entire row, sort on ratings, take N max, return indeces of columns(items)
                recommendations.append([(user, self.index_to_item[index], predMat[user_idx, index]) for index in top_indeces])
        
        output_items = []
        for rec in recommendations[0]:
            output_items.append(rec[1])
        return output_items
    
    def get_event(self):
        return np.random.choice(self.type_events)

    def unkown_user(self, n):
        return random.sample(self.items,n)
        