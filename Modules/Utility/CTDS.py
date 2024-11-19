import numpy as np


"""
CTGAN Data Sampler
--------------------------------------

HELPER SCRIPT FOR CTGAN
This code contains the DataSampler class, which is essential for preparing 
data for the CTGAN (Conditional Tabular GAN) model. It facilitates the sampling of 
conditional vectors and corresponding data, crucial for training and generating synthetic 
tabular data. The purpose is to enable the CTGAN model to learn and generate synthetic data 
while preserving the statistical characteristics of the original dataset. Conditional vectors
are added to the latent space (noise) and fed into the Generator as described by Xu et al."""

""" Code is based upon the methodology of Xu et al. (2019)
    DOI: https://doi.org/10.48550/arXiv.1907.00503"""



class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data, output_info, log_frequency):
        self.data = data
        self.output_info = output_info
        self.log_frequency = log_frequency
        self.data_length = len(data)
        self._initialize_discrete_columns()


    def _compute_row_idx_by_category(self):
        """
        Compute and store row indices for each category in discrete columns. For each discrete column 
        in the dataset, this function computes and stores the row indices where each category appears. 
        These row indices are used during sampling to ensure that generated data maintains the distribution 
        of the original data within each category.
        """

        #Init dictionary (better for memory with many cats) to store categories and their index
        self.row_idx_by_category = {}
        col_index = 0

        #Loop through each column in the output info, and store info about discrete columns
        for column_info in self.output_info:
            if self.is_discrete_column(column_info):
                
                #Get information about the span of the column
                span_info = column_info[0]
                end_index = col_index + span_info.dim

                for j in range(span_info.dim):
                    category_indices = np.nonzero(self.data[:, col_index + j])[0]
                    self.row_idx_by_category[(col_index, j)] = category_indices 
                col_index = end_index
            else:
                col_index += sum([span_info.dim for span_info in column_info])
        assert col_index == self.data.shape[1]


    # Helper function to check if a column is discrete
    def is_discrete_column(self, column_info):
        return (len(column_info) == 1
                and column_info[0].activation_fn == 'softmax') 
    
    
    def _initialize_discrete_columns(self):
        """Initialize discrete columns and compute relevant attributes."""
        
        #Count number of discrete columns
        num_discrete_columns = sum([1 for column_info in self.output_info if self.is_discrete_column(column_info)])

        #Init Arrays to store information about discrete columns
        self.discrete_column_matrix_start = np.zeros(
            num_discrete_columns, dtype='int32')
        
        #Get idx where each category is placed
        self._compute_row_idx_by_category()

        # Compute max number of categories among discrete columns
        max_category = max([
            column_info[0].dim
            for column_info in self.output_info
            if self.is_discrete_column(column_info)
        ], default=0)

        #Store arrays to store probs and other information for each discrete columns (base for matrix)
        self.discrete_column_cond_start = np.zeros(num_discrete_columns, dtype='int32')
        self.discrete_column_num_category = np.zeros(num_discrete_columns, dtype='int32')
        self.discrete_column_category_prob = np.zeros((num_discrete_columns, max_category))
        
        # initialize variables to keep track of column and category indices
        self.num_discrete_columns = num_discrete_columns
        self.num_categories = sum([
            column_info[0].dim
            for column_info in self.output_info
            if self.is_discrete_column(column_info)
        ])

        # Reset col index and init variables for tracking column and cat indices
        col_index = 0
        current_column_id = 0
        current_cond_start = 0

        # Loop through each column again to compute category probabilities
        for column_info in self.output_info:
            if self.is_discrete_column(column_info):
                span_info = column_info[0]
                end_index = col_index + span_info.dim
                
                #Compute cat frequency for each discrete column
                category_freq = np.sum(self.data[:, col_index:end_index], axis=0)
                if self.log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                
                #Store category probabilities and other info for each discrete column
                self.discrete_column_category_prob[current_column_id, :span_info.dim] = category_prob
                self.discrete_column_cond_start[current_column_id] = current_cond_start
                self.discrete_column_num_category[current_column_id] = span_info.dim
                
                #update indices for the next column
                current_cond_start += span_info.dim
                current_column_id += 1
                col_index = end_index
            else:
                #if the column is not discrete move to the next column
                col_index += sum([span_info.dim for span_info in column_info])


    def _random_choice_prob_index(self, discrete_column_id):
        """Randomly sample indices based on probability distribution."""
        
        #Get the probability distribution for each category in the discrete column
        probs = self.discrete_column_category_prob[discrete_column_id]

        #Generate random values to select indices
        random_values = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)

        #Select indices based on the cumulative probability distribution
        return (probs.cumsum(axis=1) > random_values).argmax(axis=1)

    def sample_condvec(self, batch_size):
        """Generate the conditional vector for training."""
        if self.num_discrete_columns == 0:
            return None

        #Randomly select discrete column indices
        discrete_column_id = np.random.choice(
            np.arange(self.num_discrete_columns), batch_size)

        #Init conditional vector and mask
        cond_vector = np.zeros((batch_size, self.num_categories), dtype='float32')
        mask = np.zeros((batch_size, self.num_discrete_columns), dtype='float32')
        
        #Set mask values for selected discrete columns
        mask[np.arange(batch_size), discrete_column_id] = 1

        #Determine category indices within each selected discrete column
        category_id_in_column = self._random_choice_prob_index(discrete_column_id)
        category_id = (self.discrete_column_cond_start[discrete_column_id] + category_id_in_column)
        
        #Set corresponding values in the conditional vector
        cond_vector[np.arange(batch_size), category_id] = 1

        return cond_vector, mask, discrete_column_id, category_id_in_column

    def sample_original_condvec(self, batch_size):
        """Generate the conditional vector for generation using original frequency."""
        if self.num_discrete_columns == 0:
            return None

        # Flatten the category probability distribution
        category_freq = self.discrete_column_category_prob.flatten()
        category_freq = category_freq[category_freq != 0]
        category_freq = category_freq / np.sum(category_freq)

        #Randomly select category indices based on original frequency
        col_indices = np.random.choice(np.arange(len(category_freq)), batch_size, p=category_freq)
        
        #Init and set values in the conditional vector
        cond_vector = np.zeros((batch_size, self.num_categories), dtype='float32')
        cond_vector[np.arange(batch_size), col_indices] = 1

        return cond_vector

    def sample_data(self, data, sample_size, column_indices, option_indices):
        """Sample data from original training (real) data satisfying the sampled conditional vector."""
        
        #if no column idx is provided, sample randomly entire dataset
        if column_indices is None:
            sampled_indices = np.random.randint(len(data), size=sample_size)
            return data[sampled_indices]

        # Convert column and option indices to arrays for vectorized indexing
        column_indices = np.array(column_indices)
        option_indices = np.array(option_indices)

        #Sample indices based on selected column and option indices (vectorized)
        sampled_indices = []
        for col_idx, opt_idx in zip(column_indices, option_indices):
                indices = self.row_idx_by_category.get((col_idx, opt_idx))
                if indices is not None:
                    sampled_indices.append(np.random.choice(indices))
                else:
                    sampled_indices.append(np.random.randint(len(data)))

        return data[sampled_indices]

    def dim_cond_vector(self):
        """Return the total number of categories."""
        return self.num_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch_size):
        """Generate the condition vector."""
        cond_vector = np.zeros((batch_size, self.num_categories), dtype='float32')
        id_ = self.discrete_column_matrix_start[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        cond_vector[:, id_] = 1
        return cond_vector
