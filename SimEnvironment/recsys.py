

##### PLACEHOLDER RECOMMENDER SCRIPT ####
# -----------------------------------------------------------------------------------------------
# place here your own recommender script. The current 
# item-based collaborative recommender is for demonstration purposes. Connect
# recsys with the user&item databases to incorperate simulated user and item features into your recsys
# if it uses context for recommendations.
# -----------------------------------------------------------------------------------------------


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recsys:

    def __init__(self, ratings_df, userfield, itemfield, valuefield):
        self.ratings_df = ratings_df
        self.userfield = userfield
        self.itemfield = itemfield
        self.valuefield = valuefield
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(ratings_df[self.itemfield].unique())}
        self.index_to_item_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()}
        self.similarity_matrix = self.calculate_similarity_matrix()

    def calculate_similarity_matrix(self):
        # Pivot the DataFrame to create the user-item matrix
        user_item_matrix = self.ratings_df.pivot_table(index=self.userfield, columns=self.itemfield, values=self.valuefield, aggfunc='sum', fill_value=0)
        # Calculate item-item similarity matrix using cosine similarity
        return cosine_similarity(user_item_matrix.T)

    def recommend(self, item_id, top_n=10):
        # Map item ID to index
        item_index = self.item_id_to_index.get(item_id)
        if item_index is None:
            raise ValueError(f"Item ID {item_id} does not exist in the ratings DataFrame.")

        # Get similarity scores for the given item
        similarity_scores = self.similarity_matrix[item_index]

        # Sort indices based on similarity scores
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Exclude the item itself from the list of similar items
        similar_items_indices = [idx for idx in sorted_indices if idx != item_index]

        # Map indices to item IDs
        similar_items = [self.index_to_item_id[idx] for idx in similar_items_indices]

        # Return top N similar items
        return similar_items[:top_n]