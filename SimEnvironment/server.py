from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import io
from recsys import Recsys  # Import your recommender system class

app = Flask(__name__)


# DEMONSTRATION SERVER SCRIPT
# -----------------------------------------------------------------------------------------------
# This local server simulates a server on which a recommender system can be placed.
# To run this server: Run in terminal in SimEnvironment Directory 'python server.py'
# In production, replace `server_url` with the URL of your actual server where the recommender system is hosted.
# tip: The other simulation script sends user data to a database. In a real-world scenario, connect this to the recsys
# -----------------------------------------------------------------------------------------------





## -------------- RECOMMENDATION ENDPOINT ----------------------------

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Endpoint to handle recommendation requests.
    In a real system, you would use the data received to generate and return recommendations.
    """
    data = request.json
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    event = data.get('event')
    
    # Here, you would normally process the data and generate recommendations
    # For this demonstration a simple item based recommender is used.
    recommendations = recsys.recommend(item_id)
    recommendations = convert_int64(recommendations)
    # Return the recommendations back to the user
    return jsonify({'recommendations': recommendations})


## -------------- TRAIN RECSYS ENDPOINT ----------------------------

@app.route('/train', methods=['POST'])
def train():
    
    """
    Endpoint to trigger training of the recommender system.
    In a real system, you would implement the training logic for your recsys here.
    """

    #Get Training Data
    train_data = request.get_json()
    train_data_string = train_data['data']
    train_df = pd.read_json(io.StringIO(train_data_string), orient='split')
    userfield = train_data['userfield']
    itemfield = train_data['itemfield']
    valuefield = train_data['valuefield']
    
    #Train the Recsys
    global recsys
    recsys = Recsys(train_df, userfield, itemfield, valuefield)
    return jsonify({'status': 'Recommender system trained'})


# Helper function for converting data
def convert_int64(data):
    if isinstance(data, dict):
        return {k: convert_int64(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_int64(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    else:
        return data


if __name__ == '__main__':
    app.run(debug=True, port=5001)


