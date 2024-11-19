import numpy as np
from sklearn.cluster import KMeans


class MMPP:

    def __init__(self, n_states=3, n_init=20):
        self.n_states = n_states
        self.n_init = n_init
        self.manual = False

    ### AUTOMATIC SAMPLING OF ARRIVAL RATES AND TRANSITION MATRIX ###
    def fit(self, timestamps, interval=60):
        self.timestamps = timestamps
        interarrival_times = np.diff(timestamps.sort_values())
        arrival_rates = self._estimate_arrival_rates(interarrival_times, interval)
        states, self.pred_arrival_rates = self._cluster_arrival_rates(arrival_rates, self.n_states)
        self.transition_matrix = self._estimate_transition_matrix(states)
        self.intensity_states = list(set(states))
        self.current_state = np.random.choice(self.intensity_states)

    ### MANUAL INPUT OF ARRIVAL RATES AND TRANSITION MATRIX ###
    def manual_input(self, input_transition_matrix, input_arrival_rates):
        self.manual = True
        self.pred_arrival_rates = input_arrival_rates
        self.transition_matrix = input_transition_matrix
        self.intensity_states = list(range(len(self.transition_matrix)))
        self.current_state = np.random.choice(self.intensity_states)

    ### FUNCTION THAT OUTPUTS DYNAMICALLY CHANGING ARRIVAL RATE ###
    def get_arrival_rate(self):
        return self._switch_states()

    ### HELPER FUNCTIONS BELOW ###
    def _switch_states(self):
        '''Switches states based on transition matrix'''
        probs = self.transition_matrix[self.current_state]
        next_state = np.random.choice(self.intensity_states, p = probs)
        self.current_state = next_state
        return np.clip(self.pred_arrival_rates[self.current_state],0.01,999)

    def _estimate_arrival_rates(self, interarrival_times, interval):
        '''Estimates arrival rates per interval (default = minute)'''
        time_series = np.cumsum(interarrival_times)
        bins = np.arange(0, time_series[-1] + interval, interval)
        counts, _ = np.histogram(time_series, bins=bins)
        arrival_rates = counts / interval
        return arrival_rates
    
    def _cluster_arrival_rates(self, arrival_rates, n_clusters):
        '''Clusters the arrival rates in n states'''
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=self.n_init).fit(arrival_rates.reshape(-1, 1))
        states = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_.flatten()
        return states, cluster_centers

    def _estimate_transition_matrix(self, states):
        '''Creates a transition marix'''
        n_states = len(np.unique(states))
        transition_matrix = np.zeros((n_states, n_states))
        
        for (i, j) in zip(states[:-1], states[1:]):
            transition_matrix[i, j] += 1
        
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        return transition_matrix
    
    def _calc_avg_arrival_rate(self):
        if self.manual:
            P = self.transition_matrix
            ar = self.pred_arrival_rates
            n = P.shape[0]
            arrival_rates = np.array([ar[state] for state in sorted(ar.keys())])
            A = np.append(np.transpose(P) - np.eye(n), [np.ones(n)], axis=0)
            b = np.append(np.zeros(n), [1])
            steady_state_probs = np.linalg.lstsq(A, b, rcond=None)[0]
            average_arrival_rate = np.dot(steady_state_probs, arrival_rates)
            return average_arrival_rate

        ts = self.timestamps.sort_values()
        ts = ts - ts.min()
        max_time = ts.max()
        average_arrival_rate = len(ts) / max_time
        return average_arrival_rate

    








