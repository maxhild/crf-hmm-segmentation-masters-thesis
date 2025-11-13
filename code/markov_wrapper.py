import numpy as np

class MarkovStateWrapper:
    def __init__(self):
        """
        Initializes a random transition matrix for a fixed set of known states.
        """
        self.states = [
            'appendix', 'blood', 'diverticule', 'grasper', 'ileocecalvalve',
            'ileum', 'low_quality', 'nbi', 'needle', 'outside',
            'snare', 'polyp', 'water_jet', 'wound'
        ]
        self.n_states = len(self.states)
        self.state_matrix = self._random_stochastic_matrix(self.n_states)
        self.current_state = "outside"
        
    @staticmethod
    def process_frame_by_state():
        
        

    @staticmethod
    def _random_stochastic_matrix(n):
        A = np.random.rand(n, n)
        return A / A.sum(axis=1, keepdims=True)

    @staticmethod
    def is_stochastic(A, tol=1e-8):
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return False
        return np.all(A >= -tol) and np.allclose(A.sum(axis=1), 1, atol=tol)

    @staticmethod
    def normalize_to_law_of_total_prob(A):
        A = np.maximum(A, 0)
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return A / row_sums

    def ensure_stochastic(self):
        """Ensures state_matrix satisfies the law of total probability."""
        if not self.is_stochastic(self.state_matrix):
            self.state_matrix = self.normalize_to_law_of_total_prob(self.state_matrix)
