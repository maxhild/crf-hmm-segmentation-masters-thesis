import numpy as np
from typing import Sequence, Callable

class MarkovStateWrapper:
    def __init__(self, states: Sequence[str] | None = None):
        """
        Markov wrapper with:
        - fixed state set
        - stochastic transition matrix
        - belief vector over current state
        """
        if states is None:
            states = [
                'appendix', 'blood', 'diverticule', 'grasper', 'ileocecalvalve',
                'ileum', 'low_quality', 'nbi', 'needle', 'outside',
                'snare', 'polyp', 'water_jet', 'wound'
            ]
        self.states: list[str] = list(states)
        self.n_states: int = len(self.states)

        self.state_matrix: np.ndarray = self._random_stochastic_matrix(self.n_states)

        self.belief: np.ndarray = self._init_belief()

    def _init_belief(self) -> np.ndarray:
        b = np.zeros(self.n_states, dtype=float)
        if "outside" in self.states:
            idx = self.states.index("outside")
            b[idx] = 1.0
        else:
            b[:] = 1.0 / self.n_states
        return b

    @staticmethod
    def _random_stochastic_matrix(n: int) -> np.ndarray:
        A = np.random.rand(n, n)
        return A / A.sum(axis=1, keepdims=True)

    @staticmethod
    def is_stochastic(A: np.ndarray, tol: float = 1e-8) -> bool:
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return False
        return np.all(A >= -tol) and np.allclose(A.sum(axis=1), 1.0, atol=tol)

    @staticmethod
    def normalize_to_law_of_total_prob(A: np.ndarray) -> np.ndarray:
        A = np.maximum(A, 0.0)
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return A / row_sums

    def ensure_stochastic(self) -> None:
        """Ensure state_matrix satisfies the law of total probability."""
        if not self.is_stochastic(self.state_matrix):
            self.state_matrix = self.normalize_to_law_of_total_prob(self.state_matrix)

    def step(self, likelihood: np.ndarray) -> np.ndarray:
        """
        Filtering step.

        Args:
            likelihood: shape (n_states,)
                        likelihood[j] ∝ p(x_t | S_t = j).
                        Must be non-negative; will be renormalized.

        Returns:
            Updated belief vector b_t.
        """
        likelihood = np.asarray(likelihood, dtype=float)
        assert likelihood.shape == (self.n_states,)
        likelihood = np.maximum(likelihood, 0.0)
        if likelihood.sum() == 0.0:
            likelihood[:] = 1.0

        # 1) Predict: b_pred(j) = sum_i b_{t-1}(i) * T[i, j]
        b_pred = self.belief @ self.state_matrix

        # 2) Update with observation: b_t(j) ∝ b_pred(j) * likelihood(j)
        b_unnorm = b_pred * likelihood
        s = b_unnorm.sum()
        if s == 0.0:
            b_unnorm = b_pred
            s = b_unnorm.sum()
        self.belief = b_unnorm / s
        return self.belief

    def current_state(self) -> str:
        """Return the MAP state name."""
        idx = int(self.belief.argmax())
        return self.states[idx]
