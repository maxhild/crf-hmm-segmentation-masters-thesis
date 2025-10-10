import numpy as np
import sklearn

frames = np.array(['appendix', 'blood', 'diverticule', 'grasper', 'ileocecalvalve', 'ileum', 'low_quality', 'nbi', 'needle', 'outside', 'snare', 'polyp', 'water_jet', 'wound'])
n_states = len(frames)
initial = np.ones((len(frames), len(frames)))
transitions_test = np.random.rand(len(frames), len(frames))


def simulate_model():
    state_idx = np.random.choice(len(frames))
    state = frames[state_idx]
    observations = []
    for _ in range(100):
        observations.append(state)
        # Normalisiere die Übergangswahrscheinlichkeiten für den aktuellen Zustand
        transition_probs = transitions_test[state_idx] / np.sum(transitions_test[state_idx])
        # Wähle den nächsten Zustand
        state_idx = np.random.choice(len(frames), p=transition_probs)
        state = frames[state_idx]
    return observations

print(f"{simulate_model()}")

def decode_viterbi(sequence):
    