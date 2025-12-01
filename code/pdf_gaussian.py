import numpy as np
import matplotlib.pyplot as plt

# Gaussian PDF
def gaussian(x, mu, sigma2):
    return 1/np.sqrt(2*np.pi*sigma2) * np.exp(-(x - mu)**2 / (2*sigma2))

# Prior (a priori)
mu_prior = 0.0
sigma2_prior = 1.0

# Likelihood (measurement)
mu_meas = 1.0   # measurement says "probably near 1"
sigma2_meas = 0.25  # quite confident

# Posterior (Bayesian update formula for Gaussians)
mu_post = (sigma2_meas * mu_prior + sigma2_prior * mu_meas) / (sigma2_prior + sigma2_meas)
sigma2_post = (sigma2_prior * sigma2_meas) / (sigma2_prior + sigma2_meas)

# Plot
x = np.linspace(-3, 3, 400)
plt.plot(x, gaussian(x, mu_prior, sigma2_prior), label='Prior (a priori, super minus)', color='blue')
plt.plot(x, gaussian(x, mu_meas, sigma2_meas), label='Likelihood (measurement)', color='red')
plt.plot(x, gaussian(x, mu_post, sigma2_post), label='Posterior (a posteriori, super plus)', color='green')
plt.legend()
plt.title("Gaussian update: prior × likelihood → posterior")
plt.xlabel("x")
plt.ylabel("probability density")
plt.show()
