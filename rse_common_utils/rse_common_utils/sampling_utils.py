
import numpy as np
import matplotlib.pyplot as plt

def sample_normal_distribution(b_sqrd):
    return 0.5 * np.sum(np.random.default_rng().uniform(-np.sqrt(b_sqrd), np.sqrt(b_sqrd), 12))



def main():
	# Number of samples to collect
	num_samples = 10000
	b_sqrd = 4  # Variance, b^2

	print(sample_normal_distribution(b_sqrd))

	# Collect samples
	samples = np.array([sample_normal_distribution(b_sqrd).sum() for _ in range(num_samples)])

	# Plotting the histogram of the samples
	plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

	# Plot formatting
	plt.title('Histogram of Normal Distribution Samples')
	plt.xlabel('Value')
	plt.ylabel('Frequency')

	# Show the plot
	plt.show()

if __name__ == '__main__':
	main()