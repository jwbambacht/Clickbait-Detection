from dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np

# Import the small and big dataset.
small_dataset = Dataset("datasets/small_training")
big_dataset = Dataset("datasets/big_training")

# Plot and save the class distributions for the big training set
big_clickbait = np.sum(big_dataset.get_target_labels())
big_non_clickbait = len(big_dataset.get_target_labels()) - big_clickbait
ypos = range(2)
plt.bar(ypos, [big_non_clickbait, big_clickbait], align='center')
plt.xticks(ypos, ['No-clickbait', 'Clickbait'])
plt.ylabel('Amount')

plt.savefig('figures/big_distribution')
plt.show()

# Plot and save the class distributions for the small training set
small_clickbait = np.sum(small_dataset.get_target_labels())
small_non_clickbait = len(small_dataset.get_target_labels()) - small_clickbait
plt.bar(ypos, [small_non_clickbait, small_clickbait], align='center')
plt.xticks(ypos, ['No-clickbait', 'Clickbait'])
plt.ylabel('Amount')

plt.savefig('figures/small_distribution')
plt.show()
