import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points and categories
n_points = 20
n_categories = 3

# Generate random probabilities for each of the 3 categories for 20 data points
probabilities = np.array([
    [0.0556, 0.0509, 0.0363, 0.0385, 0.0518, 0.0471, 0.0336, 0.0645, 0.0413, 0.0308, 0.0562, 0.0573, 0.0671, 0.0548, 0.0451, 0.0298, 0.0631, 0.0403, 0.095, 0.0408],
    [0.0265, 0.0454, 0.0321, 0.044, 0.0354, 0.0629, 0.0281, 0.047, 0.0404, 0.0282, 0.0477, 0.0458, 0.0517, 0.0844, 0.056, 0.0381, 0.08, 0.0469, 0.1122, 0.0472],
    [0.0285, 0.0355, 0.0477, 0.0407, 0.0553, 0.0607, 0.0257, 0.0299, 0.0361, 0.0349, 0.0385, 0.0363, 0.0577, 0.0495, 0.0453, 0.0364, 0.1291, 0.0313, 0.1284, 0.0526],
]).T

# Create x locations for each data point's bars
x = np.arange(n_points)

# Set width of bars
bar_width = 0.25

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

labels = ["OT", "NT", "Quran"]
# Create bars for each category
for i in range(n_categories):
    ax.bar(x + i * bar_width - (n_categories - 1) * bar_width / 2, 
           probabilities[:, i], bar_width, label=f'{labels[i]}')

# Adding labels and title
ax.set_xlabel('Topics')
ax.set_ylabel('Probability')
ax.set_title('Probabilities for 3 Texts Across 20 Topics')
ax.set_xticks(x)
ax.set_xticklabels([f'Topic {i}' for i in range(n_points)], rotation=45)
ax.legend()
ax.grid()

# Display the plot
plt.tight_layout()
plt.show()