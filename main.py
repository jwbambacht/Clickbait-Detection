from dataset import Dataset

# Import the small and big dataset.
small_dataset = Dataset("datasets/small_training")
big_dataset = Dataset("datasets/big_training")

# Print summaries of both.
small_dataset.print_summary()
big_dataset.print_summary()

# Print the 10th element of both datasets.
small_dataset.get_elements()[10].pretty_print(verbose=True)
big_dataset.get_elements()[10].pretty_print(verbose=True)

# Get their features.
print(small_dataset.get_features().shape)
print(big_dataset.get_features().shape)

# Get target labels.
print(small_dataset.get_target_labels())
print(big_dataset.get_target_labels())
