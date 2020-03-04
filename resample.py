import numpy as np


class Resample:
    @staticmethod
    def undersample(small_dataset, big_dataset):
        # Combine the two datasets
        all_features = np.r_[small_dataset.get_features(), big_dataset.get_features()]
        all_labels = np.r_[small_dataset.get_target_labels(), big_dataset.get_target_labels()]

        # Resample (undersampling)
        combined = np.c_[all_features, all_labels]
        no_clickbait = combined[combined[:, -1] == 0]
        clickbait = combined[combined[:, -1] == 1]
        nr_of_samples = np.size(clickbait, axis=0)
        ids = np.random.randint(np.size(no_clickbait, axis=0), size=nr_of_samples)
        no_clickbait_resampled = no_clickbait[ids, :]

        resampled_features = np.r_[clickbait[:, :-1], no_clickbait_resampled[:, :-1]]
        resampled_labels = np.r_[clickbait[:, -1], no_clickbait_resampled[:, -1]]

        return resampled_features, resampled_labels
