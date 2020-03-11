import classifier
import matplotlib.pyplot as plt

# Set figure AUC
plt.figure(figsize=(8, 8))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")

x = [0.0, 1.0]
plt.plot(x, x, linestyle='dashed', color='black', linewidth=2, label='Random')

# Run the random_forest classifier
plt = classifier.run_classifier("random_forest",plt)

# Run the AdaBoost classifier
plt = classifier.run_classifier("ada_boost",plt)

# Finish plot AUC
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig("auc-roc")
