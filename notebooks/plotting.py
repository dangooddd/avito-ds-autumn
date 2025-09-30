# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme()

# Load data
df = pd.read_csv("data/output/val.csv")

# Create figure with 3 subplots arranged horizontally
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Define colors for different line types
colors = ["blue", "orange", "green", "red", "purple"]

# Plot 1: spaces parameter
param = "spaces"
sns.lineplot(data=df, x=param, y="f1", color="green", ax=axes[0])
axes[0].set_title(f"F1 Score vs {param}")
axes[0].set_xlabel(param)
axes[0].set_ylabel("F1 Score")
axes[0].legend(labels=[param + " avg", param + " 95%"], loc="lower right")

# Plot 2: min_tries parameter
param = "min_tries"
sns.lineplot(data=df, x=param, y="f1", color="green", ax=axes[1])
axes[1].set_title(f"F1 Score vs {param}")
axes[1].set_xlabel(param)
axes[1].set_ylabel("F1 Score")
axes[1].legend(labels=[param + " avg", param + " 95%"], loc="lower right")

# Plot 3: max_tries parameter
param = "max_tries"
sns.lineplot(data=df, x=param, y="f1", color="green", ax=axes[2])
axes[2].set_title(f"F1 Score vs {param}")
axes[2].set_xlabel(param)
axes[2].set_ylabel("F1 Score")
axes[2].legend(labels=[param + " avg", param + " 95%"])

# Adjust layout and display
plt.tight_layout()
plt.show()
