import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (adjust filename if needed)
df = pd.read_csv("titanic.csv")

# ----------------------------
# 1. Summary Statistics
# ----------------------------
print("\n📊 Summary Statistics:\n")
print(df.describe())

# ----------------------------
# 2. Histograms for Numeric Columns
# ----------------------------
numeric_cols = df.select_dtypes(include='number').columns

df[numeric_cols].hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Histograms for Numeric Features")
plt.tight_layout()
plt.show()

# ----------------------------
# 3. Boxplots (Outlier Detection)
# ----------------------------
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# ----------------------------
# 4. Correlation Heatmap
# ----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# 5. (Optional) Pairplot — Takes Time
# ----------------------------
# Uncomment if you want pairwise scatterplots
# sns.pairplot(df[numeric_cols])
# plt.show()
