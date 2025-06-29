import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (adjust filename if needed)
df = pd.read_csv("titanic.csv")



print("\n📊 Summary Statistics:\n")
print(df.describe())

numeric_cols = df.select_dtypes(include='number').columns

df[numeric_cols].hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Histograms for Numeric Features")
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


