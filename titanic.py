import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read Titanic dataset
df = pd.read_csv("/content/titanic 1.csv")

# Select only 10 rows
df = df.head(10).copy()

# Display dataset
print(df.head(10))

# Dataset shape
print("Shape:", df.shape)

# Check missing values
print(df.isnull().sum())

# Scatter plot
sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived")

plt.title("Titanic Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.show()
