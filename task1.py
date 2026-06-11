import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV file
df = pd.read_csv("data.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Basic Information
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Select a numeric column and calculate average
column_name = df.select_dtypes(include=['number']).columns[0]

average_value = df[column_name].mean()

print(f"\nAverage of '{column_name}' = {average_value:.2f}")

# --------------------------
# BAR CHART
# --------------------------
plt.figure(figsize=(8, 5))

if len(df.columns) >= 2:
    plt.bar(df.iloc[:, 0], df.iloc[:, 1])
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title("Bar Chart")

plt.tight_layout()
plt.show()

# --------------------------
# SCATTER PLOT
# --------------------------
numeric_cols = df.select_dtypes(include=['number']).columns

if len(numeric_cols) >= 2:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title("Scatter Plot")
    plt.tight_layout()
    plt.show()

# --------------------------
# HEATMAP
# --------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# --------------------------
# INSIGHTS
# --------------------------
print("\nInsights:")
print(f"1. Average value of {column_name} is {average_value:.2f}.")
print("2. Bar chart helps compare values across categories.")
print("3. Scatter plot shows relationships between two numerical variables.")
print("4. Heatmap highlights strong positive or negative correlations.")