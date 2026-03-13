import pandas as pd
import matplotlib.pyplot as plt


# Load Dataset


df = pd.read_csv(r"C:\Users\adity\OneDrive\Documents\student_data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
df.info()


# Data Cleaning


print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Check duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()


# Basic Statistics


print("\nStatistical Summary:")
print(df.describe())


# Feature Engineering


# Create Total Score column
df["Total_Score"] = df["Math_Score"] + df["Science_Score"] + df["English_Score"]


# Group Analysis


# Average score by gender
gender_avg = df.groupby("Gender")["Total_Score"].mean()
print("\nAverage Total Score by Gender:")
print(gender_avg)

# Average score by parental education
parent_avg = df.groupby("Parental_Education")["Total_Score"].mean()
print("\nAverage Score by Parental Education:")
print(parent_avg)

# Study hours analysis
study_analysis = df.groupby("Study_Hours")["Total_Score"].mean()
print("\nAverage Score by Study Hours:")
print(study_analysis)


# Correlation Analysis


print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))


# Top Students


top_students = df.sort_values(by="Total_Score", ascending=False).head(10)

print("\nTop 10 Students:")
print(top_students[["Student_ID", "Total_Score"]])


# Save Processed Data


df.to_excel("cleaned_student_data.xlsx", index=False)

gender_avg.to_csv("gender_analysis.csv")
parent_avg.to_csv("parent_education_analysis.csv")


# Visualizations


# 1 Gender vs Average Score
gender_avg.plot(kind="bar")
plt.title("Average Total Score by Gender")
plt.ylabel("Average Score")
plt.show()

# 2 Study Hours vs Total Score
plt.scatter(df["Study_Hours"], df["Total_Score"])
plt.title("Study Hours vs Total Score")
plt.xlabel("Study Hours")
plt.ylabel("Total Score")
plt.show()

# 3 Distribution of Total Scores
plt.hist(df["Total_Score"], bins=10)
plt.title("Distribution of Total Scores")
plt.xlabel("Total Score")
plt.ylabel("Frequency")
plt.show()

# 4 Study Hours vs Average Score
study_analysis.plot(kind="line")
plt.title("Study Hours vs Average Score")
plt.xlabel("Study Hours")
plt.ylabel("Average Score")
plt.show()

# 5 Box Plot for Score Distribution
plt.boxplot(df["Total_Score"])
plt.title("Total Score Distribution")
plt.show()

# 6 Correlation Heatmap
plt.imshow(df.corr(numeric_only=True))
plt.colorbar()
plt.title("Correlation Matrix")
plt.show()