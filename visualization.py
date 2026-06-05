import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

survival = df['Survived'].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    survival,
    labels=['Died','Survived'],
    autopct='%1.1f%%'
)

plt.title("Passenger Survival Percentage")

plt.savefig("pie_chart.png")
plt.show()

plt.figure(figsize=(6,5))

sns.countplot(
    x='Sex',
    data=df
)

plt.title("Gender Distribution")

plt.savefig("bar_chart.png")
plt.show()

fare_by_class = df.groupby('Pclass')['Fare'].mean()

plt.figure(figsize=(6,5))

plt.plot(
    fare_by_class.index,
    fare_by_class.values,
    marker='o'
)

plt.title("Average Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare")

plt.savefig("line_chart.png")
plt.show()

plt.figure(figsize=(7,5))

sns.boxplot(
    x='Pclass',
    y='Fare',
    data=df
)

plt.title("Fare Distribution by Passenger Class")

plt.savefig("boxplot.png")
plt.show()

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig("heatmap.png")
plt.show()

plt.figure(figsize=(7,5))

sns.scatterplot(
    x='Age',
    y='Fare',
    hue='Survived',
    data=df
)

plt.title("Age vs Fare")

plt.savefig("scatter_plot.png")
plt.show()