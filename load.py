import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df=pd.read_csv('D:\ML\placement_predict_50k Dataset.csv')
print(df.head())
print("===============================================================================")
print(df.info())
print("===============================================================================")
print(df.describe())
print("===============================================================================")
print(df.shape)
print("===============================================================================")
print(df.isnull().sum())
print("===============================================================================")
print(df.tail())
print("===============================================================================")


plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x='CGPA',
    y='Salary Package'
)

plt.title('Salary Package vs CGPA')
plt.xlabel('CGPA')
plt.ylabel('Salary Package')
plt.show()