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

print("===============================================================================")

target = 'PlacementStatus'
features = ['CGPA', 'Internships', 'Projects', 'CodingTestScore', 'AttendancePercent', 'MockInterviewScore', 'Salary Package']

X = df[features]
y = df[target]

print(X.shape, y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

print("===============================================================================")

# Mean
mean = df['CGPA'].mean()
print("Mean:", mean)

# Median
median = df['CGPA'].median()
print("Median:", median)

# Mode
mode = df['CGPA'].mode()
print("Mode:", mode.tolist())

print("===============================================================================")

df.dtypes

num_cols = df.select_dtypes(include='number').columns
obj_cols = df.select_dtypes(include=['object']).columns

df['CGPA_Category'] = pd.cut(df['CGPA'], bins=[0,6,8,10], labels=['low', 'medium', 'high'], ordered=True)
print(df[['CGPA', 'CGPA_Category']].head())

df['CGPA'].describe()
sns.histplot(df['CGPA'])
plt.show()

from scipy import stats
stats.ttest_ind(
    df[df.PlacementStatus=='Yes']['CGPA'],
    df[df.PlacementStatus=='No']['CGPA']
)

df['CGPA'].max() - df['CGPA'].min()

df['CGPA'].var()
df['CGPA'].std()
q1 , q3 = df['CGPA'].quantile([0.25 , 0.75])
iqr = q3 - q1
print("\nSummary Statistics:")
print(df['CGPA'].describe())

print("===============================================================================")

