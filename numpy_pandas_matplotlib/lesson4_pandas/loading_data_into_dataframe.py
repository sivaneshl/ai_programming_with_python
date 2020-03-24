import pandas as pd

df =  pd.read_csv('goog-1.csv')
print(df)
print(df.head())
print(df.tail())
print(df.head(7))

# check any cols have nan values
print(df.isnull().any())

# stats
print(df.describe())
print(df['Adj Close'].describe())

# col wise stats
print(df.max())
print(df.mean())
print(df['Close'].min())

# correlation
print(df.corr())

df = pd.DataFrame({'Year': [1990, 1990, 1990, 1991, 1991, 1991, 1992, 1992, 1992],
                   'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie'],
                   'Department': ['HR', 'RD', 'Admin', 'HR', 'RD', 'Admin', 'Admin', 'RD', 'Admin'],
                   'Age': [25, 30, 45, 26, 31, 46, 27, 32, 47],
                   'Salary': [50000, 48000, 55000, 52000, 50000, 60000, 60000, 52000, 62000]})

# calculate how much money the company spent each year
print(df.groupby(['Year'])['Salary'].sum())

# calculate how much avg salary each year
print(df.groupby(['Year'])['Salary'].mean())

# each employee got in 3 years
print(df.groupby(['Name'])['Salary'].sum())

# by year and dept
print(df.groupby(['Year', 'Department'])['Salary'].sum())
