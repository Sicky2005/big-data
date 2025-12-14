#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
df = pd.read_excel('data/UsersSmall.xlsx')
#%%
print(df.head())
#%%
print(df.describe())
#%%
df.replace("?", np.nan, inplace=True)
#%%
print(df.isnull().sum())
#%%
for col in df.columns:
    if df[col].dtype == 'object':
        moda = df[col].mode()
        if not moda.empty:
            df[col] = df[col].fillna(moda[0])
    else:
        df[col] = df[col].fillna(df[col].mean())
#%%
print(df.isnull().sum())
#%%
sns.histplot(df['Age'], kde=True)
plt.title("Distribuzione dell'età")
plt.show()
#%%
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]
print(outliers[['Age']])
#%%
print(outliers.groupby('Response').size())
print(outliers.groupby('Workclass').size())
#%%
sns.histplot(df['Age'], kde=True)
plt.axvline(lower_bound, color='red', linestyle='--', label='Limite inferiore')
plt.axvline(upper_bound, color='red', linestyle='--', label='Limite superiore')
plt.legend()
plt.title("Distribuzione età con outlier evidenziati")
plt.savefig("grafico_age_kde.png", dpi=300, bbox_inches='tight')
plt.show()
#%%
# Discretize by Equal-Width Binning
df['Age_binned'] = pd.cut(df['Age'], bins=5)

plt.figure(figsize=(8, 5))
sns.countplot(x='Age_binned', data=df)
plt.title("Equal-width binning (5 fasce)")
plt.xlabel("Fasce di età (ampiezza costante)")
plt.ylabel("Numero di utenti")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafico_equal_width.png", dpi=300, bbox_inches='tight')
plt.show()
#%%
# Discretize by Equal-Frequency Binning
df['Age_freq'] = pd.qcut(df['Age'], q=5)

plt.figure(figsize=(8, 5))
sns.countplot(x='Age_freq', data=df)
plt.title("Equal-frequency binning (5 fasce)")
plt.xlabel("Fasce di età (frequenza costante)")
plt.ylabel("Numero di utenti")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafico_equal_freq.png", dpi=300, bbox_inches='tight')
plt.show()
#%%
# Discretize by Size
bins = [0, 18, 25, 35, 50, 65, 100, df['Age'].max()]
labels = ['Minorenne', 'Giovane adulto', 'Adulto giovane', 'Adulto maturo', 'Pre-pensionamento', 'Anziano', 'Estremamente anziano']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(10, 5))
sns.countplot(x='Age_group', data=df, order=labels)
plt.title("Discretizzazione semantica dell’età")
plt.xlabel("Fasce semantiche")
plt.ylabel("Numero di utenti")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("grafico_semantico.png", dpi=300, bbox_inches='tight')
plt.show()