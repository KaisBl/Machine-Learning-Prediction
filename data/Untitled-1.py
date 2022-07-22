# %%
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# %%
data_train = pd.read_csv("train.csv", sep=';')
df = data_train.copy(deep=True)
data_data = pd.read_csv("train.csv", sep=';')


# %%
print(data_train)

# %%
# checking for null values
data_train. isnull(). sum()

# %%
data_train.describe()

# %%
data_train.info()

# %%
data_train["age_cat"] = pd.cut(data_train["age"], bins=8, labels=[
                               "18 - 27", "28 - 37", "38 - 46", "47 - 56", "57 - 66", "67 - 75", "76 - 85", "86 - 95"])

# %%
data_train["duration_min"] = data_train["duration"]/60

# %%
bins = [0, 0.50, 1, 2, 3, 4, 5, 10, 15, 30, 60, 90]
category = ["0 s <30s", "30s-1min", "1 min-2min", "2 min-3min", "3 min-4min", "4 min-5min",
    "5 min-10min", "10 min-15min", "15 min-30min", "30 min-60min", "60 min-90min"]
data_train['duration_cat'] = pd.cut(
    data_train['duration_min'], bins, labels=category)
print(data_train)

# %%
bins_1 = [-10000, -5000, -2500, -1000, -500, -100, 0, 100, 250,
    500, 1000, 1500, 2500, 5000, 10000, 20000, 50000, 110000]
category_1 = ["-10000 <balance<-5000", "-5000 <balance<-2500", "-2500 <balance<-1000", "-1000 <balance<-500", "-500 <balance<-100", "-100 -0", "0 - 100",
    "100 - 250", "250 - 500", "500 - 1000", "1000 - 1500", "1500 - 2500", "2500 - 5000", "5000 - 10000", "10000 - 20000", "20000 - 50000", "50000 - 110000"]
data_train['balance_cat'] = pd.cut(
    data_train['balance'], bins_1, labels=category_1)
print(data_train)

# %%
data_train['day_cat'] = data_train['day'].astype(str)

# %%
print(data_train.info())

# %%
bins_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 65]
category_2 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10 - 15",
    "15 - 20", "20 - 25", "25 - 30", "30 - 35", "35 - 40", "40 - 50", "50 - 65"]
data_train['campaign_cat'] = pd.cut(
    data_train['campaign'], bins_2, labels=category_2)
print(data_train)

# %%
bins_3 = [-2, -1, 50, 100, 150, 200, 250,
    300, 350, 400, 450, 500, 550, 650, 880]
category_3 = ["npc", "0 - 50", "50 - 100", "100 - 150", "150 - 200", "200 - 250", "250 - 300",
    "300 - 350", "350 - 400", "400 - 450", "450 - 500", "500 - 550", "550 - 650", "650 - 880"]
data_train['pdays_cat'] = pd.cut(
    data_train['pdays'], bins_3, labels=category_3)
print(data_train)

# %%
data_train["pdays"].describe()

# %%
bins_4 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
category_4 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "10 - 15", "15 - 20", "20 - 25", "25 - 30", "30 - 35", "35 - 40"]
data_train['previous_cat'] = pd.cut(
    data_train['previous'], bins_4, labels=category_4)
print(data_train)

# %%
print(data_train["previous"].describe())

# %%


def cat(ds, c):
    """Visualising categorical columns"""
    for i in c:
        print('Information about', i)
        print(round((ds[i].value_counts(normalize=True) * 100), 2))
        sns.catplot(x=i, kind='count', data=ds, aspect=8,
                    height=2, order=ds[i].value_counts().index)
        plt.show()


# %%
cat(ds=data_train, c=['age_cat', 'job', 'marital', 'education', 'default', 'balance_cat', 'housing', 'loan',
    'contact', 'day_cat', 'month', 'duration_cat', 'campaign_cat', 'pdays_cat', 'previous_cat', 'poutcome', 'y'])

# %%
# we notice the exitence of unknown data.
# we will treat unknown data:
# unknown values exists in : poutcome, contact, education and job columns
# these unknown values have to be replaced
# job column has 288 rows of unknown:we will ignore them as it is very small compared to our dataset
# poutcome: we will drop this column as most of the data contained is unknown
# Unknown values in Education and Contact will be replaced by mode

# %% [markdown]
#
# Treating missing data:
#

# %%
# matching missing value from job with education mode and the unkown will be dropped
df[df['job'] == 'unknown']['education'].value_counts()

# %%


# %%
data = df[df['job'] != 'unknown']


# %%
# data.drop('poutcome', axis = 1, inplace = True)
print(data.columns)
print(data_train.columns)


# %%
data['education'].replace("unknown", data['education'].mode()[0], inplace=True)

# %%
data['contact'].replace("unknown", data['contact'].mode()[0], inplace=True)

# %%
data.drop("poutcome", axis=1, inplace=True)

# %%
print(data)

# %%
data[data['job'] == 'unknown']['education'].value_counts()

# %%


# %% [markdown]
# Checking if "unknown" data is left in the dataset

# %%
cat(ds=data, c=['age_cat', 'job', 'marital', 'education', 'default', 'balance_cat', 'housing', 'loan',
    'contact', 'day_cat', 'month', 'duration_cat', 'campaign_cat', 'pdays_cat', 'previous_cat', 'y'])

# %% [markdown]
# End of missing value handling

# %%
plt.figure(figsize=(20, 10))
sns.heatmap(data.corr(), annot=True)

# %%
data.describe()

# %%
data[['age', 'balance', 'day', 'duration', 'campaign', 'pdays',
    'previous']].hist(bins=15, figsize=(15, 6), layout=(2, 4))

# %%
print(data)

# %%
# dropping poutcome column as it is mainly constituted of unknown
# data.drop("poutcome", axis= 1, inplace=True)

# %% [markdown]
# PCA

# %%
pca_df = data.copy(deep=True)
# columns_to_transform = ['age', 'balance', 'day', 'duration', 'pdays']
# object_col = pca_df.select_dtypes('object').columns
# object_col

# %%
binary_columns = []
for column in pca_df.select_dtypes('object').columns:
    if len(pca_df[column].unique()) == 2:
        binary_columns.append(column)

    print(f"Column - {column} ", pca_df[column].unique(), end='\n\n')

binary_columns.remove("contact")
print(binary_columns)

# %%
for column in binary_columns:
    pca_df[column] = pca_df[column].map({'yes': 1, 'no': 0})

pca_df.loc[:, binary_columns]

# %%
print(pca_df.columns)
print(pca_df)

# %%
# normalizer = preprocessing.Normalizer().fit(X_train)
# X_train = normalizer.transform(X_train)
# X_test = normalizer.transform(X_test)

# %%
# Create train_x dataframe
pca_data = pca_df.iloc[:, :-1]


# pca_data.drop("poutcome", axis=1, inplace=True)

# %%
target = pca_df[['y']]
print(target)

# %%
pca_data.head()

# %%
# Perform label encoding on education as it is an ordinal data
education_category = {
          'primary': 1,
          'secondary': 2,
          'tertiary': 3}
pca_data['education'] = pca_data['education'].replace(education_category)
pca_data['education'].value_counts()

# %%
# train_data['contact'] = labelencoder.fit_transform(train_data['contact'])
# train_data

# %%
# Get a list of columns for one-hot encoding
ohe_cols = list(pca_data.select_dtypes(include='object').columns.values)

# We want to label encode education
# le_col = ['education']

# Drop education
# ohe_cols.remove('education')
ohe_cols

pca_data_x = pd.get_dummies(
    pca_data, prefix=ohe_cols, columns=ohe_cols, drop_first=True)
pca_data_x.head()

# %%
pca_col = pca_data_x
print(pca_col)

# %%


# %%
normalizer = preprocessing.Normalizer().fit(pca_data_x)
pca_data_x = normalizer.transform(pca_data_x)
print(pca_data_x)

# %% [markdown]
# PCA

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(pca_data_x)

print(X_pca)
print(len(X_pca))

# %%
pca.explained_variance_ratio_
# pc1 explains 27.4%
# pc2 explains 24.5%
sns.scatterplot(data=X_pca, x='principal_component_1',
                y='principal_component_2', hue='target')
plt.title('2 component PCA')
plt.show()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# %%
# Loop Function to identify number of principal components that explain at least 85% of the variance
for comp in range(3, pca_data_x.shape[1]):
    pca = PCA(n_components=comp, random_state=42)
    pca.fit(pca_data_x)
    comp_check = pca.explained_variance_ratio_
    final_comp = comp
    if comp_check.sum() > 0.85:
        break

Final_PCA = PCA(n_components=final_comp, random_state=42)
Final_PCA.fit(pca_data_x)
cluster_df = Final_PCA.transform(pca_data_x)
num_comps = comp_check.shape[0]
print("Using {} components, we can explain {}% of the variability in the original data.".format(
    final_comp, comp_check.sum()))

# %%
pca1 = PCA(n_components=5)
principalComponents1 = pca1.fit_transform(pca_data_x)
principalDf1 = pd.DataFrame(data=principalComponents1, columns=[
                            'principal_component_1', 'principal_component_2', 'principal_component_3', 'principal_component_4', 'principal_component_5'])
# principalDf1.index = pca_ds["target"].index

# %%
principalDf1.index = target.index

# %%
final_Df = pd.concat([principalDf1, target], axis=1)
print(final_Df)

# %%
pca1.explained_variance_ratio_
# pc1 explains 27.4%
# pc2 explains 24.5%


# %%
sns.scatterplot(data=final_Df, x='principal_component_1',
                y='principal_component_2', hue='y')
plt.title('2 component PCA')
plt.show()

# %%
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# %%
loadings = pd.DataFrame(pca1.components_.T, columns=[
                        'PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=pca_col.columns)
loadings


# %%
p

# %%
print(pca_data_x)

# %%
df[df['job'] == 'unknown']['education'].value_counts()
data = df[df['job'] != 'unknown']
data['education'].replace("unknown", data['education'].mode()[0], inplace=True)
data['contact'].replace("unknown", data['contact'].mode()[0], inplace=True)

# %%

# binary_columns.remove("contact")
print(binary_columns)
print(pca_train["campaign_cat"].unique())
print(pca_train["pdays_cat"].unique())

# %%
train_df = data.rename(columns={'y': 'target'})
train_df
print(train_df)

# %%
print(train_df.columns)
train_data = train_df.copy(deep=True)
pca_train = train_df.copy(deep=True)

# %%
labelencoder = LabelEncoder()


# %%
print(pca_train_f)

# %%


# %%
print(train_data.columns)
print(train_df.columns)

# %%
# label encoding

# %%
labelencoder = LabelEncoder()
train_data['job'] = labelencoder.fit_transform(train_data['job'])
train_data

# %%
train_data['marital'] = labelencoder.fit_transform(train_data['marital'])
train_data

# %%
train_data['contact'] = labelencoder.fit_transform(train_data['contact'])
train_data

# %%
train_data.info()

# %%
train_data['education'] = labelencoder.fit_transform(train_data['education'])
train_data

# %%
train_data['month'] = labelencoder.fit_transform(train_data['month'])
train_data

# %%
train_data.info()

# %%
pca_ds = train_data.copy(deep=True)

# %%
pca_ds.drop("duration_cat", axis=1, inplace=True)
pca_ds.drop("age_cat", axis=1, inplace=True)
pca_ds.drop("duration_min", axis=1, inplace=True)

pca_ds.drop("balance_cat", axis=1, inplace=True)
pca_ds.drop("day_cat", axis=1, inplace=True)
pca_ds.drop("campaign_cat", axis=1, inplace=True)

pca_ds.drop("pdays_cat", axis=1, inplace=True)
pca_ds.drop("previous_cat", axis=1, inplace=True)
pca_ds.drop("campaign_cat", axis=1, inplace=True)

# %%
pca_ds.info()

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(pca_ds)

print(X_pca)
print(len(X_pca))


# %%
print(pca_ds)

# %%
pca_dddd = train_df.copy(deep=True)
# columns_to_transform = ['age', 'balance', 'day', 'duration', 'pdays']
# object_col = pca_df.select_dtypes('object').columns
# object_col

# %%
pca_dddd.drop("duration_cat", axis=1, inplace=True)
pca_dddd.drop("age_cat", axis=1, inplace=True)
pca_dddd.drop("duration_min", axis=1, inplace=True)

pca_dddd.drop("balance_cat", axis=1, inplace=True)
pca_dddd.drop("day_cat", axis=1, inplace=True)
pca_dddd.drop("campaign_cat", axis=1, inplace=True)

pca_dddd.drop("pdays_cat", axis=1, inplace=True)
pca_dddd.drop("previous_cat", axis=1, inplace=True)
# pca_dddd.drop("campaign_cat", axis = 1, inplace=True)

# %%
print(pca_dddd)

# %%

# %%
columns_to_transform = ['age', 'balance', 'day', 'duration', 'pdays']

# %%
pca_ds.loc[:, columns_to_transform] = StandardScaler().fit_transform(
    pca_ds.loc[:, columns_to_transform].values)

# %%
print(pca_ds)

# %% [markdown]
# Feature Extraction

# %%

# %%
pca = PCA(n_components=2)

# %%
x = pca_ds.iloc[:, :-1]
y = pca_ds.iloc[:, -1]

# %%
print(x)

# %%
print(y)

# %%
print(pca_ds["target"])

# %%
principalComponents = pca.fit_transform(x)

# %%
principalDf = pd.DataFrame(data=principalComponents, columns=[
                           'principal_component_1', 'principal_component_2'])

# %%
print(principalDf)

# %%
print(pca_ds["target"].value_counts())

# %%
pca_ds["target"].isnull().sum()

# %%
y.isnull().sum()

# %%
principalDf.index = pca_ds["target"].index

# %%


# %%


# %%
finalDf = pd.concat([principalDf, pca_ds["target"]], axis=1)

# %%
finalDf.isnull().sum()

# %%
print(finalDf["target"].value_counts())

# %%
finalDf["target"].isnull().sum()

# %%
 # df=finalDf.dropna(subset=['target'])

# %%
print(finalDf)

# %%
finalDf.isnull().sum()

# %%
 # df1=df.dropna(subset=['principal_component_2', 'principal_component_1'])
 # print(df1)

# %%
# print(df1["target"].value_counts())

# %%
sns.scatterplot(data=finalDf, x='principal_component_1',
                y='principal_component_2', hue='target')
plt.title('2 component PCA')
plt.show()

# %%
pca.explained_variance_ratio_
# pc1 explains 27.4%
# pc2 explains 24.5%


# %%
pca1 = PCA(n_components=5)
principalComponents1 = pca1.fit_transform(x)
principalDf1 = pd.DataFrame(data=principalComponents1, columns=[
                            'principal_component_1', 'principal_component_2', 'principal_component_3', 'principal_component_4', 'principal_component_5'])


# %%
principalDf1.index = pca_ds["target"].index

# %%
print(principalDf1)

# %%
finalDf1 = pd.concat([principalDf1, pca_ds["target"]], axis=1)

# %%
pca1.explained_variance_ratio_
# PC1 + PC2 + PC3 + PC4 = 0.85443839
# PC1 + PC2 + PC3 + PC4  together explain about 85% of the variance of the data

# %%
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# %%


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# plot data
plt.scatter(x[:, 0], x[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')

# %%
loadings = pd.DataFrame(pca1.components_.T, columns=[
                        'PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=x.columns)
loadings


# %%
 # Each element represents a loading, namely how much (the weight) each original variable contributes to the corresponding principal component.
 # PC1 is mainly composed of job and campaign
 # PC2 is mainly composed of campaign
 # PC3 is mainly composed of month, campaign and job
 # PC4 is mainly composed of previous and pdays

# %%


# %%
loadings = pca1.components_.T * np.sqrt(pca1.explained_variance_)

loading_matrix = pd.DataFrame(
    loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=x.columns)
loading_matrix

# %%


# %% [markdown]
# Forward Feature Selection

# %%

# %%
# Sequential Forward Selection(sfs)
sfs = SFS(LinearRegression(),
          k_features=(1, 5),
          forward=True,
          floating=False,
          scoring='r2',
          cv=0)

# %%
sfs.fit(pca_col, target)
sfs.k_feature_names_

# %%
print(train_df)

# %%
print(train_df.info())

# %%
train_df.describe()

# %%
columns_to_transform_1 = ["age_cat", "duration_cat",
    "balance_cat", "campaign_cat", "pdays_cat", "previous_cat"]

# %%
train_ds.loc[:, columns_to_transform_1] = StandardScaler().fit_transform(
    train_ds.loc[:, columns_to_transform_1].values)

# %%
columns_to_transform_1 = ["age_cat", "duration_cat",
    "balance_cat", "campaign_cat", "pdays_cat", "previous_cat"]

# %%
# applying label encoder on ordinal data
labelencoder = LabelEncoder()
train_df['age_cat'] = labelencoder.fit_transform(train_df["age_cat"])
train_df

# %%
train_df['duration_cat'] = labelencoder.fit_transform(train_df["duration_cat"])
train_df

# %%
labelencoder.inverse_transform(train_ds["month"])

# %%
train_ds['job'] = labelencoder.inverse_transform(train_ds['job'])
train_ds

# %%
print(train_ds)

# %%
"marital", "education", "contact", "month"

# %%
print(data_train)

# %%
print(data)

# %%
print(train_data)

# %%
print(df)

# %%
df["age_cat"] = pd.cut(df["age"], bins=8, labels=[
                       "18 - 27", "28 - 37", "38 - 46", "47 - 56", "57 - 66", "67 - 75", "76 - 85", "86 - 95"])

# %%
df["duration_min"] = df["duration"]/60

# %%
bins = [0, 0.50, 1, 2, 3, 4, 5, 10, 15, 30, 60, 90]
category = ["0 s <30s", "30s-1min", "1 min-2min", "2 min-3min", "3 min-4min", "4 min-5min",
    "5 min-10min", "10 min-15min", "15 min-30min", "30 min-60min", "60 min-90min"]
df['duration_cat'] = pd.cut(df['duration_min'], bins, labels=category)
print(df)

# %%
bins_1 = [-10000, -5000, -2500, -1000, -500, -100, 0, 100, 250,
    500, 1000, 1500, 2500, 5000, 10000, 20000, 50000, 110000]
category_1 = ["-10000 <balance<-5000", "-5000 <balance<-2500", "-2500 <balance<-1000", "-1000 <balance<-500", "-500 <balance<-100", "-100 -0", "0 - 100",
    "100 - 250", "250 - 500", "500 - 1000", "1000 - 1500", "1500 - 2500", "2500 - 5000", "5000 - 10000", "10000 - 20000", "20000 - 50000", "50000 - 110000"]
df['balance_cat'] = pd.cut(df['balance'], bins_1, labels=category_1)
print(df)

# %%
bins_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 65]
category_2 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10 - 15",
    "15 - 20", "20 - 25", "25 - 30", "30 - 35", "35 - 40", "40 - 50", "50 - 65"]
df['campaign_cat'] = pd.cut(df['campaign'], bins_2, labels=category_2)
print(df)

# %%
bins_3 = [-2, -1, 50, 100, 150, 200, 250,
    300, 350, 400, 450, 500, 550, 650, 880]
category_3 = ["npc", "0 - 50", "50 - 100", "100 - 150", "150 - 200", "200 - 250", "250 - 300",
    "300 - 350", "350 - 400", "400 - 450", "450 - 500", "500 - 550", "550 - 650", "650 - 880"]
df['pdays_cat'] = pd.cut(df['pdays'], bins_3, labels=category_3)
print(df)

# %%
bins_4 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
category_4 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "10 - 15", "15 - 20", "20 - 25", "25 - 30", "30 - 35", "35 - 40"]
df['previous_cat'] = pd.cut(df['previous'], bins_4, labels=category_4)
print(df)

# %%
df[df['job'] == 'unknown']['education'].value_counts()

# %%
dff = df[df['job'] != 'unknown']

# %%
dff['education'].replace("unknown", dff['education'].mode()[0], inplace=True)

# %%
dff['contact'].replace("unknown", dff['contact'].mode()[0], inplace=True)

# %%
print(dff)

# %%
dff.drop('poutcome', axis=1, inplace=True)

# %%
print(dff.columns)

# %%
dff.drop('duration_min', axis=1, inplace=True)

# %%
print(dff)

# %%
dff["y"] = dff["y"].map({'yes': 1, 'no': 0})

# %%
print(dff)

# %%
dff = dff.rename(columns={'y': 'target'})

# %%
print(dff)

# %%
print(dff.columns)

# %%
x = pca_ds.iloc[:, :-1]
y = pca_ds.iloc[:, -1]

# %%
y = dff.iloc[:, 15]

# %%
print(dff[dff.columns.difference(["target"])])
# df[df.columns.difference(['b'])]

# %%
x = dff[dff.columns.difference(["target"])]

# %%
print(x)

# %%
x.drop('age', axis=1, inplace=False)
x.drop('balance', axis=1, inplace=False)
x.drop('campaign', axis=1, inplace=False)
x.drop('duration', axis=1, inplace=False)
x.drop('day', axis=1, inplace=False)
x.drop('pdays', axis=1, inplace=False)
x.drop('previous', axis=1, inplace=False)


# %%
print(x)

# %%
x.drop('age', axis=1, inplace=True)
x.drop('balance', axis=1, inplace=True)
x.drop('campaign', axis=1, inplace=True)
x.drop('duration', axis=1, inplace=True)
x.drop('day', axis=1, inplace=True)
x.drop('pdays', axis=1, inplace=True)
x.drop('previous', axis=1, inplace=True)

# %%
print(x)

# %%


def label_encoder(x, column):
    for i in x:

        ohe = preprocessing.OneHotEncoder()
        temp_array = ohe.fit_transform(x[[column]]).toarray()
        column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(temp_array, columns=column_names))


for i in x:

x_encoded = pd.DataFrame(encoder.fit_transform(x).toarray())

# %%
print(x_encoded)

# %%
x = x.reset_index(drop=True)


def label_encoder(x, column):
    le = preprocessing.LabelEncoder()
    le.fit_transform(x[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(x[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(temp_array, columns=column_names))


# Got this wise bit of code from the amazing 'Andrada Olteanu'.
numerical_variables = [
    col for col in dfg.columns if dfg[col].dtype in ['int64', 'float64']]

print(numerical_variables)

# Got this wise bit of code from the amazing 'Andrada Olteanu'.
categorical_variables = [
    col for col in dfg.columns if dfg[col].dtype == 'object']

new_df = dfg[numerical_variables]

for column in categorical_variables:
    new_df = pd.concat([new_df, label_encoder(dfg, column)], axis=1)

# Label Encoder Target

le = preprocessing.LabelEncoder()
le.fit(a)

new_df['TARGET'] = le.transform(a)

new_df.columns

# new_df is the TRAINING dataframe encoded


# Link to Reference code from Andrada Olteanu: https://www.kaggle.com/andradaolteanu/housing-prices-competition-iowa-dataset

# %%
train_dataset = x.join(y)

# %%
train_dataset = train_dataset.reset_index(drop=True)


def label_encoder(train_dataset, column):
    le = preprocessing.LabelEncoder()
    le.fit_transform(train_dataset[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(train_dataset[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(temp_array, columns=column_names))


numerical_variables = [col for col in train_dataset.columns if train_dataset[col].dtype in [
    'int64', 'float64']]  # Got this wise bit of code from the amazing 'Andrada Olteanu'.

print(numerical_variables)

# Got this wise bit of code from the amazing 'Andrada Olteanu'.
categorical_variables = [
    col for col in train_dataset.columns if train_dataset[col].dtype == 'object']

new_df = train_dataset[numerical_variables]

for column in categorical_variables:
    new_df = pd.concat([new_df, label_encoder(train_dataset, column)], axis=1)

# Label Encoder Target

le = preprocessing.LabelEncoder()
le.fit(a)

new_df['TARGET'] = le.transform(a)

new_df.columns

# new_df is the TRAINING dataframe encoded


# Link to Reference code from Andrada Olteanu: https://www.kaggle.com/andradaolteanu/housing-prices-competition-iowa-dataset

# %%
numerical_variables_12 = [
    col for col in dff.columns if dff[col].dtype in ['int64', 'float64']]
print(numerical_variables_12)

# %%
x = x.reset_index(drop=True)


def label_encoder(x, column):
    le = preprocessing.LabelEncoder()
    le.fit_transform(x[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(x[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(temp_array, columns=column_names))


print(x.columns)

# new_df is the TRAINING dataframe encoded


# Link to Reference code from Andrada Olteanu: https://www.kaggle.com/andradaolteanu/housing-prices-competition-iowa-dataset

# %%
# a= dfg['TARGET']
x = x.reset_index(drop=True)


def label_encoder(x, column):
    le = preprocessing.LabelEncoder()
    le.fit_transform(x[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(x[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(temp_array, columns=column_names))


# Got this wise bit of code from the amazing 'Andrada Olteanu'.
all_variables = [col for col in x.columns]

print(all_variables)

new_df = x[all_variables]

for column in all_variables:
    new_df = pd.concat([new_df, label_encoder(x, column)], axis=1)


new_df.columns

# new_df is the TRAINING dataframe encoded


# Link to Reference code from Andrada Olteanu: https://www.kaggle.com/andradaolteanu/housing-prices-competition-iowa-dataset

# %%
df = pd.get_dummies(x, columns=["age_cat", "balance_cat", " campaign_cat", "duration_cat", "pdays_cat", "previous_cat",
                    'job', 'marital', 'education', 'default', 'housing', 'month', 'loan', 'contact'], drop_first=True)
df.head()

# %%
print(x.columns)

# %%
x.shape
x.columns

# %%
print(x)

# %%
print(y)

# %%
le = preprocessing.LabelEncoder()

# %%
X_2 = x.apply(le.fit_transform)

# %%
X_2.head()

# %%
enc = preprocessing.OneHotEncoder()

# %%
ohenc = OneHotEncoder(sparse=False)
x_cat_df = pd.DataFrame(ohenc.fit_transform(xtrain_lbl))
x_cat_df.columns = ohenc.get_feature_names_out(
    input_features=xtrain_lbl.columns)

# %%
enc.get_feature_names_out()
x_cat_df.columns = enc.get_feature_names_out()

# %%
ohenc = OneHotEncoder(use_cat_names=True)

# %%
ohenc = OneHotEncoder(sparse=False)
x_cat_df = pd.DataFrame(ohenc.fit_transform(x))
x_cat_df.columns = ohenc.get_feature_names_out(input_features=x.columns)

# %%
print(x_cat_df.columns)

# %%
X_2.drop('age', axis=1, inplace=True)
X_2.drop('balance', axis=1, inplace=True)
X_2.drop('campaign', axis=1, inplace=True)
X_2.drop('duration', axis=1, inplace=True)
X_2.drop('day', axis=1, inplace=True)
X_2.drop('pdays', axis=1, inplace=True)
X_2.drop('previous', axis=1, inplace=True)

# %%
print(X_2.info())

# %%
ohenc = OneHotEncoder(sparse=False)
x_cat_df = pd.DataFrame(ohenc.fit_transform(X_2))
x_cat_df.columns = ohenc.get_feature_names_out(input_features=X_2.columns)

# %%
print(x_cat_df)

# %%
X_1 = x.apply(le.fit_transform)

# %%
X_1.drop('age', axis=1, inplace=True)
X_1.drop('balance', axis=1, inplace=True)
X_1.drop('campaign', axis=1, inplace=True)
X_1.drop('duration', axis=1, inplace=True)
X_1.drop('day', axis=1, inplace=True)
X_1.drop('pdays', axis=1, inplace=True)
X_1.drop('previous', axis=1, inplace=True)

# %%
X_1.head

# %%
column_names = [X_1[column]+"_"+str(m) for m in le.classes_]

# %%
for m in le.classes_:
    print(m)


# %%

# %%
ohe = OneHotEncoder(sparse=False)

# %%
print(x)

# %%
x.drop('age', axis=1, inplace=True)
x.drop('balance', axis=1, inplace=True)
x.drop('campaign', axis=1, inplace=True)
x.drop('duration', axis=1, inplace=True)
x.drop('day', axis=1, inplace=True)
x.drop('pdays', axis=1, inplace=True)
x.drop('previous', axis=1, inplace=True)

# %%
print(x)

# %%
ohe.fit_transform(x)

# %%
feature_arry = ohe.fit_transform(x)

# %%
print(feature_arry)

# %%
ohe.categories_

# %%
x[x.isnull().any(axis=1)]

# %%
x = x.dropna(axis=0, how='all')

# %%
x[x.isnull().any(axis=1)]

# %%
x.isnull().sum()

# %%
x.dropna(inplace=True)

# %%
x.isnull().sum()

# %%
feature_labels = ohe.categories_

# %%
print(ohe.categories_)

# %%
print(np.array(feature_labels).ravel())

# %%
feature_labels = np.array(f).ravel()

# %%
feature_label = feature_labels.tolist()

# %%
print(feature_label)

# %%
data_tt = pd.DataFrame(feature_arry, columns=['18 - 27', '28 - 37', '38 - 46', '47 - 56', '57 - 66', '67 - 75',
       '76 - 85', '86 - 95', '-100 -0', '-1000 <balance<-500', '-10000 <balance<-5000',
       '-2500 <balance<-1000', '-500 <balance<-100',
       '-5000 <balance<-2500', '0 - 100', '100 - 250', '1000 - 1500',
       '10000 - 20000', '1500 - 2500', '20000 - 50000', '250 - 500',
       '2500 - 5000', '500 - 1000', '5000 - 10000', '50000 - 110000', '1', '10', '10 - 15', '15 - 20', '2', '20 - 25', '25 - 30', '3',
       '30 - 35', '35 - 40', '4', '40 - 50', '5', '50 - 65', '6', '7',
       '8', '9', 'cellular', 'telephone', 'no', 'yes', '0 s <30s', '1 min-2min', '10 min-15min', '15 min-30min',
       '2 min-3min', '3 min-4min', '30 min-60min', '30s-1min',
       '4 min-5min', '5 min-10min', '60 min-90min', 'primary', 'secondary', 'tertiary', 'no', 'yes', 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'unemployed', 'no', 'yes', 'divorced', 'married', 'single', 'apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may',
       'nov', 'oct', 'sep', '0 - 50', '100 - 150', '150 - 200', '200 - 250', '250 - 300',
       '300 - 350', '350 - 400', '400 - 450', '450 - 500', '50 - 100',
       '500 - 550', '550 - 650', '650 - 880', 'npc', '0', '1', '10', '10 - 15', '15 - 20', '2', '20 - 25', '25 - 30',
       '3', '30 - 35', '35 - 40', '4', '5', '6', '7', '8', '9'])

# %%
print(feature_arry.shape)

# %%
print(len(feature_label))

# %%
columns = ['18 - 27', '28 - 37', '38 - 46', '47 - 56', '57 - 66', '67 - 75',
       '76 - 85', '86 - 95', '-100 -0', '-1000 <balance<-500', '-10000 <balance<-5000',
       '-2500 <balance<-1000', '-500 <balance<-100',
       '-5000 <balance<-2500', '0 - 100', '100 - 250', '1000 - 1500',
       '10000 - 20000', '1500 - 2500', '20000 - 50000', '250 - 500',
       '2500 - 5000', '500 - 1000', '5000 - 10000', '50000 - 110000', '1', '10', '10 - 15', '15 - 20', '2', '20 - 25', '25 - 30', '3',
       '30 - 35', '35 - 40', '4', '40 - 50', '5', '50 - 65', '6', '7',
       '8', '9', 'cellular', 'telephone', 'no', 'yes', '0 s <30s', '1 min-2min', '10 min-15min', '15 min-30min',
       '2 min-3min', '3 min-4min', '30 min-60min', '30s-1min',
       '4 min-5min', '5 min-10min', '60 min-90min', 'primary', 'secondary', 'tertiary', 'no', 'yes', 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'unemployed', 'no', 'yes', 'divorced', 'married', 'single', 'apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may',
       'nov', 'oct', 'sep', '0 - 50', '100 - 150', '150 - 200', '200 - 250', '250 - 300',
       '300 - 350', '350 - 400', '400 - 450', '450 - 500', '50 - 100',
       '500 - 550', '550 - 650', '650 - 880', 'npc', '0', '1', '10', '10 - 15', '15 - 20', '2', '20 - 25', '25 - 30',
       '3', '30 - 35', '35 - 40', '4', '5', '6', '7', '8', '9']

# %%
print(len(columns))

# %%
train_set = data_tt.join(y)

# %%

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    data_tt, y, test_size=0.25, random_state=20)

# %%
print(df)

# %%
print(data_data)

# %% [markdown]
# Exploratory Data Analysis

# %%
data_data['age'].describe()

# %%
# visualising age data

# %%
data_data['age'].plot.hist(bins=30, density=True)
plt.show()

# %%
print(data_data['job'].value_counts())

# %%
data_data['marital'].value_counts()

# %%
train_df['education'].value_counts()

# %%
data_data['default'].value_counts()

# %%
data_data['balance'].describe()

# %%
data_data['balance'].plot.hist(bins=50, density=True)
plt.show()

# %%
data_data['balance'].plot.box()
plt.show()

# %%
# some outliers are present in Balance column

# %%
data_data['housing'].value_counts()

# %%
data_data['loan'].value_counts()

# %%
data_data['contact'].value_counts()

# %%
data_data['day'].value_counts()

# %%
data_data['day'].value_counts().describe()

# %%
data_data['month'].value_counts()

# %%
# customers are less likely to be contacted in dec: might be explained by the fact that it is a holiday period

# %%
data_data['duration'].describe()

# %%
data_data['duration'].plot.hist(bins=50, density=True)
plt.show()

# %%
data_data['duration'].plot.box()
plt.show()

# %%
# we can notice the presence of outliers in the duration column
# there seems to be a duration of zero, which seems irrational
# there seems to be a duration of 5000 and 4000 which might be irrational

# %%
data_data['campaign'].describe()

# %%
data_data['campaign'].plot.hist(bins=30, density=True)
plt.show()

# %%
data_data['pdays'].describe()

# %%
data_data['previous'].value_counts()

# %%
# some values seems to be irrational like 275

# %%
data_1 = data_data.copy(deep=True)

# %%
data_1

# %%
data_1['day_cat'] = data_1['day'].astype(str)

# %%
data_1["age_cat"] = pd.cut(data_1["age"], bins=8, labels=[
                           "18 - 27", "28 - 37", "38 - 46", "47 - 56", "57 - 66", "67 - 75", "76 - 85", "86 - 95"])

# %%
data_1["duration_min"] = data_1["duration"]/60

# %%
bins = [0, 0.50, 1, 2, 3, 4, 5, 10, 15, 30, 60, 90]
category = ["0 s <30s", "30s-1min", "1 min-2min", "2 min-3min", "3 min-4min", "4 min-5min",
    "5 min-10min", "10 min-15min", "15 min-30min", "30 min-60min", "60 min-90min"]
data_1['duration_cat'] = pd.cut(data_1['duration_min'], bins, labels=category)
print(data_1)

# %%
print(data_data)

# %%
bins_1 = [-10000, -5000, -2500, -1000, -500, -100, 0, 100, 250,
    500, 1000, 1500, 2500, 5000, 10000, 20000, 50000, 110000]
category_1 = ["-10000 <balance<-5000", "-5000 <balance<-2500", "-2500 <balance<-1000", "-1000 <balance<-500", "-500 <balance<-100", "-100 -0", "0 - 100",
    "100 - 250", "250 - 500", "500 - 1000", "1000 - 1500", "1500 - 2500", "2500 - 5000", "5000 - 10000", "10000 - 20000", "20000 - 50000", "50000 - 110000"]
data_1['balance_cat'] = pd.cut(data_1['balance'], bins_1, labels=category_1)
print(data_1)

# %%
bins_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 65]
category_2 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10 - 15",
    "15 - 20", "20 - 25", "25 - 30", "30 - 35", "35 - 40", "40 - 50", "50 - 65"]
data_1['campaign_cat'] = pd.cut(data_1['campaign'], bins_2, labels=category_2)
print(data_1)

# %%
bins_3 = [-2, -1, 50, 100, 150, 200, 250,
    300, 350, 400, 450, 500, 550, 650, 880]
category_3 = ["npc", "0 - 50", "50 - 100", "100 - 150", "150 - 200", "200 - 250", "250 - 300",
    "300 - 350", "350 - 400", "400 - 450", "450 - 500", "500 - 550", "550 - 650", "650 - 880"]
data_1['pdays_cat'] = pd.cut(data_1['pdays'], bins_3, labels=category_3)
print(data_1)

# %%
bins_4 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
category_4 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "10 - 15", "15 - 20", "20 - 25", "25 - 30", "30 - 35", "35 - 40"]
data_1['previous_cat'] = pd.cut(data_1['previous'], bins_4, labels=category_4)
print(data_1)

# %%


def cat(ds, c):
    """Visualising categorical columns"""
    for i in c:
        print('Information about', i)
        print(round((ds[i].value_counts(normalize=True) * 100), 2))
        sns.catplot(x=i, kind='count', data=ds, aspect=8,
                    height=2, order=ds[i].value_counts().index)
        plt.show()


# %%
cat(ds=data_1, c=['age_cat', 'job', 'marital', 'education', 'default', 'balance_cat', 'housing', 'loan',
    'contact', 'day_cat', 'month', 'duration_cat', 'campaign_cat', 'pdays_cat', 'previous_cat', 'poutcome', 'y'])

# %% [markdown]
# Missing Data Handling

# %%
# we notice the exitence of unknown data.
# we will treat unknown data:
# unknown values exists in : poutcome, contact, education and job columns
# these unknown values have to be replaced
# job column has 288 rows of unknown:we will ignore them as it is very small compared to our dataset
# poutcome: we will drop this column as most of the data contained is unknown
# Unknown values in Education and Contact will be replaced by mode

# %%
# matching missing value from job with education mode and the unkown will be dropped
data_data[data_data['job'] == 'unknown']['education'].value_counts()

# %%


# %%
data_off = data_data[data_data['job'] != 'unknown']

# %%
print(data_off['job'].value_counts())

# %%
data_off['education'].replace(
    "unknown", data_off['education'].mode()[0], inplace=True)

# %%
print(data_off['education'].value_counts())

# %%
data_off['contact'].replace(
    "unknown", data_off['contact'].mode()[0], inplace=True)

# %%
print(data_off['contact'].value_counts())

# %%
data_off.drop('poutcome', axis=1, inplace=True)

# %%
plt.figure(figsize=(20, 10))
sns.heatmap(data_off.corr(), annot=True)

# %% [markdown]
# End of Missing Data handling

# %% [markdown]
# PCA

# %% [markdown]
# Test Train Split

# %%
normalizer = preprocessing.Normalizer().fit(X_train)
X_train = normalizer.transform(X_train)
X_test = normalizer.transform(X_test)


# %%
# Create train_x dataframe
train_x_data = data_off.iloc[:, :-1]
train_x_data.head()

# %%

# Create train_y dataframe
train_y = data_off[['y']]
train_y.head()

# %%
train_x_data.info()

# %%
numerical_variables = [
    col for col in train_x_data.columns if train_x_data[col].dtype in ['int64', 'float64']]

# %%
print(numerical_variables)

# %%
# Normalizing the train and test data with TRAINING mean and var.


for column in train_x_data.columns:
    if train_x_data[column].dtype in ['int64', 'float64']:

        X = train_x_data[column].array.reshape(-1, 1)

        # build the scaler model
        scaler = MinMaxScaler()

        # fit using the train set
        scaler.fit(X)

        train_x_data[column] = scaler.transform(
            train_x_data[column].array.reshape(-1, 1))  # Apply to train
        # dft[column]=scaler.transform(dft[column].array.reshape(-1, 1)) # Apply to test


train_x_data.describe()

# %%
train_x_data.info()

# %%
categorical_columns = ['month', 'contact', 'loan',
    'housing', 'default', 'education', 'job', 'marital']
# Encoding cat
encoders = OneHotEncoder(dropLast=False, inputCol=categorical_columns.getOutputCol(),
            outputCol="{0}_encoded".format(categorical_columns.getOutputCol()))


# %%
ohe = OneHotEncoder(sparse=False)

# %%
X_feature = train_x_data[['month', 'contact', 'loan',
    'housing', 'default', 'education', 'job', 'marital']]
print(X_feature)

# %%
print(train_x_data)

# %%
train_x_data.drop("day", axis=1, inplace=True)

# %%
column_trans = make_column_transformer((OneHotEncoder(), [
                                       'month', 'contact', 'loan', 'housing', 'default', 'job', 'marital']), remainder='passthrough')
column_trans.fit_transform(train_x_data)

# %%
X = np.array(column_trans.fit_transform(train_x_data)))

# %%
print(train_x_data.info())

# %%
train_x_data['contact'].value_counts()

# %% [markdown]
# Encoding the dependent variable:

# %%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(train_y['y'])

# %%
print(y)
# print(train_y)

# %%
train_x_data["job"].value_counts()

# %%
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse = False, handle_unknown = 'ignore')
train_enc=ohe.fit_transform(
    train_x_data[['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month']])
# Converting back to a dataframe
train_set_x=pd.DataFrame(train_enc, columns = ohe.get_feature_names())[:3]
print(train_set_x.columns)
print(train_set_x.type())


# %%
ed_data=train_x_data["education"]
print(ed_data)

# %%
# Perform label encoding on education as it is an ordinal data
education_category={'unknown': 0,
          'primary': 1,
          'secondary': 2,
          'tertiary': 3}
train_x_data['education'] = train_x_data['education'].replace(
    education_category)
# train_set_set = train_x_data['education'].join(train_set_x)
train_set_set=train_set_x.merge(
    train_x_data['education'], left_index = True, right_index = True)
print(train_set_set)
train_final_set=train_set_set.merge(
    train_x_data['age'], left_index = True, right_index = True)
print(train_final_set), 'balance', 'duration', 'campaign', 'pdays', 'previous'


# %% [markdown]
# Get_dummy encoding

# %%
# Get a list of columns for one-hot encoding
ohe_cols=list(train_x_data.select_dtypes(include='object').columns.values)

# We want to label encode education
le_col=['education']

# Drop education
ohe_cols.remove('education')
ohe_cols

# %%
train_x_2=pd.get_dummies(
    train_x_data, prefix = ohe_cols, columns = ohe_cols, drop_first = True)
train_x_2.head()

# %%
train_x_2.columns

# %%
# Perform label encoding on education as it is an ordinal data
education_category={'unknown': 0,
          'primary': 1,
          'secondary': 2,
          'tertiary': 3}
train_x_2['education'] = train_x_2['education'].replace(education_category)
train_x_2['education'].value_counts()

# %%
train_x_2.head()

# %%
# Encode target variable
y_category = {'no': 0,
         'yes': 1}
train_y['y'] = train_y['y'].replace(y_category)
train_y['y'].value_counts()

# %%
print(train_y)

# %%
from sklearn.model_selection import train_test_split

# Splitting the data into train and test data
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(train_x_2,train_y,test_size = 0.25, random_state = 20)

# %%
y_train_4.value_counts(normalize=True)

# %%
y_train_4.shape

# %%
y_test_4.value_counts(normalize=True)

# %%
y_test_4.shape

# %%
X_train_4.shape

# %%
X_train_4.columns

# %%
X_test_4.shape

# %% [markdown]
# building the model

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, make_scorer, confusion_matrix, classification_report, fbeta_score

SEED = 42
np.random.seed(SEED)

# %%
dc = DecisionTreeClassifier()

# %%
# Define a scorer
rs = make_scorer(recall_score)

# Cross validation
cv = cross_val_score(dc, train_x_2, train_y, cv=10, n_jobs=-1, scoring=rs)
print("Cross validation scores: {}".format(cv))
print("%0.2f recall with a standard deviation of %0.2f" % (cv.mean(), cv.std()))

# %%
dc.fit(X_train_4, y_train_4)


# %%
dc.score(X_train_4, y_train_4)

# %%
dc.score(X_test_4,y_test_4)

# %%
# Get predictions from the train dataset
pred = dc.predict(X_test_4)
# print("The train recall score is {}".format(np.round(recall_score(train_y, pred), 2)))

# %%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test_4,pred))

# %%
print(confusion_matrix(y_test_4,pred))

# %%


# %% [markdown]
# Random Forrest

# %% [markdown]
# 

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rfc = RandomForestClassifier(n_estimators=300)

# %%
rfc.fit(X_train_4, y_train_4)

# %%
predictions = rfc.predict(X_test_4)

# %%
print(classification_report(y_test_4,predictions))

# %%
print(confusion_matrix(y_test_4,predictions))

# %%
xg_clas = xgb.XGBClassifier(colsample_bytree = 0.5, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 3000, subsample=0.5, eval_metric='auc', verbosity=1)

# %%
eval_set = [(X_test_4, y_test_4)]
xg_clas.fit(X_train_4, y_train_4, early_stopping_rounds=10, verbose=True, eval_set=eval_set)

preds = xg_clas.predict(X_test_4)

# %% [markdown]
# Evaluation
# 

# %%
from sklearn.metrics import roc_auc_score
y_train_pred = xg_clas.predict_proba(X_train_4)[:,1]
y_valid_pred = xg_clas.predict_proba(X_test_4)[:,1]
print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train_4, y_train_pred),
                                                    roc_auc_score(y_test_4, y_valid_pred)))

# %%


# %%


# %%
print(classification_report(y_test_4,preds))

# %%


# %% [markdown]
#  Hyperparameter Tuning for XGBoost

# %% [markdown]
# earning_rate_list = [0.02, 0.05, 0.1]
# max_depth_list = [2, 3, 5]
# n_estimators_list = [1000, 2000, 5000]
# 
# params_dict = {"learning_rate": learning_rate_list,
#                "max_depth": max_depth_list,
#                "n_estimators": n_estimators_list}
# 
# num_combinations = 1
# for v in params_dict.values(): num_combinations *= len(v) 
# 
# print(num_combinations)
# params_dict

# %%
learning_rate_list = [0.02, 0.05, 0.1]
max_depth_list = [2, 3, 5]
n_estimators_list = [1000, 2000, 5000]

params_dict = {"learning_rate": learning_rate_list,
               "max_depth": max_depth_list,
               "n_estimators": n_estimators_list}

num_combinations = 1
for v in params_dict.values(): num_combinations *= len(v) 

print(num_combinations)
params_dict

# %%
def my_roc_auc_score(model, X, y): return roc_auc_score(y, model.predict_proba(X)[:,1])

model_xgboost_hp = GridSearchCV(estimator=xgb.XGBClassifier(subsample=0.5,
                                                                colsample_bytree=0.25,
                                                                eval_metric='auc',
                                                                use_label_encoder=False),
                                param_grid=params_dict,
                                cv=2,
                                scoring=my_roc_auc_score,
                                return_train_score=True,
                                verbose=4)

model_xgboost_hp.fit(X, y)

# %%
# cross_val_score
# and
# SMOTE

# %%
df_cv_results = pd.DataFrame(model_xgboost_hp.cv_results_)
df_cv_results = df_cv_results[['rank_test_score','mean_test_score','mean_train_score',
                               'param_learning_rate', 'param_max_depth', 'param_n_estimators']]
df_cv_results.sort_values(by='rank_test_score', inplace=True)
df_cv_results

# %%
# First sort by number of estimators as that would be x-axis
df_cv_results.sort_values(by='param_n_estimators', inplace=True)

# Find values of AUC for learning rate of 0.05 and different values of depth
lr_d2 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.05) & (df_cv_results['param_max_depth']==2),:]
lr_d3 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.05) & (df_cv_results['param_max_depth']==3),:]
lr_d5 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.05) & (df_cv_results['param_max_depth']==5),:]
lr_d7 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.05) & (df_cv_results['param_max_depth']==7),:]

# Let us plot now
fig, ax = plt.subplots(figsize=(10,5))
lr_d2.plot(x='param_n_estimators', y='mean_test_score', label='Depth=2', ax=ax)
lr_d3.plot(x='param_n_estimators', y='mean_test_score', label='Depth=3', ax=ax)
lr_d5.plot(x='param_n_estimators', y='mean_test_score', label='Depth=5', ax=ax)
lr_d7.plot(x='param_n_estimators', y='mean_test_score', label='Depth=7', ax=ax)
plt.ylabel('Mean Validation AUC')
plt.title('Performance wrt # of Trees and Depth')

# %%
# First sort by learning rate as that would be x-axis
df_cv_results.sort_values(by='param_learning_rate', inplace=True)

# Find values of AUC for learning rate of 0.05 and different values of depth
lr_t3k_d2 = df_cv_results.loc[(df_cv_results['param_n_estimators']==3000) & (df_cv_results['param_max_depth']==2),:]

# Let us plot now
fig, ax = plt.subplots(figsize=(10,5))
lr_t3k_d2.plot(x='param_learning_rate', y='mean_test_score', label='Depth=2, Trees=3000', ax=ax)
plt.ylabel('Mean Validation AUC')
plt.title('Performance wrt learning rate')

# %%
model_xgboost_fin = xgb.XGBClassifier(learning_rate=0.05,
                                          max_depth=2,
                                          n_estimators=5000,
                                          subsample=0.5,
                                          colsample_bytree=0.25,
                                          eval_metric='auc',
                                          verbosity=1,
                                          use_label_encoder=False)

# Passing both training and validation dataset as we want to plot AUC for both
eval_set = [(X_train_4, y_train_4),(X_test_4, y_test_4)]

model_xgboost_fin.fit(X_train_4,
                  y_train_4,
                  early_stopping_rounds=20,
                  eval_set=eval_set,
                  verbose=True)

# %%


# %%


# %%


# %% [markdown]
# cross validation before smote

# %% [markdown]
# we use stratified K-fold as we deal with an unbalenced dataset

# %%
models = [GradientBoostingClassifier(),RandomForestClassifier(),DecisionTreeClassifier()]

Y = y_train_4
X = X_train_4

# %% [markdown]
# might get rid of the below

# %%
"""
from sklearn import model_selection
def train(model, x, y):
    kfold = model_selection.StratifiedKFold(n_splits=10)
    pred = model_selection.cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    cv_mean = pred.mean()
    
    print('Model:',model)
    
    print('CV mean: %0.3f' % (cv_mean)) 
    """

# %%
def train(model, x, y):
    kfold = model_selection.StratifiedKFold(n_splits=10)
    score = model_selection.cross_val_score(model, x, y, cv=kfold, scoring="recall")
    # cv_mean = pred.mean()
    
    print('Model:',model)
    
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

# %% [markdown]
# check for precision and F1 as well

# %%
for model in models:
    train(model, X, Y)

# %%


# %%


# %% [markdown]
# Balancing with SMOTE

# %%
# Finding the RATIO of Imbalance of classes in our training data set
pd.read_csv("train.csv",sep = ';')

dfo= pd.read_csv("train.csv",delimiter=';') #Train dataset

a= len(dfo[dfo['y']=='yes'])/len(dfo[dfo['y']=='no'])

print("The ratio Yes/No is",round(a,3),"Yes to one No or for each Yes there is 7.42 No's")

# %%
X_train_4.columns

# %%
print(y_train_4)

# %%
from imblearn.over_sampling import SMOTE
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train_4[['age', 'education', 'balance', 'duration', 'campaign', 'pdays',
       'previous', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'marital_married',
       'marital_single', 'default_yes', 'housing_yes', 'loan_yes',
       'contact_telephone', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
       'month_oct', 'month_sep']],y_train_4["y"])
y_train_resampled.value_counts()

# %%
print(X_train_resampled.shape)
print(y_train_resampled.shape)

# %% [markdown]
# Cross-validation after applying SMOTE to Balance Data

# %%
for model in models:
    train(model, X_train_resampled,y_train_resampled)

# %%
# Plotting Graphs

import matplotlib.pyplot as plt

for m in models:

    When = ['Before SMOTE','After SMOTE']

    TN= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['TN'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['TN']]
    FP= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['FP'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['FP']]
    FN= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['FN'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['FN']]
    TP= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['TP'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['TP']]
  
    plt.figure(figsize=(10, 10))
    plt.plot(When, TN, color='red', marker='o')
    plt.plot(When, FP, color='blue', marker='x')
    plt.plot(When, FN, color='green', marker='8')
    plt.plot(When, TP, color='black', marker='s')

    plt.title(m, fontsize=15)
    plt.xlabel('When', fontsize=15)
    plt.ylabel('Values', fontsize=15)
    plt.legend(['TN', 'FP','FN','TP'], fontsize=15)
    plt.show()


    When = ['Before SMOTE','After SMOTE']

    AC= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['Ac'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['Ac']]
    PC= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['Pc'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['Pc']]
    RCLL= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['Rcll'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['Rcll']]
    F1= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['F1'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['F1']]
    ROC_AUC= [eval(m,X_train_4,y_train_4,X_test_4,y_test_4)['roc_auc'], eval(m, X_train_resampled, y_train_resampled,X_test_4,y_test_4)['roc_auc']]
  
    plt.figure(figsize=(10, 10))
    plt.plot(When, AC, color='red', marker='o')
    plt.plot(When, PC, color='blue', marker='x')
    plt.plot(When, RCLL, color='green', marker='8')
    plt.plot(When, F1, color='black', marker='s')
    plt.plot(When, ROC_AUC, color='orange', marker='p')

    plt.title(m, fontsize=15)
    plt.xlabel('When', fontsize=15)
    plt.ylabel('Metric Score', fontsize=15)
    plt.legend(['AC', 'PC','RCLL','F1','ROC_AUC'], fontsize=15)
    plt.show()

# %%
from sklearn.ensemble import GradientBoostingClassifier

# %%
model_gbm = GradientBoostingClassifier(n_estimators=5000,
                                       learning_rate=0.05,
                                       max_depth=3,
                                       subsample=0.5,
                                       validation_fraction=0.1,
                                       n_iter_no_change=20,
                                       max_features='log2',
                                       verbose=1)
model_gbm.fit(X_train_4, y_train_4)

# %%
len(model_gbm.estimators_)

# %%
y_train_pred = model_gbm.predict_proba(X_train_4)[:,1]
y_valid_pred = model_gbm.predict_proba(X_test_4)[:,1]

print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train_4, y_train_pred),
                                                    roc_auc_score(y_test_4, y_valid_pred)))

# %%
y_train_pred_trees = np.stack(list(model_gbm.staged_predict_proba(X_train_4)))[:,:,1]
y_valid_pred_trees = np.stack(list(model_gbm.staged_predict_proba(X_test_4)))[:,:,1]

y_train_pred_trees.shape, y_valid_pred_trees.shape

# %%
auc_train_trees = [roc_auc_score(y_train_4, y_pred) for y_pred in y_train_pred_trees]
auc_valid_trees = [roc_auc_score(y_test_4, y_pred) for y_pred in y_valid_pred_trees]

# %%
plt.figure(figsize=(12,5))

plt.plot(auc_train_trees, label='Train Data')
plt.plot(auc_valid_trees, label='Valid Data')

plt.title('AUC vs Number of Trees')
plt.ylabel('AUC')
plt.xlabel('Number of Trees')
plt.legend()

plt.show()

# %%
pd.DataFrame({"Variable_Name":var_columns,
              "Importance":model_gbm.feature_importances_}) \
            .sort_values('Importance', ascending=False)

# %%



