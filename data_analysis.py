import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import missingno as msno
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from keras import Sequential
from keras import optimizers, metrics
from keras.layers import Dense, Dropout, Input

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

### read data
folder = os.getcwd()
train = pd.read_csv(os.path.join(folder, "train.csv"))

train.shape
#train.info()

train = train.drop(columns="Id")

print('Duplicates:')
print(train.duplicated().sum())


numeric_train = train.select_dtypes(include=["int64", "float64"])
correlation_matrix = numeric_train.corr()


#####################################################################################################################################
### MISSING DATA
msno.matrix(train)
#plt.show()

msno.heatmap(train)
#plt.show()


train[train.columns[train.isnull().any()]].isna().sum()
# 19 variables with missing data

## LotFrontage: Linear feet of street connected to property
train["LotFrontage"].isna().sum() / train.shape[0] * 100
train["LotFrontage"].unique()
correlation_matrix["LotFrontage"]
# assumption: NAs == no street connected to property
train["LotFrontage"] = train["LotFrontage"].fillna(0)

## Alley: Type of alley access
train["Alley"].isna().sum() / train.shape[0] * 100
train["Alley"].unique()
# 93.77% of Nas -> rm NAs
train = train.drop(columns="Alley")

# MasVnrType: Masonry veneer type
train["MasVnrType"].isna().sum() / train.shape[0] * 100
train["MasVnrType"].unique()
# use a place holder
train["MasVnrType"] = train["MasVnrType"].fillna("Unknown")

# MasVnrArea: Masonry veneer area in square feet
train["MasVnrArea"].isna().sum() / train.shape[0] * 100
train["MasVnrArea"].unique()
correlation_matrix["MasVnrArea"]
# seems to be related to the quality of the house
# assumption: NAs == no Masonry veneer
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)

# BsmtQual
train["BsmtQual"].isna().sum() / train.shape[0] * 100
train["BsmtQual"].unique()
# use a place holder
train["BsmtQual"] = train["BsmtQual"].fillna("Unknown")

# BsmtCond
train["BsmtCond"].isna().sum() / train.shape[0] * 100
train["BsmtCond"].unique()
train[train["BsmtQual"] == "Unknown"]["BsmtCond"].unique()
# use a place holder
train["BsmtCond"] = train["BsmtCond"].fillna("Unknown")

# BsmtExposure
train["BsmtExposure"].unique()
train[train["BsmtQual"] == "Unknown"]["BsmtExposure"].unique()
# use a place holder
train["BsmtExposure"] = train["BsmtExposure"].fillna("Unknown")

# BsmtFinType1
train["BsmtFinType1"].unique()
train[train["BsmtQual"] == "Unknown"]["BsmtFinType1"].unique()
# use a place holder
train["BsmtFinType1"] = train["BsmtFinType1"].fillna("Unknown")

# BsmtFinType2
train["BsmtFinType2"].unique()
train[train["BsmtQual"] == "Unknown"]["BsmtFinType2"].unique()
# use a place holder
train["BsmtFinType2"] = train["BsmtFinType2"].fillna("Unknown")


# Electrical
train["Electrical"].unique()
# use a place holder
train["Electrical"] = train["Electrical"].fillna("Unknown")

# FireplaceQu
train["FireplaceQu"].isna().sum() / train.shape[0] * 100
train["FireplaceQu"].unique()
# use a place holder
train["FireplaceQu"] = train["FireplaceQu"].fillna("Unknown")

# GarageType
train["GarageType"].isna().sum() / train.shape[0] * 100
train["GarageType"].unique()
# use a place holder
train["GarageType"] = train["GarageType"].fillna("NoGarage")

# GarageYrBlt
train["GarageYrBlt"].isna().sum() / train.shape[0] * 100
train["GarageYrBlt"].unique()
train[train["GarageType"] == "NoGarage"]["GarageYrBlt"].unique()
# use a place holder
train["GarageYrBlt"] = train["GarageYrBlt"].fillna(9999)

# GarageFinish
train["GarageFinish"] = train["GarageFinish"].fillna("NoGarage")

# GarageQual
train["GarageQual"] = train["GarageQual"].fillna("NoGarage")

# GarageCond
train["GarageCond"] = train["GarageCond"].fillna("NoGarage")

# PoolQC
train["PoolQC"].isna().sum() / train.shape[0] * 100
train["PoolQC"].unique()
# use a place holder
train["PoolQC"] = train["PoolQC"].fillna("NoPool")

# Fence
train["Fence"].isna().sum() / train.shape[0] * 100
train["Fence"].unique()
# use a place holder
train["Fence"] = train["Fence"].fillna("NoFence")

# MiscFeature
train["MiscFeature"].isna().sum() / train.shape[0] * 100
train["MiscFeature"].unique()
# use a place holder
train["MiscFeature"] = train["MiscFeature"].fillna("NoMiscFeature")

msno.matrix(train)
plt.show()

#####################################################################################################################################
### EXPLORATORY ANALYSIS
# variable of interest: SalePrice
correlation_matrix["SalePrice"]
train.columns

sns.kdeplot(train["SalePrice"], color="blue", fill=True, alpha=0.5)
plt.xlabel("SalePrice")
plt.ylabel("Density")
plt.grid()
#plt.show()

# Heatmap of correlations between quantitative variables
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Matrix Heatmap", fontsize=20)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
#plt.show()


# Heatmap of cramer's V between qualitative variables
nominal_train = train.select_dtypes(include=["object"])


# Function to calculate Cramér's V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


vcramer_matrix = pd.DataFrame(
    index=nominal_train.columns, columns=nominal_train.columns
)

for var1 in nominal_train.columns:
    for var2 in nominal_train.columns:
        if var1 == var2:
            vcramer_matrix.loc[var1, var2] = 1  # Self-correlation
        else:
            confusion_matrix = pd.crosstab(train[var1], train[var2])
            vcramer_matrix.loc[var1, var2] = cramers_v(confusion_matrix)


plt.figure(figsize=(12, 10))
sns.heatmap(
    vcramer_matrix.astype(float),
    annot=False,
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Cramér's V Heatmap for Categorical Variables", fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
#plt.show()


#####################################################################################################################################
# MODELLING

### prepare df for modelling
# lable encoding
label_encoder = LabelEncoder()

for var in nominal_train.columns:
    train[var] = label_encoder.fit_transform(train[var])


X_df = train.drop(columns='SalePrice')
y_df = train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=22)


# remove multicollinear variables
high_corr_vars = np.where(np.abs(correlation_matrix) > 0.8)
high_corr_pairs = [(correlation_matrix.index[x], correlation_matrix.columns[y]) for x, y in zip(*high_corr_vars) if x != y and x < y]

high_cramers_v_vars = np.where(np.abs(vcramer_matrix) > 0.8)
high_cramers_pairs = [(vcramer_matrix.index[x], vcramer_matrix.columns[y]) for x, y in zip(*high_cramers_v_vars) if x != y and x < y]

X_train_lasso = X_train.drop(columns = ['GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'])
X_test_lasso = X_test.drop(columns = ['GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'])



# standardization of data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_lasso_scaled = scaler.fit_transform(X_train_lasso)
X_test_lasso_scaled = scaler.transform(X_test_lasso)



### linear model
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred_lm = lm.predict(X_test)
rmse_lm = np.sqrt(mean_squared_error(y_test, y_pred_lm))

results = pd.DataFrame()
results = pd.concat([results, pd.DataFrame({'Model': ['LM'], 'RMSE': [rmse_lm]})], ignore_index=True)



### CV ridge regression
# Define the hyperparameter grid
param_grid = {
    'alpha': np.logspace(-3, 3, 7)  # Values from 0.001 to 1000
}

grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']

ridge_model = Ridge(alpha = best_alpha)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
results = pd.concat([results, pd.DataFrame({'Model': ['Ridge'], 'RMSE': [rmse_ridge]})], ignore_index=True)



### CV lasso regression
# Define the hyperparameter grid
param_grid = {
    'alpha': np.logspace(-3, 3, 7)  # Values from 0.001 to 1000
}

grid_search = GridSearchCV(Lasso(max_iter=10000), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_lasso_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']

lasso_model = Lasso(alpha = best_alpha, max_iter=10000)
lasso_model.fit(X_train_lasso_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_lasso_scaled)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
results = pd.concat([results, pd.DataFrame({'Model': ['Lasso'], 'RMSE': [rmse_lasso]})], ignore_index=True)



### CV random forest
# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],       # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]          # Minimum number of samples required to be at a leaf node
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=125, warm_start=True), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_

rf_model = RandomForestRegressor(**best_params, random_state=125)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
results = pd.concat([results, pd.DataFrame({'Model': ['Random Forest'], 'RMSE': [rmse_rf]})], ignore_index=True)



### CV Gradient Boosting Machines
# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],        # Number of boosting stages
    'learning_rate': [0.01, 0.05, 0.1],     # Learning rate (step size)
    'max_depth': [3, 4, 5],                 # Depth of each tree
    'min_samples_split': [2, 5, 10],        # Minimum number of samples required to split a node
    'subsample': [0.8, 1.0, 1.5]            # Fraction of samples used for fitting individual trees
}

grid_search = GridSearchCV(GradientBoostingRegressor(random_state=65), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_

gbm_model = GradientBoostingRegressor(**best_params, random_state=65)
gbm_model.fit(X_train_scaled, y_train)
y_pred_gbm= gbm_model.predict(X_test_scaled)
rmse_gbm = np.sqrt(mean_squared_error(y_test, y_pred_gbm))
results = pd.concat([results, pd.DataFrame({'Model': ['Gradient Boosting Machines'], 'RMSE': [rmse_gbm]})], ignore_index=True)



### CV Neural network
# Define the model
def create_model(activation,learn_rate,dropout_rate,neurons):

    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)) )  
    model.add(Dense(neurons, activation=activation))   
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',
                optimizer=optimizers.Adam(learning_rate=learn_rate),
                metrics=[metrics.RootMeanSquaredError()])
    return model


# Define the parameters grid
activation =  ['relu','selu', 'elu', 'linear', 'tanh']
learn_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
dropout_rate = [0.0, 0.1, 0.2, 0.3]
neurons = [1, 5, 10, 20]
epochs = [10, 20, 100, 200, 300]
batch_size = [50, 100, 500, 1000]

param_grid = dict(activation=activation, 
    learn_rate=learn_rate, 
    dropout_rate=dropout_rate,
    neurons=neurons, 
    epochs=epochs, 
    batch_size=batch_size)

ann_model = KerasRegressor(model=create_model, 
    neurons=neurons, 
    learn_rate=learn_rate, 
    dropout_rate=dropout_rate,
    activation=activation)

opt = RandomizedSearchCV(
    ann_model,
    param_distributions=param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=27),
    scoring='neg_root_mean_squared_error',
    n_iter=150,
    random_state=123,
    n_jobs=-1
)
  
best_model = opt.fit(X_train_scaled, y_train).best_estimator_

print("Best Parameters for ANN:")
for param_name in opt.best_params_:
    print(f"{param_name}: {opt.best_params_[param_name]}")

y_pred_ann = best_model.predict(X_test_scaled)
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
results = pd.concat([results, pd.DataFrame({'Model': ['ANN'], 'RMSE': [rmse_ann]})], ignore_index=True)

print(results)