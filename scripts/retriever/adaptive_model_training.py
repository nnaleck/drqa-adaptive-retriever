import pandas as pd
import numpy as np
import mord
import pickle
from sklearn.model_selection import train_test_split

df = pd.read_csv('training_adaptive.csv', header=None)

# Getting features and labels
y = df[0].values

features_df = df.loc[:, 1:25].fillna(0)
X = features_df.loc[:, 1:25].values

# Generating train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training the model
reg = mord.OrdinalRidge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto')
reg.fit(X_train, y_train)

# Showing some results and saving model
y_pred = reg.predict(X_test)
b = 1
print(np.bincount(y_pred.astype(np.int32)+b))

filename = 'adaptive_model.sav'
pickle.dump(reg, open(filename, 'wb'))
