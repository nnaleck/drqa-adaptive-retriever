import pandas as pd
import numpy as np
import mord
import pickle
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--training-csv', type=str, default=None)
parser.add_argument('--out-model', type=str, default=None)
args = parser.parse_args()

df = pd.read_csv(args.training_csv, header=None)

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

filename = args.out_model
pickle.dump(reg, open(filename, 'wb'))
