import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# Load the libraries
qe_lib = pd.read_csv('QE1_Lib_POS.csv')
oe_lib = pd.read_csv('OE_Lib_POS.csv')

# Merge based on compound name to find common standards
merged = qe_lib.merge(oe_lib, on=['name', 'mz', 'Formula', 'adduct'], suffixes=('_QE', '_OE'))

# Apply Polynomial Regression with Ridge Regularization
X = merged[['rt QE_QE']].values.reshape(-1, 1)
y = merged['rt OE'].values
poly_model = make_pipeline(PolynomialFeatures(degree=7), Ridge(alpha=1.0))
poly_model.fit(X, y)

# Predict rt OE for compounds in QE library that are not in OE library
missing_oe = qe_lib[~qe_lib['name'].isin(oe_lib['name'])].copy()
missing_oe['predicted_rt_OE'] = poly_model.predict(missing_oe[['rt QE']])

# Save the new predictions
output_file = 'OE Predicted Lib POS.csv'
missing_oe[['adduct', 'mz', 'rt QE', 'name', 'Formula', 'predicted_rt_OE']].to_csv(output_file, index=False)

print(f'Predicted retention times saved to {output_file}')
