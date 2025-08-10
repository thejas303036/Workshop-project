import joblib as jb
import numpy as np
model = jb.load('redwine_model.pkl')
result = model.predict(np.array([[7.5,0.5,0.36,6.1,0.071,17,102,0.9978,3.35,0.8,10.5]]))
print("Quality:", result)