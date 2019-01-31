import numpy as np
import sklearn

from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
X, _ = load_digits(return_X_y=True)
transformer = FastICA(n_components=7,
        random_state=0)
X_transformed = transformer.fit_transform(X)
X_transformed.shape