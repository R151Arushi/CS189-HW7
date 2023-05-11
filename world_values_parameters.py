import numpy as np  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# +
regression_knn_parameters = {
    'knn__n_neighbors': np.arange(1, 50),
   

    # Apply uniform weighting vs k for k Nearest Neighbors Regression
    ##### TODO(d): Change the weighting #####
     'knn__weights': ['uniform']

   
}
# -

###k_nearest_neighbors_regression_pipeline = Pipeline([
    ##('scaler', StandardScaler()),
   ## ('knn', KNeighborsRegressor())
##])###


