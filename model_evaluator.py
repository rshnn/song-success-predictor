
import numpy as np 
from itertools import combinations 
from sklearn.model_selection import LeaveOneOut, cross_val_score 

import config as config

class ModelEvaluator():


    def cross_validation_score(self, model, X, y, fold):
        """Returns average of k-fold cross validation 
        """

        return np.mean(cross_val_score(model, X, y, cv=fold)) 





    def generate_feature_combinations(self, feature_set_list, master_DF):
        """Returns list of dataframes that contain n! combinations of features 
            from the feature set list.  

            Meant for the purpose of doing exhaustive search over feature space 
             in cross-validation.  
        """

        dataframe_list = []; 

        for feature_set_size in range(len(feature_set_list)):
            if feature_set_size == 0: 
                continue 
            for s in combinations(feature_set_list, feature_set_size):
                
                df = master_DF[list(s)]
                dataframe_list.append(df)


        return dataframe_list 


