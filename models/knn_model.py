    
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import config 
from model_evaluator import ModelEvaluator  



class kNeighborsManager():
    """k Neighrest Neighbors classifier 
        Contains sklearn KNeighborsRegressor 

        Performs training and predicting. 
        Cross validation and forward feature selection also done.   
    """


    def __init__(self, num_neighbors=5): 
        self.model = KNeighborsRegressor(n_neighbors=num_neighbors)
        
    def train(self, X, y):
        """Wraps sklearn fit function for single model use. 
        """
        return self.model.fit(X, y)


    def predict(self, X):
        """Wraps sklearn predict function for single model use.  
        """
        return self.model.predict(X)


    def build_score_list(self, train_df, cv_df, features_list, maxk=100, step=2):
        """
        """
        X = train_df[features_list]
        X_cv = cv_df[features_list]

        y = train_df['song_hotttnesss']
        y_cv = cv_df['song_hotttnesss']


        score_list = np.zeros([1, 2])

        for k in np.arange(1, maxk, step):

            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X, y)
            score = model.score(X_cv, y_cv)

            score_list = np.append(score_list, [[k, score]], axis=0)

        score_list = np.delete(score_list, (0), axis=0)     
        return score_list 





    def errors(self, train, test, features_list, k):
        """Given a training set, test set, and feature list, will return the 
            absolute mean error, mean square error, and mean error of the 
            predictive model on the test set and training sets 
        """

        X = train[features_list]
        y = train['song_hotttnesss']

        X_test = test[features_list]
        y_test = test['song_hotttnesss']


        model = KNeighborsRegressor(k)
        model.fit(X, y)

        # Predicting the training_set
        y_pred_training = model.predict(X)

        # Predicting the testing_set
        y_pred_testing = model.predict(X_test)

        # Calculating errors
        mean_abs = metrics.mean_absolute_error(y, y_pred_training) 
        mean_sq = metrics.mean_squared_error(y, y_pred_training)
        mean_err = np.sqrt(metrics.mean_squared_error(y, y_pred_training))
        training_error = {"mean_abs": mean_abs, "MSE": mean_sq, "mean_err": mean_err}

        mean_abs = metrics.mean_absolute_error(y_test, y_pred_testing) 
        mean_sq = metrics.mean_squared_error(y_test, y_pred_testing)
        mean_err = np.sqrt(metrics.mean_squared_error(y_test, y_pred_testing))
        testing_error = {"mean_abs": mean_abs, "MSE": mean_sq, "mean_err": mean_err}

        # Getting std deviation of hotttnesss    
        hot_std = test['song_hotttnesss'].std()

        return model, training_error, testing_error, hot_std

