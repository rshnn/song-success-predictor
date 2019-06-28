
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import config 
from model_evaluator import ModelEvaluator 



class LinearRegressionManager(): 
    """Simple Multi Linear Regression Model 
        Contains sklearn LinearRegression class.  

            Performs training and prediciing.  
            Forward feature selection implemented.  

    """

    def __init__(self):
        self.model = LinearRegression()


    def train(self, X, y):
        """Wraps sklearn fit function for single model use 
        """
        self.model.fit(X, y) 


    def predict(self, X):
        """Wraps sklearn predict function for single model use 
        """
        return self.model.predict(X)   



    def R2_score(self, X, y): 
        """
        Wraps the sklearn score() function.
        Returns the R2 value (coef of determination) for the model 

        The coefficient R^2 is defined as (1 - u/v), where u is the residual 
            sum of squares ((y_true - y_pred) ** 2).sum() and v is the total 
            sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
            The best possible score is 1.0 and it can be negative. 
            A constant model that always predicts the expected value of y, 
            disregarding the input features, would get a R^2 score of 0.0.

        """
        return self.model.score(X, y) 



    def find_optimal_featureset(self, master_df, cv_df, features_list): 
        """Given a list of features (string form), returns the score and 
            dataframe of features that have the strongest predictive power
            on the cross validation set using 5-fold cross validation.  

            Combinations of the features list are generated and checked for 
            candidacy of being the very best that no one ever was 

            Returned score is R2 score (default for sklearn.LinearRegression)
        """
        meval = ModelEvaluator() 

        dataframe_list = meval.generate_feature_combinations(features_list, master_df)
        cv_dataframe_list = meval.generate_feature_combinations(features_list, cv_df) 

        both_lists = zip(dataframe_list, cv_dataframe_list)
        y = master_df['song_hotttnesss']
        y_cv = cv_df['song_hotttnesss']

        scores = []
        for df, cv in both_lists: 

            X = df 
            X_cv = cv 

            model = LinearRegression()   
            model.fit(X, y) 
            # score = meval.cross_validation_score(model, X_cv, y_cv, 5) 
            score = model.score(X_cv, y_cv)
            scores.append(score)


        max_idx = scores.index(max(scores))

        return dataframe_list[max_idx], scores[max_idx]



    def find_optimal_acoustic(self, master_df, cv_df):
        """Applies optimization check over acoustic features (see config.py)
        """
        return self.find_optimal_featureset(master_df, cv_df, config.acoustic_features)




    def find_optimal_metadata(self, master_df, cv_df):
        """Applies optimization check over metadata features (see config.py)
        """
        return self.find_optimal_featureset(master_df, cv_df, config.metadata_feaures)



    def find_optimal_constructed(self, master_df, cv_df):
        """Applies optimization check over constructed features (see config.py)
        """
        features_list = config.constructed_features + config.metadata_feaures
        return self.find_optimal_featureset(master_df, cv_df, features_list)


    def find_optimal_all(self, master_df, cv_df):
        """Applies optimization check over all features (see config.py)
            Dont run this.  It takes forever and barely any value.  Clearly will 
            overfit. 
        """
        features_list = config.constructed_features + config.metadata_feaures + config.acoustic_features 
        return self.find_optimal_featureset(master_df, cv_df, features_list)



    def errors(self, train, test, features_list):
        """Given a training set, test set, and feature list, will return the 
            absolute mean error, mean square error, and mean error of the 
            predictive model on the test set and training sets 
        """

        X = train[features_list]
        y = train['song_hotttnesss']

        X_test = test[features_list]
        y_test = test['song_hotttnesss']


        linreg = LinearRegression()
        linreg.fit(X, y)

        # Predicting the training_set
        y_pred_training = linreg.predict(X)

        # Predicting the testing_set
        y_pred_testing = linreg.predict(X_test)

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

        return linreg, training_error, testing_error, hot_std
