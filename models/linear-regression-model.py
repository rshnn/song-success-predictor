

class LinearRegressionModel()


	# temp

    def RunAndTestLinearRegModel(self, string_of_features, training_DF, testing_DF):
        """
        Put in raw string of features.  Will run sklearn.linear_model.LinearRegression() to predict hotttnesss.
        Errors are calculated and returned as tuples.
        Returns: 
            model class
            training_error (mean_absolute_error, mean_squared_error, mean_error)
            testing_error (mean_absolute_error, mean_squared_error, mean_error)
            std dev of hotttnesss
        """
        X_cols = string_of_features.split()
        X = training_DF[X_cols]
        y_true_training = training_DF['song_hotttnesss']
        y_true_testing = testing_DF['song_hotttnesss']

        linreg = LinearRegression()
        linreg.fit(X, y_true_training)

        # Predicting the training_set
        y_pred_training = linreg.predict(X)

        # Predicting the testing_set
        y_pred_testing = linreg.predict(testing_DF[X_cols])


        # Calculating errors
        mean_abs = metrics.mean_absolute_error(y_true_training, y_pred_training) 
        mean_sq =  metrics.mean_squared_error(y_true_training, y_pred_training)
        mean_err = np.sqrt(metrics.mean_squared_error(y_true_training, y_pred_training))
        training_error = (mean_abs, mean_sq, mean_err)

        mean_abs = metrics.mean_absolute_error(y_true_testing, y_pred_testing) 
        mean_sq =  metrics.mean_squared_error(y_true_testing, y_pred_testing)
        mean_err = np.sqrt(metrics.mean_squared_error(y_true_testing, y_pred_testing))
        testing_error = (mean_abs, mean_sq, mean_err)

        # Getting std deviation of hotttnesss    
        total_DF = training_DF.append(testing_DF)
        hot_std = total_DF['song_hotttnesss'].std()

        return linreg, training_error, testing_error, hot_std
