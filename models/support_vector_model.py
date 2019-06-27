    def RunAndTestSVRModel(self, string_of_features, training_DF, testing_DF):
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

        # linreg = SVR(kernel='poly', C=1e3, degree=2)
        # linreg.fit(X, y_true_training)

        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        # svr_rbf.fit(X, y_true_training)
        svr_lin = SVR(kernel='linear', C=0.5)
        svr_lin.fit(X, y_true_training)
        # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        # svr_poly.fit(X, y_true_training)
        # y_rbf = svr_rbf.fit(X, y).predict(X)
        # y_lin = svr_lin.fit(X, y).predict(X)
        # y_poly = svr_poly.fit(X, y).predict(X)


        # Predicting the training_set
        # y_pred_training_rbf = svr_rbf.predict(X)
        y_pred_training_lin = svr_lin.predict(X)
        # y_pred_training_poly = svr_poly.predict(X)

        # Predicting the testing_set
        # y_pred_testing_rbf = svr_rbf.predict(testing_DF[X_cols])
        y_pred_testing_lin = svr_lin.predict(testing_DF[X_cols])
        # y_pred_testing_poly = svr_poly.predict(testing_DF[X_cols])


        # Calculating errors
        # mean_abs_rbf = metrics.mean_squared_error(y_true_training, y_pred_training_rbf) 
        mean_abs_lin = metrics.mean_squared_error(y_true_training, y_pred_training_lin) 
        # mean_abs_poly = metrics.mean_squared_error(y_true_training, y_pred_training_poly) 
        training_error = (0, mean_abs_lin, 0)
        # training_error = (mean_abs_rbf, mean_abs_lin, mean_abs_poly)
        

        # mean_abs_rbf = metrics.mean_squared_error(y_true_testing, y_pred_testing_rbf) 
        mean_abs_lin = metrics.mean_squared_error(y_true_testing, y_pred_testing_lin) 
        # mean_abs_poly = metrics.mean_squared_error(y_true_testing, y_pred_testing_poly)
        testing_error = (0, mean_abs_lin, 0)
        # testing_error = (mean_abs_rbf, mean_abs_lin, mean_abs_poly)

        # Getting std deviation of hotttnesss    
        total_DF = training_DF.append(testing_DF)
        hot_std = total_DF['song_hotttnesss'].std()

        return training_error, testing_error, hot_std
