import visualization as v1
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures



# Function to calculate mean square error
def mse(t_given, t_pred):
    return np.mean(np.power(np.subtract(t_given,t_pred),2))


def model_fit(model, X_train, t_train, X_test, t_test):
    
    # Train the model
    model.fit(X_train, t_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mse = mse(t_test, model_pred)
    
    # Return the performance metric
    return (model, model_mse, model_pred)


fname = 'trainingDa14.csv'

# Read the file
data = pd.read_csv(fname)

if  data.isnull().values.any() == False:
    
    ## Some exploratory plots
    v1.plot_1(data)
    
    # Create dummpy variables for categorical features
    data1 = pd.get_dummies(data)
    
    df_corr = data1.corr()['y'][1:]
    features_list = df_corr[abs(df_corr) > 0.1].sort_values(ascending=False)
    colnames = list(features_list.index)
    
    # Density plot of X256 column and scatter plot of X8 as they are in the feature list
    v1.plot_2(data)
    
    features = data1[colnames]
    #features = data1.drop(columns=['y'])
    targets = pd.DataFrame(data1['y'])
    
    # Split 80% trainng and 10% validation
    X_train, V_test, t_train, vt_test = train_test_split(features, targets, 
                                                        test_size = 0.1, random_state = 10)
    
    # Split 80% trainng and 20% testing 
    X_train, X_test, t_train, t_test = train_test_split(X_train, t_train, 
                                                        test_size = 0.2, random_state = 10)
    
    
    # A naive check by taking the median of the train target value
    primary_guess = np.median(t_train)
    print("Naive guess performance on the test: MSE = %.5f" % mse(t_test, primary_guess))
    
    # Scale the features with in the range of 0-1
    scaler = StandardScaler()
    
    # Fit on the training data
    scaler.fit(X_train)
    
    # Transform both the training and testing data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert target to one-dimensional array (vector)
    t_train = np.array(t_train).reshape((-1, ))
    t_test = np.array(t_test).reshape((-1, ))
    
    
    lr = LinearRegression()
    (lr_model, lr_mse, lr_pred) = model_fit(lr, X_train, t_train, X_test, t_test)
    
    print('Linear Regression Performance: MSE = %0.3f' % lr_mse)
    
    lr_p = LinearRegression()
    poly = PolynomialFeatures(degree=2)
    poly_train = poly.fit_transform(X_train)
    poly_test = poly.fit_transform(X_test)
    
    (poly_model, poly_mse, poly_pred) = model_fit(lr_p, poly_train, t_train, poly_test, t_test)
    
    print('Polynomial Regression Performance: MSE = %0.3f' % poly_mse)
    
    random_forest = RandomForestRegressor(random_state=100)
    (rf_model, random_forest_mse, rf_pred) = model_fit(random_forest, X_train, t_train, X_test, t_test)
    
    print('Random Forest Regression Performance: MSE = %0.3f' % random_forest_mse)
    
    gradient_boost = GradientBoostingRegressor(random_state=100)
    (gr_model, gradient_boost_mse, gr_pred) = model_fit(gradient_boost, X_train, t_train, X_test, t_test)
    
    print('Gradient Boosted Regression Performance: MSE = %0.3f' % gradient_boost_mse)
    
    
    
    ## Model comparison plots
    
    model_names = ("LinearRegression", "Polynomial", "RandomForest", "GradientBoost")
    errors = [lr_mse, poly_mse, random_forest_mse, gradient_boost_mse] 
    v1.plot_3(model_names, errors)
    v1.plot_4(t_test, lr_pred, poly_pred, rf_pred, gr_pred, flag=0)
    
    
    ## Now Check on validation data
    
    V_test = scaler.transform(V_test)
    vt_test = np.array(vt_test).reshape((-1, ))
    
    lr_predict = lr_model.predict(V_test)
    lr_mse = mse(vt_test, lr_predict)
    
    print('\n\n')
    print('Linear Regression on validation set: MSE = %0.3f' % lr_mse)
    
    poly_test = poly.fit_transform(V_test)
    poly_predict = poly_model.predict(poly_test)
    poly_mse = mse(vt_test, poly_predict)

    print('Polynomial regression on validation set: MSE = %0.3f' % poly_mse)
    
    rf_predict = rf_model.predict(V_test)
    rf_mse = mse(vt_test, rf_predict)
    
    print('Random Forest Regression on validation set: MSE = %0.3f' % rf_mse)
    
    gr_predict = gr_model.predict(V_test)
    gr_mse = mse(vt_test, gr_predict)
    
    print('Gradient Boosted Regression on validation set: MSE = %0.3f' % gr_mse)
    
    ## Plot prediction vs validation target data
    
    v1.plot_4(vt_test, lr_predict, poly_predict, rf_predict, gr_predict, flag=1)
    
    
    
    ## Finally use the test data to predict using polynomial regression
    
    fname_test = 'testDat9.csv'
    
    data_test = pd.read_csv(fname_test)
    
    if data_test.isnull().values.any() == False:
        
        data_test = pd.get_dummies(data_test)
    
        X = data_test[colnames]
        X = scaler.transform(X)
        
        poly_test = poly.fit_transform(X)
    
        # Make predictions
        pred_Y = poly_model.predict(poly_test)
        
        np.savetxt('predict_Y',pred_Y)
        
        









#mis_val = data.isnull().sum()

#feature = remove_collinear_features(data1, 0.3)
#category_subset = data.select_dtypes(exclude=['number'])
#x = ftr
#threshold = 0.6

