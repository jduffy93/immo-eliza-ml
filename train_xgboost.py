import joblib
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from feature_engineering import Feature_engineering 


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    #f_engineering = Feature_engineering()
    #f_engineering.f_engineering()


    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ["latitude", "longitude", "construction_year", "total_area_sqm",
                    "nbr_frontages", "nbr_bedrooms",
                    "terrace_sqm", "garden_sqm", "primary_energy_consumption_sqm",
                    "cadastral_income"]
    fl_features = ["fl_furnished", "fl_terrace", "fl_garden",
                   "fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]
    cat_features = ["property_type","subproperty_type", "region",
                    "locality", "state_building", "heating_type",
                    "epc"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )


    '''
    param_test1 = {'n_estimators':range(100, 1200, 100), 'learning_rate':[0.01, 0.1, 1]}
    gbm_tune1 = GradientBoostingClassifier(max_features='sqrt', min_samples_leaf=0.001, max_depth=4)

    gs1 = GridSearchCV(estimator=gbm_tune1, param_grid=param_test1, scoring='roc_auc', n_jobs=6, cv=5)
    gs1.fit(X_train, np.ravel(y_train))

    print(f"The best parameters: {gs1.best_params_}, and the highest mean_test_score is {gs1.best_score_}")
    '''

    # Train the model
    bst = XGBRegressor()
    bst.fit(X_train, y_train)

    # Various hyper-parameters to tune
    xgb1 = XGBRegressor(colsample_bytree= 0.7, learning_rate= 0.05, max_depth= 7, min_child_weight= 4, n_estimators= 500, nthread= 4, objective='reg:squarederror', subsample= 0.7)
    xgb1.fit(X_train,y_train)


    xgb1_train_score = r2_score(y_train, xgb1.predict(X_train))
    xgb1_test_score = r2_score(y_test, xgb1.predict(X_test))
    print(f"Train R² score {type}: {xgb1_train_score}")
    print(f"Test R² score {type}: {xgb1_test_score}")
    print('==========================================================================')


    # Evaluate the model
    train_score = r2_score(y_train, bst.predict(X_train))
    test_score = r2_score(y_test, bst.predict(X_test))
    print(f"Train R² score {type}: {train_score}")
    print(f"Test R² score {type}: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": xgb1,
    }
    joblib.dump(artifacts, "models/artifacts_xgboost.joblib")


if __name__ == "__main__":
    train()
