import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from feature_engineering import Feature_engineering 


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    #f_engineering = Feature_engineering()
    #f_engineering.f_engineering()


    # Load the data
    data = pd.read_csv("data/input.csv")
    artifacts = {}
    # Define features to use
    num_features = ["latitude", "longitude", "construction_year", "total_area_sqm",
                    "nbr_frontages", "nbr_bedrooms",
                    "terrace_sqm", "garden_sqm", "primary_energy_consumption_sqm",
                    "cadastral_income"]
    fl_features = ["fl_furnished", "fl_open_fire", "fl_terrace", "fl_garden",
                   "fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]
    cat_features = ["property_type", "subproperty_type", "region",
                    "locality", "state_building", "heating_type",
                    "epc"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=44
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


    # Train the model
    bst = XGBRegressor()
    bst.fit(X_train, y_train)

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
        "model": bst,
    }
    joblib.dump(artifacts, "models/artifacts_xgboost.joblib")


if __name__ == "__main__":
    train()
