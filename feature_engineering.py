import numpy as np
import pandas as pd
from scipy.stats import zscore
 

class Feature_engineering():

    def __init__(self):
        pass

    def f_engineering(self):
        """Carries out ."""
        # Load the data
        df = pd.read_csv("data/properties.csv")

        #--------------------------------------------------------------
        # Preprocessing
        #--------------------------------------------------------------
        
        # Filling in missing longitude and latitude values with the average value for that locality
        for locality in df.locality.unique():
            mean_lat = round(df[df['locality'] == locality].latitude.mean(), 6)
            mean_long = round(df[df['locality'] == locality].longitude.mean(), 6)

            #print(f'{locality}\n  mean latitude: {mean_lat}\n  mean longitude: {mean_long}')
            
            # Condition
            cond = df['locality'] == locality

            # Replacing the empt values with means
            df.loc[cond,'latitude'] = df.loc[cond,'latitude'].fillna(mean_lat)
            df.loc[cond,'longitude'] = df.loc[cond,'longitude'].fillna(mean_long)
        
        df.drop(df[df['subproperty_type'] == "CASTLE"].index)

        # Removing outliers from numerical columns based on the z-score
        
        for column in df.select_dtypes(include=["float64"]).columns:
            z = np.abs(zscore(df[column]))

            # Identify outliers with a z-score greater than 3
            threshold = 3
            outliers = df[z > threshold]

            df = df.drop(outliers.index) # Drop outliers
        
        #--------------------------------------------------------------
        # Output
        #--------------------------------------------------------------
        df.to_csv("data/input.csv", index=False)
