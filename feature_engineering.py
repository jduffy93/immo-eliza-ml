import numpy as np
import pandas as pd
from scipy.stats import zscore
 


def f_engineering():
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
    
    df = df.drop(df[df['subproperty_type'] == "CASTLE"].index)

    # Removing outliers from numerical columns based on the interquartile range
    '''
    z = np.abs(zscore(df['price']))

    # Identify outliers as students with a z-score greater than 3
    threshold = 3
    outliers = df[z > threshold]

    # Print the outliers
    df = df.drop(outliers.index)

    '''

    for column in df.select_dtypes(include=["float64"]).columns:
        # Calculate the z-score for each student's height
        z = np.abs(zscore(df[column]))

        # Identify outliers as students with a z-score greater than 3
        threshold = 3
        outliers = df[z > threshold]


        df = df.drop(outliers.index)

    #--------------------------------------------------------------
    # Output
    #--------------------------------------------------------------
    df.to_csv("data/input.csv", index=False)

if __name__ == "__main__":
    f_engineering()