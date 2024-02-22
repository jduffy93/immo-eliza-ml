import numpy as np
import pandas as pd
from scipy.stats import zscore
 

class Feature_engineering():

    def __init__(self):
        pass

    def f_engineering(self):
        """Carries out ."""
        # Load the data
        data = pd.read_csv("data/properties.csv")

        #--------------------------------------------------------------
        # Preprocessing
        #--------------------------------------------------------------
        
        # Filling in missing longitude and latitude values with the average value for that locality
        for locality in data.locality.unique():
            mean_lat = round(data[data['locality'] == locality].latitude.mean(), 6)
            mean_long = round(data[data['locality'] == locality].longitude.mean(), 6)

            #print(f'{locality}\n  mean latitude: {mean_lat}\n  mean longitude: {mean_long}')
            
            # Condition
            cond = data['locality'] == locality

            # Replacing the empt values with means
            data.loc[cond,'latitude'] = data.loc[cond,'latitude'].fillna(mean_lat)
            data.loc[cond,'longitude'] = data.loc[cond,'longitude'].fillna(mean_long)
        
        data.drop(data[data['subproperty_type'] == "CASTLE"].index)
        data = data.drop(data[(data["epc"] == "A++") & (data["region"] == "Brussels-Capital")].index)

        # Removing outliers from numerical columns based on the z-score
        for column in data.select_dtypes(include=["float64"]).columns:
            z_house = np.abs(zscore(data.loc[data["property_type"] == "HOUSE", column]))
            z_apartment = np.abs(zscore(data.loc[data["property_type"] == "APARTMENT", column]))

            # Identify outliers with a z-score greater than 3
            threshold = 3
            z = pd.concat([z_house, z_apartment]).sort_index()
            outliers = data[z > threshold]

            # Print the outliers
            #print(f"{column}: {len(outliers)}")
            data = data.drop(outliers.index)
        '''
        data['epc_flanders'] = data[data['region'] == "Flanders"]["epc"]
        data['epc_wallonia'] = data[data['region'] == "Wallonia"]["epc"]
        data['epc_brussels'] = data[data['region'] == "Brussels-Capital"]["epc"]
        data["epc_flanders"].fillna(value='None', inplace=True)
        data["epc_wallonia"].fillna(value='None', inplace=True)
        data["epc_brussels"].fillna(value='None', inplace=True)
        '''
        
        # Remap equipped_kitchen to lessen the n. categories
        # Define map
        kitchen_map = {
            'HYPER_EQUIPPED': 'INSTALLED',
            'SEMI_EQUIPPED': 'INSTALLED',
            'USA_HYPER_EQUIPPED': 'INSTALLED',
            'USA_INSTALLED': 'INSTALLED',
            'USA_SEMI_EQUIPPED': 'INSTALLED',
            'USA_UNINSTALLED': 'NOT_INSTALLED',
        }
        # Remap
        data["equipped_kitchen_short"] = data["equipped_kitchen"].map(kitchen_map)
        
        #--------------------------------------------------------------
        # Output
        #--------------------------------------------------------------
        data.to_csv("data/input.csv", index=False)
