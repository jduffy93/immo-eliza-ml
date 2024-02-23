# Model card

## Project context

Develop a machine learning model for Immo Eliza (fake client) to predict property prices in belgium.

## Data

The input dataset (properties.csv) was created from previously scraped and cleaned property details from Immoweb.<br>
* Target feature: ``price``
* Features:

| Numerical Features | Dummy variables | Categorical Features |
| :---------------- | :------: | ----: |
| latitude |   fl_furnished   | property_type |
| longitude |   fl_terrace   | subproperty_type |
| construction_year |  fl_garden   | region |
| total_area_sqm |  fl_swimming_pool   | locality |
| nbr_frontages | fl_floodzone | state_building |
| nbr_bedrooms | fl_double_glazing | heating_type |
| terrace_sqm | - | epc |
| garden_sqm | - | - |
| primary_energy_consumption_sqm | - | - |
| cadastral_income | - | - |


## Model details

**Models tested:**
* Linear Regression from scikit-learn
* XGBRegressor from xgboost

**Model chosen:**<br>
XGBRegressor with hyper-parameter tuning.

**Hyper-parameters:**
* colsample_bytree= 0.7
* learning_rate= 0.05
* max_depth= 7
* min_child_weight= 4
* n_estimators= 500
* nthread= 4
* objective='reg:squarederror'
* subsample= 0.7


## Performance

For 200 different iterations of the random seed when splitting, the model had an Average Test R2 score of 0.7284.


## Limitations

... probably many

## Usage

**Dependencies:**

* click==8.1.7
* joblib==1.3.2
* numpy==1.26.4
* pandas==2.2.0
* pyarrow==15.0.0
* scikit-learn==1.4.1.post1
* scipy==1.12.0
* xgboost==2.0.3

**Installation:**

1. Clone this repository to your local machine:

```cmd
git clone https://github.com/jduffy93/immo-eliza-ml.git
```
2. Go to the **immo-eliza-ml** directory:

```cmd
cd directory/immo-eliza-ml
```

3. Install the required packages:

```cmd
pip install -r requirements.txt
```

4. If xgboost does not install properly, you may have to install it while not in your virtual environment, with the command:

```cmd
pip install xgboost==2.0.3
```

**Running immo-eliza-ml**

Make sure you are in the correct directory (/immo-eliza-ml), if not go to the directory:

 ```cmd
 cd directory/immo-eliza-ml
 ```

Run `train_xgboost.py` to train the model and generate artifacts:

```cmd
python train_xgboost.py
```

To generate predictions, run:

```cmd
python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
```


## Maintainers

Myself