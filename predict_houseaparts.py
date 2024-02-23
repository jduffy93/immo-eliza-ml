import click
import joblib
import pandas as pd
import numpy as np


@click.command()
@click.option("-i", "--input-dataset", help="path to input .csv dataset", required=True)
@click.option(
    "-o",
    "--output-dataset",
    default="output/predictions.csv",
    help="full path where to store predictions",
    required=True,
)
def predict(input_dataset, output_dataset):
    """Predicts house prices from 'input_dataset', stores it to 'output_dataset'."""
    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Load the data
    input_data = pd.read_csv(input_dataset)
    ### -------------------------------------------------- ###

    list_predictions = []
    list_index = []

    # Load the model artifacts using joblib
    artifacts_mult = joblib.load("models/artifacts.joblib")

    for key in artifacts_mult:
        #print(type(key),key)
        data = input_data.copy()
        list_index.extend(data[data["property_type"] == key].index)
        #print(list_index[:5], list_index[-5:])
        data = data[data["property_type"] == key]

        artifacts = artifacts_mult[key]

        # Unpack the artifacts
        num_features = artifacts["features"]["num_features"]
        fl_features = artifacts["features"]["fl_features"]
        cat_features = artifacts["features"]["cat_features"]
        imputer = artifacts["imputer"]
        enc = artifacts["enc"]
        model = artifacts["model"]


        # Extract the used data
        data = data[num_features + fl_features + cat_features]

        # Apply imputer and encoder on data
        data[num_features] = imputer.transform(data[num_features])
        data_cat = enc.transform(data[cat_features]).toarray()

        # Combine the numerical and one-hot encoded categorical columns
        data = pd.concat(
            [
                data[num_features + fl_features].reset_index(drop=True),
                pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
            ],
            axis=1,
        )

        # Make predictions
        list_predictions.append(model.predict(data))
    
    predictions = np.append(list_predictions[0], list_predictions[1])
    # Make predictions a df, add in the indices, sort, and then turn predictions into a narray
    df = pd.DataFrame(predictions, columns=['predictions'])

    df["indices"] = list_index
    df = df.sort_values(by=['indices'])
    predictions = df["predictions"].to_numpy()

    print('\n',df.columns, df.head(), '\n')

    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input!)
    pd.DataFrame({"predictions": predictions}).to_csv(output_dataset, index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {output_dataset}")
    click.echo(
        f"Nbr. observations: {input_data.shape[0]} | Nbr. predictions: {predictions.shape[0]}"
    )
    ### -------------------------------------------------- ###


if __name__ == "__main__":
    # how to run on command line:
    # python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
    predict()
