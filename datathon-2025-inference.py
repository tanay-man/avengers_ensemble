import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.utils import resample
import argparse
from typing import Any

def load_model() -> tuple[Any, argparse.Namespace, pd.DataFrame]:
    parser = argparse.ArgumentParser(description="Inference model to predict housing prices")
    parser.add_argument("ip", help="Input data file")
    parser.add_argument("--op", help="Prediction file name")

    args = parser.parse_args()

    try:
        test_data = pd.read_csv(args.ip)
    except:
        print("Error in reading input file: {args.ip}")
    with open("best_model/updated_xgb_model_iteration_2.pkl", "rb") as file:
        model = pickle.load(file)
    return model, args, test_data

def preprocess(test_data: pd.DataFrame) -> pd.DataFrame:
    df = test_data
    df.drop(columns=["Id"], inplace=True)
    replacing_text_NAN_cols = ["Alley", "BsmtCond", "BsmtFinType2", "BsmtQual", "Fence", "FireplaceQu", "GarageCond", "GarageFinish", "GarageQual", "GarageType", "MasVnrType", "MiscFeature", "PoolQC"]
    df.replace({col: {np.nan: 'DoesntExist'} for col in replacing_text_NAN_cols}, inplace=True)
    # Feature engineering
    df["AgeWhenSold"] = df["YrSold"] - df["YearBuilt"]
    df["RenovatedAgeSold"] = df["YrSold"] - df["YearRemodAdd"]
    df.drop(columns=["YrSold", "YearBuilt", "YearRemodAdd"], inplace=True)
    numerical_with_na = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
    categorical_to_numerical = ["Alley", "Street", "MSZoning", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageCond", "GarageQual", "GarageFinish", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]
    normalise = ["LotArea", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal"]
    for col in categorical_to_numerical:
        df[col] = df[col].map({k:v for k,v in zip(df[col].sort_values().unique(), [i for i in range(df[col].nunique())])})
    for col in normalise:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    # Handle null in numerical data using your mom
    for col in numerical_with_na:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    return df  

def main():
    model, args, test_data = load_model()
    test_data = preprocess(test_data)
    predictions = model.predict(test_data)
    ids = [i for i in range(len(test_data))]
    ids = np.array(ids)

    op_df = pd.concat([pd.DataFrame(ids), pd.DataFrame(predictions)], ignore_index=True, axis=1)
    op_df.columns = ["Ids","SalePrice"]
    op_df.to_csv(args.op, index=False)

main()
