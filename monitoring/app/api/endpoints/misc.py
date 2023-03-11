import pickle
import sklearn
from pydantic import BaseModel
import pandas as pd
import os

dir_path = os.path.dirname(__file__)

def clean_df(model, df: pd.DataFrame):
    for column in df.columns:
        if column not in model.feature_names_in_:
            df.drop(axis=1, labels=[column], inplace=True, errors='ignore')
    return df

def load_model():
    with open(dir_path+"/files/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model

def create_dataframe(registers: list, remove_NaN=False):
    df = pd.DataFrame.from_dict(registers)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    if remove_NaN: df.dropna(axis=0, inplace=True)
    return df

def create_dataframe_from_path(path: str, remove_NaN=False):
    df = pd.read_csv(path, low_memory=False)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    if remove_NaN: df.dropna(axis=0, inplace=True)
    return df

def separate_x_y(df: pd.DataFrame):
    X = df.drop(axis=1, labels=["REF_DATE", "TARGET"], errors='ignore')
    y = df["TARGET"]
    return X, y

class Registro(BaseModel):
    registers: list

class RegistroPath(BaseModel):
    path: str

class PerformanceResponse(BaseModel):
    month_frequency: dict
    auc: float
    FPRs: list
    TPRs: list

class AderenciaResponse(BaseModel):
    ks: float
    test: list
    other: list

if __name__=="__main__":
    model = load_model()
    print(model.named_steps)
    print(model.classes_)
    print(model.feature_names_in_)
    print(model.n_features_in_)