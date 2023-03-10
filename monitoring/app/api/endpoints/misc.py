import pickle
import sklearn
from pydantic import BaseModel
import pandas as pd

def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model

def create_dataframe(registers: list, remove_NaN=False):
    df = pd.DataFrame.from_dict(registers)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    if remove_NaN: df.dropna(axis=0, inplace=True)
    return df

def separate_x_y(df: pd.DataFrame):
    X = df.drop(axis=1, labels=["REF_DATE", "TARGET"])
    y = df["TARGET"]
    return X, y

class Registro(BaseModel):
    registers: list

class PerformanceResponse(BaseModel):
    month_frequency: dict
    auc: float
    FPRs: list
    TPRs: list

if __name__=="__main__":
    model = load_model()
    print(model.named_steps)
    print(model.classes_)
    print(model.feature_names_in_)
    print(model.n_features_in_)