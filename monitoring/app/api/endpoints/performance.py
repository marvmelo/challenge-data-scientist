"""Endpoint para cálculo de Performance."""
from fastapi import APIRouter
from .misc import *
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

router = APIRouter(prefix="/performance")

@router.post("/")
def performance(register_list: Registro):
    df = create_dataframe(register_list.registers)

    df_months = df.groupby(df["REF_DATE"].dt.month)["TARGET"].count()
    df_months = df_months.to_dict()

    model = load_model()
    X, y = separate_x_y(df.dropna(axis=0))
    predicted_y = model.predict_proba(X)[:, 1]
    registers_roc_curve = roc_curve(y, predicted_y)
    registers_auc = roc_auc_score(y, predicted_y)

    performance_response = PerformanceResponse(month_frequency = df_months,
    auc = registers_auc,
    FPRs = registers_roc_curve[0].tolist(),
    TPRs = registers_roc_curve[1].tolist())

    return performance_response