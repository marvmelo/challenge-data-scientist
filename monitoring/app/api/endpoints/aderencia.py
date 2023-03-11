"""Endpoint para cálculo de aderência."""
from fastapi import APIRouter
from scipy.stats import ks_2samp
import os
from .misc import *

router = APIRouter(prefix="/aderencia")

@router.post("/")
def aderencia(register_path: RegistroPath):

    df = create_dataframe_from_path(dir_path+"/files/"+register_path.path)
    df_test = pd.read_csv(dir_path+"/files/test", low_memory=False)
    model = load_model()
    
    X = clean_df(model, df)
    _, y_test = separate_x_y(df_test)
    predicted_y = model.predict_proba(X)[:, 1]
    ks = ks_2samp(y_test, predicted_y)

    aderencia_response = AderenciaResponse(ks=ks.statistic,
                                           test=y_test.to_list(),
                                           other=predicted_y.tolist())
    return aderencia_response
