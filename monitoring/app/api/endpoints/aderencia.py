"""Endpoint para cálculo de aderência."""
from fastapi import APIRouter

router = APIRouter(prefix="/aderencia")

@router.post("/")
def aderencia():
    return "Hello aderencia"