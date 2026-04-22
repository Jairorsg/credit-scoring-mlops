from pydantic import BaseModel, Field, field_validator

class SolicitudCredito(BaseModel):
    AMT_INCOME_TOTAL:    float = Field(..., gt=0,  description="Ingreso anual total")
    AMT_CREDIT:          float = Field(..., gt=0,  description="Monto del préstamo")
    AMT_ANNUITY:         float = Field(..., gt=0,  description="Cuota anual")
    AMT_GOODS_PRICE:     float = Field(..., gt=0,  description="Valor del bien")
    DAYS_BIRTH:          int   = Field(..., lt=0,  description="Días desde nacimiento (negativo)")
    DAYS_EMPLOYED:       int   = Field(...,        description="Días empleado (negativo=activo)")
    CNT_FAM_MEMBERS:     float = Field(..., ge=1,  description="Miembros del hogar")
    NAME_CONTRACT_TYPE:  str   = Field(...,        description="Cash loans o Revolving loans")
    CODE_GENDER:         str   = Field(...,        description="M o F")

    @field_validator("CODE_GENDER")
    @classmethod
    def validar_genero(cls, v: str) -> str:
        if v.upper() not in ["M", "F"]:
            raise ValueError("CODE_GENDER debe ser M o F")
        return v.upper()

    @field_validator("NAME_CONTRACT_TYPE")
    @classmethod
    def validar_contrato(cls, v: str) -> str:
        opciones = ["Cash loans", "Revolving loans"]
        if v not in opciones:
            raise ValueError(f"NAME_CONTRACT_TYPE debe ser uno de: {opciones}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "AMT_INCOME_TOTAL":   135000.0,
                "AMT_CREDIT":         450000.0,
                "AMT_ANNUITY":        22500.0,
                "AMT_GOODS_PRICE":    400000.0,
                "DAYS_BIRTH":         -12000,
                "DAYS_EMPLOYED":      -2000,
                "CNT_FAM_MEMBERS":    2.0,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER":        "F"
            }
        }
    }


class RespuestaCredito(BaseModel):
    decision:       str   = Field(..., description="APROBADO o RECHAZADO")
    probabilidad:   float = Field(..., description="Probabilidad de mora (0-1)")
    score:          int   = Field(..., description="Score crediticio (0-1000)")
    nivel_riesgo:   str   = Field(..., description="BAJO / MEDIO / ALTO / MUY ALTO")
    umbral_usado:   float
    modelo_version: str