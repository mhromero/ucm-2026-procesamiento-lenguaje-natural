# === PLN BUTLER - MODELOS ===
#
# Modelos de datos Pydantic para el juego:
# - Carta: mensaje entre puestos
# - Inventario: recursos con operaciones aritméticas
# - InfoPuesto: estado de un puesto de juego

from typing import Any, Self

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


class Carta(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "remi": "Remitente",
                "dest": "Destinatario",
                "asunto": "Hola",
                "cuerpo": "Hola Mundo!",
            }
        }
    }

    remi: str = Field("", description="Remitente (texto libre)")
    dest: str = Field("", description="Destinatario (texto libre)")
    asunto: str = Field("", description="Asunto del mensaje")
    cuerpo: str = Field("", description="Cuerpo del mensaje")
    id: str | None = Field(None, description="ID del mensaje, asignado por el servidor")
    fecha: str | None = Field(
        None, description="Fecha de entrega en formato ISO, asignado por el servidor"
    )

    def formatear(self):
        return f"De: {self.remi}\nPara: {self.dest}\n**{self.asunto}**\n\n{self.cuerpo}"


class Inventario(dict):
    """Diccionario de recursos (mapa recurso -> cantidad)."""

    def __missing__(self, key: str) -> int:
        return 0

    def __setitem__(self, key: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"Inventario no puede ser negativo: {key}={value}")
        super().__setitem__(key, value)

    def __repr__(self) -> str:
        return ", ".join(f"{k}: {v}" for k, v in self.items() if v > 0)

    def __add__(self, otro: dict[str, int]) -> Self:
        nuevo = Inventario()
        for k in set(self.keys()) | set(otro.keys()):
            nuevo[k] = self[k] + otro.get(k, 0)
        return nuevo

    def __iadd__(self, otro: dict[str, int]) -> Self:
        """Operador += in-place"""
        for k, v in otro.items():
            self[k] += v
        return self

    def __sub__(self, otro: dict[str, int]) -> Self:
        """Operador -"""
        nuevo = Inventario()
        for k in set(self.keys()) | set(otro.keys()):
            nuevo[k] = self[k] - otro.get(k, 0)
        return nuevo

    def __isub__(self, otro: dict[str, int]) -> Self:
        """Operador -= in-place"""
        for k, v in otro.items():
            self[k] -= v
        return self

    def __lshift__(self, otro: dict[str, int]) -> Self:
        """Operador <<. Resta otro si se puede, si no deja a cero"""
        nuevo = Inventario()
        for k in set(self.keys()) | set(otro.keys()):
            nuevo[k] = max(0, self[k] - otro.get(k, 0))
        return nuevo

    @classmethod
    def _from_dict(cls, d: dict[str, int]) -> Self:
        inv = cls()
        for k, v in d.items():
            inv[k] = v  # Valida que no sea negativo
        return inv

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Hook para que Pydantic sepa serializar/deserializar Inventario."""
        return core_schema.no_info_after_validator_function(
            lambda v: cls._from_dict(v) if not isinstance(v, cls) else v,
            core_schema.dict_schema(
                keys_schema=core_schema.str_schema(),
                values_schema=core_schema.int_schema(),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(dict),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: Any
    ) -> JsonSchemaValue:
        """Personaliza el schema JSON para mostrar ejemplo claro."""
        json_schema = handler(core_schema)
        json_schema["example"] = {"madera": 4, "oro": 2}
        return json_schema


class InfoPuestoBase(BaseModel):
    Alias: str = Field(
        default="",
        description="Nombre por el que puede recibir correo",
    )
    Recursos: Inventario = Field(
        default_factory=Inventario, description="Recursos disponibles"
    )
    Objetivo: Inventario = Field(
        default_factory=Inventario, description="Recursos objetivo"
    )


class InfoPuestoBuzon(InfoPuestoBase):
    Buzon: dict[str, Carta] | None = Field(
        default=None,
        description="Mensajes recibidos, indexados por su id",
        examples=[
            {
                "abc-123": {
                    "remi": "Remitente",
                    "dest": "Destinatario",
                    "asunto": "Hola",
                    "cuerpo": "Hola Mundo!",
                    "id": "abc-123",
                    "fecha": "2025-01-01T00:00:00",
                }
            }
        ],
    )


from butler.settings import settings  # noqa: E402

InfoPuesto = InfoPuestoBuzon if settings.server.buzon else InfoPuestoBase
