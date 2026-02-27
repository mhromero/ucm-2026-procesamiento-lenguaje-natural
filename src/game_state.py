"""
Estado del juego: clase State con alias, inventario, objetivo, necesidades,
excedentes y buzón. Incluye la lógica de extracción y comprobación de objetivo.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from . import api
from .config import GOLD_RESOURCE_NAME


@dataclass
class State:
    """
    Representa nuestro estado en la partida:
    - alias: quiénes somos
    - inventario: recursos que tenemos
    - objetivo: recursos que queremos conseguir
    - needs: lo que nos falta para el objetivo
    - surplus: lo que nos sobra y podemos ofrecer
    - buzon: cartas recibidas (id -> contenido)
    """

    alias: str
    inventory: Dict[str, int]
    target: Dict[str, int]
    needs: Dict[str, int]
    surplus: Dict[str, int]
    mailbox: Dict[str, Any]

    @classmethod
    def from_info(cls, info: Dict[str, Any]) -> "State":
        """
        Construye un State a partir de la respuesta de /info.
        Incluye alias, inventario, objetivo y buzón; calcula needs y surplus.
        """
        raw_alias = info.get("Alias") or info.get("alias")
        if not raw_alias:
            raise ValueError("No se ha encontrado el alias en la respuesta de /info")
        alias = raw_alias[0] if isinstance(raw_alias, list) else raw_alias

        raw_resources = info.get("Recursos") or {}
        raw_target = info.get("Objetivo") or {}
        inventory = {k: int(v) for k, v in raw_resources.items()}
        target = {k: int(v) for k, v in raw_target.items()}
        mailbox = info.get("Buzon") or {}

        needs, surplus = cls._compute_needs_and_surplus(inventory, target)
        return cls(
            alias=alias,
            inventory=inventory,
            target=target,
            needs=needs,
            surplus=surplus,
            mailbox=mailbox,
        )

    @staticmethod
    def _compute_needs_and_surplus(
        inventory: Dict[str, int], target: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Calcula needs (lo que nos falta) y surplus (lo que nos sobra).
        El oro se excluye del surplus.
        """
        needs: Dict[str, int] = {}
        surplus: Dict[str, int] = {}

        for resource, target_amount in target.items():
            actual = inventory.get(resource, 0)
            if target_amount > actual:
                needs[resource] = target_amount - actual
            elif actual > target_amount:
                surplus[resource] = actual - target_amount

        for resource, actual in inventory.items():
            if resource not in target and actual > 0:
                surplus[resource] = surplus.get(resource, 0) + actual

        surplus = {k: v for k, v in surplus.items() if k != GOLD_RESOURCE_NAME}
        return needs, surplus

    def update(self) -> None:
        """
        Actualiza el estado completo llamando a la API /info y refrescando
        alias, inventario, objetivo, buzón, needs y surplus.
        """
        info = api.get_info()
        raw_alias = info.get("Alias") or info.get("alias")
        if not raw_alias:
            raise ValueError("No se ha encontrado el alias en la respuesta de /info")
        alias = raw_alias[0] if isinstance(raw_alias, list) else raw_alias

        raw_resources = info.get("Recursos") or {}
        raw_target = info.get("Objetivo") or {}
        self.alias = alias
        self.inventory = {k: int(v) for k, v in raw_resources.items()}
        self.target = {k: int(v) for k, v in raw_target.items()}
        self.mailbox = info.get("Buzon") or {}
        self.recompute()

    def recompute(self) -> None:
        """
        Recalcula needs y surplus a partir del inventario y el objetivo actuales.
        """
        self.needs, self.surplus = self._compute_needs_and_surplus(
            self.inventory, self.target
        )

    def has_reached_objective(self) -> bool:
        """
        Comprueba si ya hemos alcanzado el objetivo de recursos.
        """
        for resource, target_amount in self.target.items():
            if self.inventory.get(resource, 0) < target_amount:
                return False
        return True

    def only_has_gold_to_trade(self) -> bool:
        """
        True si no hemos alcanzado el objetivo, no tenemos surplus de otros recursos
        y tenemos al menos 1 de oro para ofrecer (intercambiar 1 oro por cualquier recurso).
        """
        if self.has_reached_objective():
            return False
        if self.surplus:
            return False
        return self.inventory.get(GOLD_RESOURCE_NAME, 0) >= 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Exporta el estado a un diccionario (alias, inventario, objetivo, needs, surplus, buzon).
        """
        return {
            "alias": self.alias,
            "inventory": self.inventory,
            "target": self.target,
            "needs": self.needs,
            "surplus": self.surplus,
            "mailbox": self.mailbox,
        }
