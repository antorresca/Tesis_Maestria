"""
Extractor de entidades basado en reglas para el dominio del robot manipulador móvil.
Etapa 2 del pipeline PLN (opción B: clasificación + reglas).

Extrae: target (objeto a manipular) y destination (lugar destino).
"""

import re
from dataclasses import dataclass
from typing import Optional

# --- Vocabulario del dominio (expandir según la escena Gazebo) ---

OBJECTS = [
    # Formas básicas
    "cubo", "cube", "caja", "box", "cilindro", "cylinder",
    "esfera", "sphere", "botella", "bottle", "bloque", "block",
    "objeto", "object", "pieza", "piece", "item",
]

COLORS = [
    "rojo", "red", "azul", "blue", "verde", "green",
    "amarillo", "yellow", "naranja", "orange", "blanco", "white",
    "negro", "black", "morado", "purple",
]

LOCATIONS = [
    # Zonas
    "cocina", "kitchen", "mesa", "table", "bandeja", "tray",
    "área", "area", "zona", "zone", "punto", "point",
    "origen", "origin", "inicio", "start", "descarga", "drop zone",
    "workspace", "área de trabajo",
    # Colores de zona
    "zona verde", "green zone", "zona roja", "red zone",
    "zona azul", "blue zone", "zona amarilla", "yellow zone",
    "punto rojo", "red point", "punto azul", "blue point",
    "punto verde", "green point",
]

# Preposiciones de destino (ES/EN)
DEST_PREPS_ES = r"(?:a|hacia|hasta|en|sobre|encima de|dentro de)"
DEST_PREPS_EN = r"(?:to|towards|onto|on|in|into|at)"

# Preposiciones de origen (para fetch/transport)
SRC_PREPS_ES = r"(?:de|desde|en)"
SRC_PREPS_EN = r"(?:from|at|on)"


@dataclass
class Entities:
    target: Optional[str] = None
    destination: Optional[str] = None


def _build_object_pattern() -> str:
    colors_pat = "|".join(COLORS)
    objects_pat = "|".join(OBJECTS)
    # Coincide con: "cubo", "cubo azul", "el cubo azul", "blue cube"
    return (
        rf"(?:(?:{colors_pat})\s+)?(?:{objects_pat})"
        rf"|(?:{objects_pat})(?:\s+(?:{colors_pat}))?"
    )


def _build_location_pattern() -> str:
    locs_pat = "|".join(sorted(LOCATIONS, key=len, reverse=True))
    colors_pat = "|".join(COLORS)
    return (
        rf"(?:(?:{colors_pat})\s+)?(?:{locs_pat})"
        rf"|(?:{locs_pat})(?:\s+(?:{colors_pat}))?"
    )


OBJ_PAT = _build_object_pattern()
LOC_PAT = _build_location_pattern()


def _find_object(text: str) -> Optional[str]:
    m = re.search(OBJ_PAT, text, re.IGNORECASE)
    return m.group(0).strip() if m else None


def _find_destination(text: str) -> Optional[str]:
    # Busca: [preposición de destino] [artículo?] [location]
    pattern = (
        rf"(?:{DEST_PREPS_ES}|{DEST_PREPS_EN})"
        rf"\s+(?:el|la|los|las|the|a|an\s+)?"
        rf"({LOC_PAT})"
    )
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract(intent: str, text: str) -> Entities:
    """
    Extrae entidades según la intención detectada por el clasificador.

    Intenciones y entidades esperadas:
        navigate   → destination
        pick       → target
        place      → target (lo que se tiene), destination
        fetch      → target
        transport  → target, destination
        go_home    → (ninguna)
    """
    text_lower = text.lower()

    if intent == "navigate":
        return Entities(destination=_find_destination(text_lower))

    elif intent == "pick":
        return Entities(target=_find_object(text_lower))

    elif intent == "place":
        return Entities(
            target=_find_object(text_lower),
            destination=_find_destination(text_lower),
        )

    elif intent == "fetch":
        return Entities(target=_find_object(text_lower))

    elif intent == "transport":
        return Entities(
            target=_find_object(text_lower),
            destination=_find_destination(text_lower),
        )

    elif intent == "go_home":
        return Entities()

    return Entities()
