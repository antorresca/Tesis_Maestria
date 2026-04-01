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
# \b garantiza que "a" no coincida dentro de palabras como "la", "casa", etc.
DEST_PREPS_ES = r"(?:\b(?:a|hacia|hasta|en|sobre)\b|encima de|dentro de)"
DEST_PREPS_EN = r"(?:\b(?:to|on|in|at)\b|towards|onto|into)"


@dataclass
class Entities:
    target: Optional[str] = None
    destination: Optional[str] = None


def _build_object_pattern() -> str:
    colors_pat = "|".join(COLORS)
    objects_pat = "|".join(OBJECTS)
    # Las alternativas van de más específica a menos para que re.search
    # prefiera "cubo rojo" sobre solo "cubo".
    return (
        rf"(?:{colors_pat})\s+(?:{objects_pat})"   # "blue cube" / "rojo cubo"
        rf"|(?:{objects_pat})\s+(?:{colors_pat})"   # "cubo rojo" / "cube red"
        rf"|(?:{objects_pat})"                      # "cubo" (sin color)
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
        rf"\s+(?:el|la|los|las|the|an?|un[ao]?)?\s*"
        rf"({LOC_PAT})"
    )
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract(intent: str, text: str) -> Entities:
    """
    Extrae entidades según la intención detectada por el clasificador.

    Intenciones y entidades esperadas:
        navigate   → destination (si es "go home/casa/inicio" → destination="home")
        pick       → target (+ destination si el usuario indica de dónde recoger)
        place      → target (objeto que se porta), destination (dónde dejarlo)
    """
    text_lower = text.lower()

    if intent == "navigate":
        # Detectar go_home semánticamente
        home_keywords = r"\b(home|casa|inicio|base|dock|origen)\b"
        if re.search(home_keywords, text_lower):
            return Entities(destination="home")
        return Entities(destination=_find_destination(text_lower))

    elif intent == "pick":
        return Entities(
            target=_find_object(text_lower),
            destination=_find_destination(text_lower),
        )

    elif intent == "place":
        return Entities(
            target=_find_object(text_lower),
            destination=_find_destination(text_lower),
        )

    return Entities()
