# Prompt para Augmentación del Dataset con LLMs

Usa este prompt en ChatGPT, Gemini, Deepseek, etc. para generar variaciones lingüísticas de los ejemplos semilla. El objetivo es aumentar el dataset a ~50-80 ejemplos por intención (300-480 total).

---

## Prompt (copiar y pegar en el LLM)

```
Eres un asistente de investigación para una tesis sobre control de robots manipuladores móviles con procesamiento de lenguaje natural.

Necesito que generes variaciones lingüísticas de comandos de voz para un robot. El robot puede realizar estas acciones:

- navigate: moverse a un lugar sin manipular objetos
- pick: agarrar/levantar un objeto
- place: soltar/depositar un objeto que ya se tiene
- fetch: ir a buscar un objeto y traerlo
- transport: llevar un objeto de un lugar a otro
- go_home: regresar a la posición inicial

REGLAS:
1. Genera 10 nuevas frases por cada intención (60 total)
2. Mezcla español e inglés (aproximadamente 50/50)
3. Varía el vocabulario: sinónimos, estructuras gramaticales distintas, nivel de formalidad (coloquial, técnico, imperativo, interrogativo como "puedes ir a...")
4. Las frases deben ser naturales, como las diría una persona real a un robot
5. Incluye variaciones con y sin artículos, con y sin objetos/lugares específicos
6. Algunos ejemplos pueden ser ambiguos o menos directos (para robustez)

Devuelve el resultado en formato CSV con columnas: text,intent,lang
Usa comillas dobles si el texto contiene comas. Lang es "es" o "en".

Ejemplos semilla de referencia:
- navigate: "ve a la cocina" | "go to the table"
- pick: "recoge el cubo azul" | "grab the red object"
- place: "deja el cubo en la mesa" | "put the object on the tray"
- fetch: "trae el cubo rojo" | "fetch the blue cube"
- transport: "lleva el cilindro a la bandeja" | "move the box to the drop zone"
- go_home: "vuelve a casa" | "return to home position"
```

---

## Instrucciones de post-procesamiento

1. Revisar manualmente cada frase generada — eliminar las que cambien de intención o sean incorrectas
2. Añadir columna `source` con el nombre del LLM usado (e.g., `chatgpt`, `gemini`, `deepseek`)
3. Agregar al archivo `data/raw/augmented_dataset.csv`
4. Correr `src/prepare_dataset.py` para mezclar, deduplicar y generar los splits train/val/test

---

## Distribución objetivo

| Intent     | Mínimo | Objetivo |
|------------|--------|----------|
| navigate   | 50     | 70       |
| pick       | 50     | 70       |
| place      | 50     | 70       |
| fetch      | 50     | 70       |
| transport  | 50     | 70       |
| go_home    | 50     | 70       |
| **Total**  | **300**| **420**  |

Split: 70% train / 15% val / 15% test — estratificado por intención y lengua.
