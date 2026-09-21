<!--
Colocar los logos en:
  - docs/assets/logo-upct.png
  - docs/assets/logo-lhicarsa.png
-->

<div align="center">
  <img src="assets/logo-upct.png" alt="UPCT" height="70" style="margin-right:16px"/>
  <img src="assets/logo-lhicarsa.png" alt="Lhicarsa" height="70"/>
</div>

> **Nota:** este es el README original del kit del reto, publicado por la Universidad Politécnica de Cartagena. Se conserva tal cual como referencia. La descripción de nuestra solución está en el [README principal](../README.md).

# SmartEcoRutas · Guía didáctica para alumnos (StudentKit)

Este repositorio forma parte del programa **Retos‑UPCT** (<https://retos.upct.es>), una iniciativa que conecta el aprendizaje universitario con **retos reales**: problemas que existen fuera del aula, con datos y restricciones operativas realistas y un objetivo claro de impacto.

**Reto oficial:** **SmartEcoRutas — Optimización algorítmica de rutas de recogida de residuos urbanos en Cartagena**  
**Patrocinio:** **Lhicarsa**, concesionaria del servicio de recogida de residuos en Cartagena. Lhicarsa está interesada en analizar y aprender de las soluciones que desarrolléis. Además, **el premio económico de 1000 EUROS para el equipo ganador lo aporta Lhicarsa**.

Descripción pública del reto: <https://retos.upct.es/informacion/reto-smartecorutas>

> **Datos realistas, impacto real.** Las posiciones de contenedores y la estructura operativa están basadas en información real del entorno (Cartagena). Algunos datos pueden estar modificados para fines académicos y para preservar privacidad/operativa, pero la naturaleza del problema y sus decisiones clave son las de un escenario real.

---

## ¿Por qué este reto tiene impacto en la realidad?

Porque aquí no optimizáis “una lista de puntos”: optimizáis un **servicio urbano** que ocurre **cada día**.

Una mejora de minutos por ruta, multiplicada por camiones y jornadas, puede traducirse en:
- menos tiempo total en calle,
- menos congestión y menos interferencia con el tráfico urbano,
- menos costes operativos,
- y (como efecto colateral positivo) menos emisiones.

**Lo que hagáis aquí puede tener impacto en la ciudad**: no porque mañana se vaya a cambiar el servicio, sino porque aprender a resolver bien este problema es lo que se necesita para mejorar servicios urbanos modernos.

---

## Tabla de contenidos

- [1) Qué vas a programar (tu tarea real)](#1-qué-vas-a-programar-tu-tarea-real)
- [2) El problema en una frase (VRP con restricciones)](#2-el-problema-en-una-frase-vrp-con-restricciones)
- [3) Las 4 instancias oficiales](#3-las-4-instancias-oficiales)
- [4) Datos de entrada y el punto clave: **ProblemInstance**](#4-datos-de-entrada-y-el-punto-clave-probleminstance)
- [5) Formato de salida: qué debe devolver tu algoritmo](#5-formato-de-salida-qué-debe-devolver-tu-algoritmo)
- [6) Restricciones duras (lo que da ERROR)](#6-restricciones-duras-lo-que-da-error)
- [7) Cómo se calcula el tiempo de una ruta](#7-cómo-se-calcula-el-tiempo-de-una-ruta)
- [8) Ejecutar en tu ordenador](#8-ejecutar-en-local-y-entender-lo-que-pasa)
- [9) Salidas y visualización (KMZ/GPKG)](#9-salidas-y-visualización-kmzgpkg)
- [10) Protocolo oficial de prueba](#10-protocolo-oficial-de-prueba)
- [11) Criterio para decidir ganador](#11-criterio-para-decidir-ganador)
- [12) Inscripción y Entrega](#12-entrega)
- [13) Autoría, propiedad intelectual y evento final](#13-autoría-propiedad-intelectual-y-evento-final)
- [14) Consejos de diseño algorítmico](#14-consejos-de-diseño-algorítmico)
- [Apéndice A) instance.json (solo para entender “qué hay debajo”)](#apéndice-a-instancejson-solo-para-entender-qué-hay-debajo)

---

## 1) Qué vas a programar (tu tarea real)

Tu misión es implementar **un algoritmo de optimización** en:

- `student/algoritmoSmartEcoRutas.py`

Debe definir la función obligatoria:

```python
def solve(problem, time_limit_s: float, seed: int | None = None) -> list[list[str]]:
    ...
```

📌 **Importante:** aunque el dataset tiene ficheros como `instance.json`, `nodes.csv` y `time_matrix.npz`, **tú NO tienes que leer esos ficheros directamente**.

Lo que tu código recibe es un objeto **`ProblemInstance`**, ya cargado y listo para usar, que actúa como **API**. En la práctica, el reto consiste en “hablar” con `ProblemInstance` para construir rutas buenas.

> Por completitud, documentamos `instance.json` en el apéndice, porque ayuda a entender las reglas y los números. Pero para programar, céntrate en `ProblemInstance`. 

---

## 2) El problema en una frase (VRP con restricciones)

Esto es un problema de ruteo tipo **VRP** (*Vehicle Routing Problem*): tienes miles de contenedores repartidos por Cartagena y debes planificar **varias rutas de camión** que los visiten cumpliendo restricciones operativas.

Decisiones que tu algoritmo toma:
- cuántas rutas usar,
- qué contenedores asignar a cada ruta,
- en qué orden visitarlos,
- cuántas veces visitar el **DUMP** dentro de una ruta (descargas intermedias, punto donde el camión de recogida descarga)
  
---

## 3) Las 4 instancias oficiales

La evaluación oficial usa estas 4 instancias:

1. `LATERAL_CARTON`
2. `LATERAL_ENVASE`
3. `LATERAL_RESTO`
4. `TRASERA_RESTO`

Cada instancia representa un escenario operativo distinto (tipo de camión + tipo de residuo) y tiene **más de 1000 contenedores**.

---

## 4) Datos de entrada y el punto clave: **ProblemInstance**

Cuando se ejecuta `run.py`, el sistema hace:

1) Carga una instancia del problema con `ProblemInstance.load_from_dir(...)`  
2) Llama a tu `solve(problem, time_limit_s, seed)`  
3) Evalúa tu salida con `evaluate_solution(...)`  
4) (Opcional) genera export visual con `export_for_qgis(...)`

### 4.1) La “API” que vas a usar (lo esencial)

El objeto `problem` (tipo `ProblemInstance`) te da estas herramientas clave:

- Identificadores especiales:
  - `problem.base_uid()`  → UID de BASE (normalmente `"BASE"`)
  - `problem.dump_uid()`  → UID de DUMP (normalmente `"DUMP"`)
  - `problem.containers_uids()` → lista de UIDs de contenedores

- Tiempos de viaje (precalculados):
  - `problem.time_uid(a, b)` → segundos de viaje entre dos UIDs

- Utilidades para heurísticos:
  - `problem.k_nearest(uid, k, only_containers=True, exclude=...)` → vecinos más cercanos por tiempo

- Cálculo de tiempos de ruta:
  - `problem.travel_time_route_uids(route)`
  - `problem.service_time_route_uids(route)`
  - `problem.total_time_route_uids(route)`

> Idea clave: **tú no calculas mapas ni rutas por calle**. La matriz de tiempos ya está precalculada (a partir del callejero). Tu trabajo es diseñar la lógica de optimización.

Se espera de los equipos de estudiantes que lean el código proporcionado, investiguen, buscan, resuelvan: queremos fomentarla autonomía, la creatividad. 

### 4.2) Ejemplo mínimo (solo estructura, no competitivo)

```python
def solve(problem, time_limit_s: float, seed=None):
    base = problem.base_uid()
    dump = problem.dump_uid()
    conts = problem.containers_uids()

    # EJEMPLO didáctico: una sola ruta con un contenedor (NO resuelve el problema real)
    if not conts:
        return [[base, dump, base]]

    return [[base, conts[0], dump, base]]
```

---

## 5) Formato de salida: qué debe devolver tu algoritmo

Tu `solve(...)` devuelve:

- `list[list[str]]` → lista de rutas
- cada ruta es una lista de UIDs (strings) que existen en `nodes.csv`

Ejemplo de ruta cerrada (convención obligatoria):

```python
["BASE", "c_001", "c_017", "DUMP", "BASE"]
```

Se permiten descargas intermedias (replenishment). Esto quiere decir que un camión descarga los contenedores recogidos hasta el momento yendo al vertedero (DUMP), y siguiendo después con la recogida. Siempre terminanado con DUMP + BASE, ya que el camión a final del día debe quedar vacío. Por ejemplo:

```python
["BASE", "c_001", "c_002", "DUMP", "c_010", "DUMP", "BASE"]
```

---

## 6) Restricciones duras (lo que da ERROR)

El evaluador oficial (`framework/evaluator.py`) comprueba automáticamente:

### 6.1) Convención de ruta (obligatoria)
- cada ruta **empieza** en `BASE`
- cada ruta **termina** en `BASE`
- el **penúltimo** nodo debe ser `DUMP` → la ruta acaba en `... DUMP, BASE`

### 6.2) Cobertura
- todos los contenedores de la instancia deben visitarse
- **cada contenedor exactamente una vez**
  - faltantes ⇒ ERROR
  - duplicados ⇒ ERROR

### 6.3) Capacidad entre descargas
- no puedes visitar más de `max_containers_before_dump` contenedores consecutivos sin pasar por `DUMP`
- visitar `DUMP` “reinicia” ese contador (equivale a descargar)

### 6.4) Tiempo máximo por ruta
- debe cumplirse: `total_time(route) ≤ route_max_work_s`
- el tiempo total incluye **viaje + servicios** (ver siguiente sección)

---

## 7) Cómo se calcula el tiempo de una ruta

El tiempo total de una ruta se descompone en:

1) **Tiempo de viaje** (matriz precalculada):
- `T[i,j]` son **segundos** para ir del nodo `i` al nodo `j`
- en código: `problem.time_uid(a, b)`

2) **Tiempos de servicio** (operación real):
- visitar un **contenedor** suma `service_time_container_s`
- visitar **DUMP** suma `service_time_dump_s`
- `BASE` no añade servicio

En código, puedes obtener:
- `problem.travel_time_route_uids(route)`
- `problem.service_time_route_uids(route)`
- `problem.total_time_route_uids(route)`

---

## 8) Ejecutar en tu ordenador 

### 8.1) Preparar entorno (recomendado)

```bash
conda create -n smartecorutas python=3.11 -y
conda activate smartecorutas
pip install -r requirements.txt
```

### 8.2) Ejecución estándar

```bash
python run.py
```

Opciones útiles:
- `--seed 0` para reproducibilidad
- `--no-geo` para acelerar (sin export visual)
- `--use-simple-example` para ejecutar el algoritmo didáctico incluido como ejemplo (sin valor como optimización real)

Ejemplo:

```bash
python run.py --seed 0 --no-geo
```

### 8.3) Qué imprime `run.py`
Para cada instancia verás:
- carga de la instancia y parámetros
- tiempo de cómputo de tu algoritmo
- validación (errores y warnings)
- resumen por ruta (tabla compacta)
- `report.json` guardado en `algorithm_output/<INSTANCIA>/`

---

## 9) Salidas y visualización (KMZ/GPKG)

Para cada instancia se genera en `algorithm_output/<INSTANCIA>/`:

- `report.json` → métricas + errores/warnings (tu “hoja de resultados”)
- `export.kmz` → **Google Earth** (ideal para alumnos: capas y rutas)
- `export.gpkg` → **QGIS** (ideal para análisis GIS)

> Nota: el export visual **NO afecta** a la evaluación; sirve para inspección.

### 9.1) Visualización recomendada
- **QGIS**: carga `export.gpkg` o `export.kmz` y añade un mapa base OSM como fondo.
- **Google Earth**: abre `export.kmz` y activa/desactiva rutas por carpeta.

### 9.2) Nota sobre “trazado por calles”
Los tiempos de entrada están precalculados usando red vial basada en **OpenStreetMap**.  
En el export visual puede haber algún tramo que se dibuje con fallback (undirected o línea recta) si el grafo no permite un camino dirigido. **No es culpa del algoritmo del estudiante** y **no afecta** a la evaluación. 

---

## 10) Protocolo oficial de prueba

Recibiremos los algoritmos de los alumnos, enviados como os indicamos y evaluamos como se indica más adelante en las 4 instancias oficiales con:
- **15 minutos** de límite en cada una de las instancias 
- tolerancia máxima de **5 segundos** fuera de tiempo

Tu algoritmo debe controlar su presupuesto temporal: si estás haciendo búsquedas iterativas, implementa *early stop* por tiempo y devuelve la mejor solución conocida.

---

## 11) Criterio para decidir ganador

Se agregan resultados de las 4 instancias y se compara:

1. **Número total de rutas** (criterio principal, el que tenga menos rutas sumadas las cuatro instancias gana).
2. **Tiempo total de viaje** (`total_travel_time_s`) como desempate (menor tiempo es mejor).

---

## 12) Inscripción al reto y entrega

### Inscripción al reto (obligatoria)
Para participar en **SmartEcoRutas** dentro del programa **Retos-UPCT**, el equipo debe:

- Estar formado por **al menos 2 personas**.
- Enviar el **formulario de inscripción** disponible en:  
  https://retos.upct.es/informacion/formularios-y-recursos  
  a cualquiera de estos correos:
  - **pablo.pavon@upct.es**
  - **pilar.jimenez@upct.es**
- **Fecha tope de inscripción:** **24 de marzo de 2026**.

---

### Entrega (obligatoria)

- **Fecha límite de entrega:** **3 de mayo, 23:59**  
- **Envío por email:** **pablo.pavon@upct.es**  
  (la hora oficial es la de llegada del email)

#### Qué tenéis que enviar
- El fichero principal:
  - `student/algoritmoSmartEcoRutas.py` (**obligatorio**)
  - si fuera necesario importar librerias, un fichero `requirements.txt`

✅ **Preferencia (recomendada):** que sea **un único fichero** `.py` (más fácil de evaluar y reproducir).

Si queréis repartir el código en varios módulos:
- podéis hacerlo, pero **todo debe estar preparado para funcionar estando en el mismo directorio** (dentro de `student/`) y sin pasos manuales.

#### Qué NO tenéis que enviar
- **No enviéis ficheros de salida** (reports, exports, KMZ, GPKG, etc.).  
  Nuestro sistema automático los genera durante la evaluación.

---

### Cómo evaluamos (importante)
La evaluación se hace de forma automática. El procedimiento será:

1. Instalamos las dependencias estándar del repositorio ejecutando:
   - `pip install -r requirements.txt`
2. Ejecutamos vuestro algoritmo bajo el **protocolo oficial**, comprobando automáticamente:
   - límites de tiempo por instancia,
   - restricciones operativas (cobertura, duplicados, capacidad entre descargas, etc.),
   - validez del formato de salida,
   - y la métrica de calidad (rutas / tiempos) según el criterio del reto.


---

### Evento final: tendréis que explicar vuestro código
Recordad que **no basta con “que funcione”**: en el evento final público de **Retos-UPCT** (previsto en torno al **11 de mayo**) los equipos deberán **explicar su solución**.

Más adelante se detallará el formato exacto de los materiales, pero **incluye**:

- un **vídeo de ~2 minutos**, atractivo y comprensible, donde expliquéis:
  - qué habéis entregado,
  - cómo lo habéis hecho (idea algorítmica),
  - y por qué vuestra solución es buena.

Esto forma parte del espíritu de Retos-UPCT: aprender a **resolver** y también a **comunicar** soluciones técnicas reales.
---

## 13) Autoría, propiedad intelectual y evento final

- Los equipos/alumnos son **autores y propietarios** de sus algoritmos.
- UPCT usará el material exclusivamente para evaluación y clasificación dentro del reto.


---

## 14) Consejos de diseño algorítmico

Investiga, busca... ¿cómo se pueden resolver este tipo de problemas? ¿qué técnicas algorítmicas son más habituales, o han mostrado más éxito?

- Empieza por una solución **válida** (cumplir restricciones).
- Mide tiempos desde el inicio: el límite de cómputo es real.
- Heurísticas útiles: *nearest neighbor*, inserción, *savings*, *clustering*, GRASP, búsqueda local, ... 
- Itera: factibilidad → mejora incremental → intensificación/diversificación.
- Documenta: tendrás que explicar el enfoque de forma clara.

---

## Apéndice A) instance.json (solo para entender “qué hay debajo”)

**Tu algoritmo NO debe leer `instance.json`**, pero saber qué contiene ayuda a entender reglas y números.

Campos principales (ejemplos):
- `limits.max_routes` → referencia de rutas (no es restricción dura)
- `limits.max_containers_before_dump` → capacidad entre descargas
- `limits.route_max_work_s` → tiempo máximo total por ruta (segundos)
- `service_times_s.container` → tiempo de servicio por contenedor
- `service_times_s.dump` → tiempo de servicio por visita a DUMP
- `files.*` → nombres de ficheros y clave NPZ de la matriz

En `framework/problem_instance.py` ya está todo cargado y accesible como:
- `problem.max_routes`, `problem.max_containers_before_dump`, `problem.route_max_work_s`
- `problem.service_time_container_s`, `problem.service_time_dump_s`
- `problem.T` (matriz de tiempos), `problem.time_uid(a,b)`…

---

**Dudas:** revisad primero `run.py`, `framework/problem_instance.py`, `framework/evaluator.py` y `framework/geo_export.py`, porque reflejan exactamente el comportamiento del sistema automático. Se espera autonomía por parte de los estudiantes, pero si veis algo que no encaja, decídnoslo, podemos habernos equivocado!

