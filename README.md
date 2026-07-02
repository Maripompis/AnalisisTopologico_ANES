# Análisis Topológico de Datos — Encuestas ANES

Pipeline de análisis que aplica homología persistente a los termómetros de sentimientos del *American National Election Studies* (ANES) para estudiar la evolución de las actitudes sociales en Estados Unidos entre 1992 y 2020.

---

## Requisitos

- Python 3.10 o superior
- Las siguientes librerías:

```bash
pip install numpy pandas matplotlib scipy ripser
```

---

## Descarga de los datos

1. Accede a [https://electionstudies.org/data-center/anes-time-series-cumulative-data-file/](https://electionstudies.org/data-center/anes-time-series-cumulative-data-file/)
2. Crea una cuenta gratuita y acepta los términos de uso
3. Descarga la versión **CSV** (versión de septiembre de 2022)
4. Coloca el fichero descargado en la **misma carpeta** que `tda_analysis.py` con el nombre exacto:

```
anes_timeseries_cdf_csv_20220916.csv
```

---

## Ejecución

```bash
python tda_analysis.py
```

Los resultados se guardan automáticamente en la carpeta `resultados_tda/`, que se crea junto al script.

---

## Documentos generados

Todos los ficheros se guardan dentro de `resultados_tda/`.

### Datos

| Fichero | Descripción |
|---|---|
| `tda_statistics.csv` | Tabla con las 12 métricas TDA (H₀ y H₁) por año y bloque. Una fila por combinación año–bloque. |
| `resultados.txt` | Log completo del análisis: vector de 12 componentes por año y bloque, y tablas resumen por bloque. |

### Gráficos de evolución temporal

Un gráfico por bloque con 6 paneles que muestran cómo evolucionan las métricas a lo largo de los años.

| Fichero | Bloque |
|---|---|
| `evolucion_Bloque_1_Grupos_raciales.png` | Bloque 1 (4 variables raciales) |
| `evolucion_Bloque_2_mas_Socioeconómico.png` | Bloque 2 (+ pobres y subsidios) |
| `evolucion_Bloque_3_mas_Partidos_políticos.png` | Bloque 3 (+ partidos políticos) |

### Comparación entre bloques

| Fichero | Descripción |
|---|---|
| `comparacion_bloques.png` | Evolución de 4 métricas para los tres bloques superpuestos en el mismo gráfico. |

### Barcodes

Barras horizontales de H₀ (componentes conexas) y H₁ (bucles) para los años representativos de cada bloque.

| Fichero | Bloque |
|---|---|
| `barcodes_Bloque_1_Grupos_raciales.png` | Bloque 1 |
| `barcodes_Bloque_2_mas_Socioeconómico.png` | Bloque 2 |
| `barcodes_Bloque_3_mas_Partidos_políticos.png` | Bloque 3 |

### Diagramas de persistencia

Puntos en el plano nacimiento–muerte para H₀ y H₁. Cuanto más alejado está un punto de la diagonal, más persistente es la característica topológica.

| Fichero | Bloque |
|---|---|
| `diagramas_Bloque_1_Grupos_raciales.png` | Bloque 1 |
| `diagramas_Bloque_2_mas_Socioeconómico.png` | Bloque 2 |
| `diagramas_Bloque_3_mas_Partidos_políticos.png` | Bloque 3 |
