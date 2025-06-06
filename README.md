# Documentación del Proceso de Construcción del Modelo XGBoost
Por Andres Felipe Tellez

## Objetivo
El objetivo de este proyecto es realizar análisis de datos, diseño, procesamiento y modelado de una solución de aprendizaje automático para predecir si un artículo es nuevo o usado, y luego evaluar el modelo con datos de prueba reservados.

## Metodología Escogida
Se utilizó la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining) para guiar el proceso de análisis y modelado.

## Exploración de Datos
Se utilizó la librería `pandas-profiling` para generar un informe HTML que incluye:
- Conteo de observaciones
- Cálculo de medias
- Primeras gráficas
- Datos faltantes
- Correlación entre variables

## Preparación de los Datos
1. **Remoción de Variables como:**
   - Se eliminaron variables con valores constantes, nulos o estructuras de datos vacías, como `seller_contact`, `site_id`, `listing_source`, `coverage_areas`, y `seller_address.country.id`.
   - Se eliminaron variables que comienzan con `seller_address.{}.name` ya que `seller_address.{}.id` contiene la misma información.

2. **Creación de Nuevas Features como:**
   - `seller_address.product_count`: Número de productos listados por el vendedor.
   - `pictures.large`, `pictures.medium`, `pictures.small`: Cantidad de fotos de cada tamaño.
   - `title.new`: Indica si el título contiene las palabras "nuevo" o "new".
   - Se revisó si el vendedor agrega comentarios, información de contacto, o un `official_store_id`.

3. **Codificación de Variables como:**
   - Aunque XGBoost no requiere codificación one-hot, se crearon variables de este tipo para `non_mercado_pago_payment_methods` y para los tags debido a su estructura de datos.

## Modelado
Se eligió XGBoost por su efectividad en el manejo de datos desbalanceados, outliers y correlaciones. Además, es adecuado tanto para clasificación como para regresión.

- **Selección de Hiperparámetros:** Se utilizó Optuna para optimizar los hiperparámetros.
- **Importancia de Variables:** Se evaluó utilizando valores de Shapley.

## Selección de variables:
Con los valores de Shapley, identifiqué las características con valores iguales a 0 y las eliminé del conjunto de datos, ya que indican una probabilidad de 0.5, sin aportar al modelo binario.

## Evaluación
- **Metrica Secundaria - Precisión:**
  - Se priorizó debido al alto costo de los falsos positivos.
  - Clasificar incorrectamente un producto usado como nuevo puede afectar negativamente la experiencia del usuario.

## Contenido del Repositorio

### Notebooks

- **Data exploration**: Exploración inicial de los datos para entender las características y distribuciones de las variables.
- **Datos preparation**: Procesos aplicados para manejar valores faltantes, eliminar duplicados y transformar variables para mejorar la calidad de los datos.
- **entrenamiento**: Implementación de un modelo XGBoost para la clasificación de los ítems, incluyendo la selección de hiperparametros y la optimización de hiperparámetros.

## Estructura del Proyecto

- `data/`: Contiene los conjuntos de datos utilizados para el análisis y entrenamiento.
- `notebooks/`: Jupyter notebooks con el análisis exploratorio de datos y el proceso de limpieza.
- `models/`: Archivos del modelo entrenado y resultados de evaluación.
- `README.md`: Este archivo, que proporciona una visión general del proyecto.

## Requisitos

- Python 3.9

Puedes instalar las dependencias ejecutando:

```bash
pip install -r requirements.txt