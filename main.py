import streamlit as st
import pickle
import pandas as pd
import random
import xgboost as xgb

def load_model():
    '''Function to load the best model'''
    
    with open('models/xgboost_best_model.pickle', 'rb') as f:
        model = pickle.load(f)

    return model

model = load_model()


schema = {'warranty': 'category',
 'sub_status': 'category',
 'condition': 'category',
 'deal_ids': 'float',
 'base_price': 'float',
 'seller_id': 'category',
 'listing_type_id': 'category',
 'price': 'float',
 'buying_mode': 'category',
 'parent_item_id': 'category',
 'category_id': 'category',
 'official_store_id': 'category',
 'accepts_mercadopago': 'category',
 'original_price': 'float',
 'currency_id': 'category',
 'automatic_relist': 'category',
 'status': 'category',
 'initial_quantity': 'float',
 'sold_quantity': 'float',
 'available_quantity': 'float',
 'seller_address.longitude': 'float',
 'seller_address.id': 'category',
 'seller_address.address_line': 'category',
 'seller_address.latitude': 'float',
 'seller_address.search_location.neighborhood.id': 'category',
 'seller_address.search_location.state.id': 'category',
 'seller_address.search_location.city.id': 'category',
 'seller_address.zip_code': 'category',
 'seller_address.city.id': 'category',
 'seller_address.state.id': 'category',
 'shipping.local_pick_up': 'category',
 'shipping.tags': 'category',
 'shipping.mode': 'category',
 'shipping.free_methods': 'category',
 'seller_contact.phone2': 'category',
 'seller_contact.webpage': 'category',
 'seller_contact.email': 'category',
 'seller_contact.contact': 'category',
 'seller_contact.area_code': 'category',
 'seller_contact.other_info': 'category',
 'seller_contact.phone': 'category',
 'location.open_hours': 'category',
 'location.neighborhood.id': 'category',
 'location.longitude': 'float',
 'location.address_line': 'category',
 'location.latitude': 'float',
 'location.city.id': 'category',
 'location.state.id': 'category',
 'non_mercado_pago_payment_methods.MLAWC': 'category',
 'non_mercado_pago_payment_methods.MLACD': 'category',
 'non_mercado_pago_payment_methods.MLAVS': 'category',
 'non_mercado_pago_payment_methods.MLAMO': 'category',
 'non_mercado_pago_payment_methods.MLADC': 'category',
 'non_mercado_pago_payment_methods.MLAMP': 'category',
 'non_mercado_pago_payment_methods.MLAMC': 'category',
 'non_mercado_pago_payment_methods.MLABC': 'category',
 'non_mercado_pago_payment_methods.MLAOT': 'category',
 'non_mercado_pago_payment_methods.MLAAM': 'category',
 'non_mercado_pago_payment_methods.MLAWT': 'category',
 'non_mercado_pago_payment_methods.MLAVE': 'category',
 'non_mercado_pago_payment_methods.MLATB': 'category',
 'non_mercado_pago_payment_methods.MLAMS': 'category',
 'tags.good_quality_thumbnail': 'category',
 'tags.dragged_visits': 'category',
 'tags.free_relist': 'category',
 'tags.dragged_bids_and_visits': 'category',
 'tags.poor_quality_thumbnail': 'category',
 'pictures.large': 'float',
 'pictures.medium': 'float',
 'pictures.small': 'float',
 'title.new': 'category',
 'video': 'category',
 'seller_address.product_count': 'float'}


def load_csv(schema: dict) -> pd.DataFrame:
    '''Function to load dclean data'''

    df = pd.read_csv('data/clean/output.csv', dtype=schema)

    # Drop rows with 70+ missing values
    df = df.drop(16647)
    df = df.drop(83078)
    df = df.drop(92230)
    df = df.drop(82275)
    df = df.drop(88864)


    cols_to_drop = ['deal_ids', 'tags.poor_quality_thumbnail', 'original_price', 
                    'shipping.tags', 'seller_contact.other_info',	'seller_contact.phone',	'location.open_hours', 
                    'location.address_line', 'tags.poor_quality_thumbnail']
    
    df_cleaned = df.drop(columns=cols_to_drop)
    
    return df_cleaned


df = load_csv(schema)

def select_random_index(df: pd.DataFrame):
    '''Function that selects a random index from the DataFrame'''
    return random.choice(df.index)

random_index = select_random_index(df)

data_for_prediction = df.drop(columns=['condition']).iloc[[random_index]]

prediction = model.predict(data_for_prediction)

st.markdown('''
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
```            

# Ejemplos:
**La siguiente parte esta diseñada para pder mostrar como funciona el modelo desde una interfaz:**

Se seleccionó al azar una observación del set de datos:                                    
            ''')

st.write(data_for_prediction)

st.write(f"La predicción del modelo es: **{prediction[0]}**")
st.write(f"La respuesta correcta es: **{df['condition'][random_index]}**")