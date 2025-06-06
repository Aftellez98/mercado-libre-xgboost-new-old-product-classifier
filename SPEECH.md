Modelo


METODOLOGIA:
Para empezar, quiero destacar que para proyectos de Machine Learning, prefiero seguir la metodología CRISP-DM. Es un marco muy estructurado y efectivo y por eso la presentación va a ser guidada usando estos mismos pasos.


1 Compression del negocio:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the marketplace is new or used.
Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution to predict if an item is new or used and then evaluate the model over held-out test data.

Si bien queda clara la tarea, algo que no esta explicitó es el objetivo de porque estoy yo como científico de datos buscando resolver esto.
- De ahora en adelante vamos a pedirles a los sellers incluir si es nuevo y usado entonces toca ponerle este valor a los productos viejos?
- Muchos usuarios critican que los sellers dicen que un producto es nuevo cuando es usado

Estas dos preguntas son importantes porque pueden ayudarme a definir como estaré evaluando el modelo.


2 Compression de los datos:

Lo primero que hago en esta fase es ver el tamaño y la estructura de los datos. En este caso estoy hablando de 300mb de información con multiples JSONs. 
Lo importante de esto es que me va a decir si es algo que voy a poder trabajar desde mi computador o si tendré que optar por CPU.

En este caso me fue muy bien, y no tengo que preocuparme por irme hacia otras herramientas.

En el coding podemos ver que cargo los archivos como una lista de diccionarios y después esta la transformo en un data frame de pandas.

A mi me ha resultado muy util utilizar librerías como pandas profiling para automatizar esta primera parte. Este recibe mi data frame y devuelve un html. 

Debido a la cantidad de datos solo mande 10% de estos arbitrariamente.

- Lo primero que voy a buscar es Y


3 Preparación de los datos:
Todos aquí sabemos que el dicho es que el 80% de lo que hace un científico de datos es limpiar datos, pues en este caso también aplica. 

Si bien hice muchas cosas en este paso, solo quiero que nos enfoquemos en unos casos:

1. Warranty
2. Substatus
3. Seller_contact
4. non_mercado_pago_payment_methods
5. time related columns
6. Size
7. Title
8. seller_contact.email


4 Entrenamiento

Hagamos un repaso rapido de las condiciones de los datos:

- Tengo 100 mil observaciones para  72+ variables 
- 13% de las observaciones son valores faltantes
- Tengo más de 30 variables categóricas y 8 numéricas, y estos presentan una alta correlación entre ellos. 
- Sin mencionar, que las clases no están balanceadas
- Y mi meta es poder hacer una clasificación binaria

Con esto en mente, en mi experiencia usaría optaría por usar arboles:
        * Random Forest: Robusto a valores faltantes y puede manejar tanto variables numéricas como categóricas.
        * Gradient Boosting Machines (GBM): Modelos como XGBoost, LightGBM o CatBoost son muy efectivos y pueden manejar datos desbalanceados y correlaciones.
Que otros modelos consideré?
    * Modelos Lineales:
        * Regresión Logística: Puede ser una buena opción si las relaciones son lineales, pero tendría problemas de multicolinealidad.
    * Modelos Basados en Distancias:
        * K-Nearest Neighbors (KNN): Aunque no es el más eficiente con muchas variables, puede ser útil después de una buena selección de características.
    * Redes Neuronales:
        * Perceptrón Multicapa (MLP): Puede ser útil si tienes suficientes datos y capacidad de cómputo.


QUE ES OPTUNA

1. Optuna es una librería que uso para poder optimizar los hiperparametros de mis modelos
2.  Especifican rangos, tipos y condiciones de los hiperparámetros.
3. Por defecto usa un optimizado bayesiana


Explica como funciona el XGBoost

Los árboles de decisión son modelos que se utilizan tanto para tareas tanto clasificación y regresión.

Individualmente pueden ser sesgados y tener poca varianza, lo que significa que pueden ajustarse demasiado a los datos de entrenamiento creando el overfitting. Entonces, los algoritmos se ajustan con tectacas como prunning, bagging, boosting para compensar esto.

{'max_depth': NIVELES
               'learning_rate': porcentaje de cambio al arbol anterior
               'n_estimators': ARBOLES, 
               'min_child_weight': peso minima para crear nodo hijo
               'gamma': EN LA FUNCION DE PERDIDA CUANTO SE NECESITA PARA JUSTIFICAR EL NODO
               'subsample': 0.9891830695807934, 
               'colsample_bytree': 0.5269533861847706, 
               'reg_alpha': 0.7767916720825168, 
               'reg_lambda': 0.4022821656006917,
               'enable_categorical': True,
               }


Cual es la segunda metrica?
Precision
* Definición: La precisión es el número de verdaderos positivos sobre todo lo que mi modelo dice que es positivo
* Importancia: Y esto es por lo que el costo de los falsos positivos es alto. 
* Por ejemplo, si clasificar incorrectamente un producto usado como nuevo tiene un impacto negativo significativo, querrás maximizar la precisión.


OUTLIERS

* Box Plot: A box plot displays the distribution of data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum. Outliers are typically plotted as individual points outside the "whiskers" of the box plot, which are often set at 1.5 times the interquartile range (IQR) from the quartiles.

* Local Outlier Factor (LOF): Measures the local density deviation of a given data point with respect to its neighbors. Points with a significantly lower density than their neighbors are considered outliers.

IMBALNCE

* SMOTE (Synthetic Minority Over-sampling Technique): Generate synthetic examples by interpolating between existing minority class examples.




Que son los Shapley Values?
Shapley es una librería super poderosa que puede ser usada para explicar como y cuanto esta afectando cada feature

