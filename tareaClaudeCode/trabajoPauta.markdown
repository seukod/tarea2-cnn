---
jupyter:
  interpreter:
    hash: 50456680fccf1c41085e05954d172de063da31b4663dbf91e3dc42a7c568c15e
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.11
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
# Red neuronal artificial para reconocer el tipo de un pokemon

En esta tarea ustedes deben diseñar, entrenar y evaluar un modelo de red
neuronal con arquitectura convolucional para resolver el problema de
reconocer el tipo de un pokemon en base a una imagen del mismo y a sus
atributos. El principal desafio es el desbalance y poca cantidad de
ejemplos en el dataset.

A continuación se instancia el dataset y se itera para presentar algunos
ejemplos:
:::

::: {.cell .code execution_count="1" scrolled="false"}
``` python
%load_ext autoreload
%autoreload 2
%matplotlib inline
import matplotlib.pyplot as plt
from pokemon_utils import PokemonImages

dataset = PokemonImages('data/')

fig, ax = plt.subplots(3, 3, figsize=(7, 5), tight_layout=True)
for ax_, (image, label, name, attributes) in zip(ax.ravel(), dataset):
    ax_.imshow(image.permute(-2, -1, 0))
    ax_.axis('off');
```

::: {.output .display_data}
![](vertopal_cda7676d6409494bbf30a03d28c46ba8/38688798aea13122cd81c74ecf3bbd9a2b158fea.png)
:::
:::

::: {.cell .markdown}
Cada ejemplo tiene su imagen, su etiqueta, su nombre y sus atributos:
:::

::: {.cell .code execution_count="2"}
``` python
image, label, name, attributes = dataset[0]
type(image), image.shape, label, name, attributes
```

::: {.output .execute_result execution_count="2"}
    (torch.Tensor,
     torch.Size([3, 256, 256]),
     9,
     'Bulbasaur',
     tensor([ 0.7000,  6.9000, 45.0000, 49.0000, 49.0000, 45.0000]))
:::
:::

::: {.cell .markdown}
Se puede obtener el nombre de la clase con:
:::

::: {.cell .code execution_count="3"}
``` python
dataset.categories[label]
```

::: {.output .execute_result execution_count="3"}
    'Grass'
:::
:::

::: {.cell .markdown}
Y los atributos disponibles son:
:::

::: {.cell .code execution_count="4"}
``` python
dataset.attribute_names
```

::: {.output .execute_result execution_count="4"}
    ['Height', 'Weight', 'HP', 'Attack', 'Defense', 'Speed']
:::
:::

::: {.cell .markdown}
La cantidad de ejemplos por clase es:
:::

::: {.cell .code execution_count="6"}
``` python
from collections import Counter
import pandas as pd
x = Counter(dataset.labels)
for key in sorted(x):
    print(f"{dataset.categories[key]}: {x[key]}")
```

::: {.output .stream .stdout}
    Bug: 52
    Dark: 20
    Dragon: 13
    Electric: 32
    Fairy: 8
    Fighting: 20
    Fire: 34
    Ghost: 18
    Grass: 58
    Ground: 26
    Ice: 19
    Normal: 80
    Poison: 26
    Psychic: 44
    Rock: 32
    Steel: 17
    Water: 96
:::
:::

::: {.cell .markdown}
En lo que sigue utilice los siguientes conjuntos de entrenamiento
(train) y prueba (test).
:::

::: {.cell .code execution_count="10"}
``` python
import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(np.arange(len(dataset)),test_size=0.15, random_state=1234, 
                                       shuffle=True, stratify=dataset.labels)

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)
```
:::

::: {.cell .markdown}
### Definición del modelo

Defina e implemente un modelo apropiado para el problema. Justifique sus
decisiones de diseño.
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .markdown}
### Entrenamiento del modelo

Entrene el modelo y muestre las curvas de aprendizaje. Justifique la
elección de hiperparámetros.
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .markdown}
### Evaluación del modelo

Evalúe el modelo en el conjunto de prueba usando matriz de confusión y
reporte de clasificación. Discuta los resultados.
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .markdown}
### Análisis de errores

Analice algunos ejemplos mal clasificados y comente lo que observa.
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .markdown}
### Aumentación de datos

Implemente un esquema de aumentación aleatoria de datos y compare los
resultados con y sin aumentación.
:::

::: {.cell .code}
``` python
```
:::
