# MineriaDeDatos-ClasificadorSupervisado
Proyecto grupal para la asignatura de Minería de Datos del grado en Ingeniería Informática de Gestión y Sistemas de Información en la EHU.

## Branches
- baseline: contiene el baseline sobre el que experimentaremos los tres integrantes del grupo.
- marcos: experimentación de Marcos.
- ibai: experimentación de Ibai.
- keneth: experimentación de Keneth.

## Marcos:

Python recomendado: 3.12.

Instalar requirements.txt.
```
pip install -r requirements.txt
```
Datos - Reddit C-SSRS Suicide Dataset: https://zenodo.org/records/2667859

### Comandos
```
python experimentos_vader_xgb.py --help
```
```
python experimentos_vader_xgb.py -f .\500_Reddit_users_posts_labels.csv -t Post -l Label -s semilla.csv --conjunto-train full --runs 3
```