
# Dash
import dash
from dash import Dash, html, dcc, Input, Output, State, no_update, dash_table, MATCH, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Flask y Login
from flask import Flask, redirect, url_for, request

# Flask Caching
from flask_caching import Cache


# Data & Utilities
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import io
import uuid
import time
import pickle

# Visualización
import plotly.express as px
import plotly.graph_objects as go

# Series Temporales
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
from autots import AutoTS

# Base de Datos
import psycopg2
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Scikit-learn - Modelos
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Scikit-learn - Preprocesado y Selección de Características
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, VarianceThreshold, RFE
)
from sklearn.decomposition import PCA

# Scikit-learn - Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report, roc_curve,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# Otros Modelos
import xgboost as xgb
import lightgbm as lgb

# Sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Deep Learning - TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN


## Requisitos de instalación
"""Asegúrate de instalar las siguientes dependencias:

pip install dash
pip install dash-bootstrap-components
pip install cryptography
pip install flask
pip install flask-login
pip install flask-caching
pip install pandas
pip install numpy
pip install plotly
pip install scikit-learn
pip install psycopg2-binary
pip install lightgbm
pip install xgboost
pip install imbalanced-learn
pip install SQLAlchemy
pip install prophet
pip install statsmodels
pip install tensorflow
pip install keras-tcn
pip install autots
pip install statsforecast
"""


#Nota: Esto no es adecuado para entornos con muchos usuarios ni para servidores en producción a gran escala
import uuid
MODELS_MEMORY = {}



server = Flask(__name__)

app = dash.Dash(
    __name__, 
    server=server, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})
app.title = "Dashboard ML Pipeline"



# ----------------------------------------------------------
# ----------------------------------------------------------
# --- LAYOUT CON TABS PRINCIPALES ---
# ----------------------------------------------------------
# ----------------------------------------------------------
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard para Análisis de Datos en ML", className="text-center mb-4"), width=12)
    ]),

    dcc.Tabs(id="main-tabs", value="tab-1", children=[
        dcc.Tab(label="1. Carga y visualización de datos", value="tab-1"),
        dcc.Tab(label="2. Preprocesamiento", value="tab-2"),
        dcc.Tab(label="3. Entrenamiento y Evaluación de Modelos", value="tab-3"),
        dcc.Tab(label="4. Optimización de Modelos (Implementado en Futuras Versiones)", value="tab-4"),
    ]),

    # Contenedor dinámico para el contenido de cada tab principal
    html.Div(id="main-tab-content"),

    #almacenamiento global de datos
    dcc.Store(id="file-data-store"),  # Datos de archivo subido
    dcc.Store(id="db-data-store"),    # Datos de base de datos (varias tablas)
    dcc.Store(id="raw-data-store"),
    dcc.Store(id="cleaned-data-store"),
    dcc.Store(id="selected-data-store"),
    dcc.Store(id="trained-model-store"),
    dcc.Store(id="time-config-store"), # Guarda la configuración temporal elegida por el usuario
    dcc.Store(id="agg-store"), # Guarda un DataFrame NUEVO resultante de aplicar resample + agregaciones temporales

], fluid=True)


# ----------------------------------------------------------
# ----------------------------------------------------------
# --- TAB 1 Carga de Datos ---
# ----------------------------------------------------------
# ----------------------------------------------------------
def layout_tab1():
    return dbc.Container([
        html.H3("1. Carga de Datos"),

        # ------------- SUBIDA DE ARCHIVOS -------------
        html.Div([
            html.Label("Subir Archivo"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    "Arrastra o haz click para seleccionar un archivo"
                ]),
                style={
                    'width': '80%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='file-upload-feedback', style={'color': 'green'})
        ]),

        # ------------- CONEXIÓN A BASE DE DATOS -------------
        html.Hr(),
        html.Label("Conexión a Base de Datos"),
        dbc.Row([
            dbc.Col(dbc.Input(id='db-host', placeholder='Host o IP', type='text')),
            dbc.Col(dbc.Input(id='db-port', placeholder='Puerto (ej: 5432)', type='number')),
        ]),
        dbc.Row([
            dbc.Col(dbc.Input(id='db-name', placeholder='Nombre de la Base de Datos (ej: postgres)', type='text')),
            dbc.Col(dbc.Input(id='db-schema', placeholder='Nombre del esquema (ej: public)', type='text')),
            dbc.Col(dbc.Input(id='db-user', placeholder='Usuario', type='text')),
            dbc.Col(dbc.Input(id='db-pass', placeholder='Contraseña', type='password')),
            dbc.Col(dbc.Button("Conectar", id='connect-db-button', color='primary')),
        ]),

        html.Div(id='db-connection-feedback', style={'color': 'blue'}),
        html.Br(),
        html.Label("Selecciona una tabla para trabajar:"),
        dcc.Dropdown(id='db-table-selector', placeholder="Tabla de la base de datos"),

        # ------------- VISTA PREVIA -------------
        html.Hr(),
        html.H4("Análisis Exploratorio del Dataset (sin limpiar)"),
        html.Div(id='data-exploration-preview'),

        # ------------- VISTA PREVIA SSTT -------------
        html.Hr(),
        html.H4("Análisis Temporal Simple: Serie vs Suavizado"),

        dbc.Row([
            dbc.Col([
                html.Label("Selecciona una columna numérica:"),
                dcc.Dropdown(id="serie-suavizada-columna", placeholder="Ej: amp1rx")
            ], width=6),

            dbc.Col([
                html.Label("Ventana de suavizado (en puntos):"),
                dcc.Input(id="suavizado-ventana", type="number", value=6, min=1, step=1)
            ], width=3)
        ]),

        html.Br(),
        dbc.Button("Graficar Serie Cruda vs Suavizada", id="graficar-suavizado-btn", color="primary"),
        html.Div(id="suavizado-feedback", style={"marginTop": "10px", "color": "green"}),
        dcc.Loading(dcc.Graph(id="grafico-suavizado"), type="default"),

        html.Hr(),
        html.H4("Descomposición Estacional STL"),

        dbc.Row([
            dbc.Col([
                html.Label("Selecciona una columna temporal:"),
                dcc.Dropdown(id="stl-columna", placeholder="Ej: amp1rx")
            ], width=6),

            dbc.Col([
                html.Label("Periodo estacional (número de puntos):"),
                dcc.Input(id="stl-periodo", type="number", value=24, min=2, step=1)
            ], width=3)
        ]),

        html.Br(),
        dbc.Button("Aplicar STL", id="stl-boton", color="secondary"),
        html.Div(id="stl-feedback", style={"marginTop": "10px", "color": "green"}),
        dcc.Loading(dcc.Graph(id="stl-grafico"), type="default"),

        html.Hr(),
        html.H4("Análisis de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)"),

        dbc.Row([
            dbc.Col([
                html.Label("Selecciona una columna:"),
                dcc.Dropdown(id="acf-pacf-columna", placeholder="Ej: amp1rx")
            ], width=6),

            dbc.Col([
                html.Label("Número máximo de lags (desplazamientos):"),
                dcc.Input(id="acf-pacf-lags", type="number", value=48, min=1, step=1)
            ], width=3)
        ]),

        html.Br(),
        dbc.Button("Calcular ACF y PACF", id="acf-pacf-boton", color="dark"),
        html.Div(id="acf-pacf-feedback", style={"marginTop": "10px", "color": "green"}),

        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id="acf-grafico")), width=6),
            dbc.Col(dcc.Loading(dcc.Graph(id="pacf-grafico")), width=6),
        ]),
        html.Hr(),
        html.H4("Distribuciones por Periodo (Boxplot)"),

        dbc.Row([
            dbc.Col([
                html.Label("Selecciona una columna numérica:"),
                dcc.Dropdown(id="boxplot-columna", placeholder="Ej: amp1rx")
            ], width=6),

            dbc.Col([
                html.Label("Agrupar por:"),
                dcc.Dropdown(
                    id="boxplot-agrupar",
                    options=[
                        {"label": "Mes", "value": "month"},
                        {"label": "Día de la semana", "value": "weekday"},
                        {"label": "Hora del día", "value": "hour"},
                    ],
                    placeholder="Selecciona agrupación"
                )
            ], width=4)
        ]),

        html.Br(),
        dbc.Button("Generar Boxplot", id="boxplot-boton", color="info"),
        html.Div(id="boxplot-feedback", style={"marginTop": "10px", "color": "green"}),
        dcc.Loading(dcc.Graph(id="boxplot-grafico"), type="default"),

        html.Hr(),
        html.H4("Mapa de Calor: Día de la Semana vs Hora del Día"),

        dbc.Row([
            dbc.Col([
                html.Label("Selecciona una columna numérica:"),
                dcc.Dropdown(id="heatmap-columna", placeholder="Ej: amp1rx")
            ], width=6),
        ]),

        html.Br(),
        dbc.Button("Generar Heatmap", id="heatmap-boton", color="warning"),
        html.Div(id="heatmap-feedback", style={"marginTop": "10px", "color": "green"}),
        dcc.Loading(dcc.Graph(id="heatmap-grafico"), type="default"),

        html.Hr(),
        html.H3("Correlación entre series"),
        html.Div([
            html.Div([
                html.Label("Serie A"),
                dcc.Dropdown(id="corr-col-a", placeholder="Selecciona serie A", options=[]),
            ], style={"width":"33%", "display":"inline-block", "paddingRight":"8px"}),
            html.Div([
                html.Label("Serie B"),
                dcc.Dropdown(id="corr-col-b", placeholder="Selecciona serie B", options=[]),
            ], style={"width":"33%", "display":"inline-block", "paddingRight":"8px"}),
            html.Div([
                html.Label("Método"),
                dcc.Dropdown(
                    id="corr-method",
                    options=[
                        {"label":"Pearson", "value":"pearson"},
                        {"label":"Spearman", "value":"spearman"},
                        {"label":"DTW", "value":"dtw"},
                    ],
                    value="pearson"
                )
            ], style={"width":"33%", "display":"inline-block"}),
        ], style={"marginBottom":"8px"}),

        html.Div([
            html.Div([
                html.Label("Ventana DTW (Sakoe-Chiba, en lags)"),
                dcc.Slider(id="dtw-window", min=0, max=48, step=1, value=0,
                        marks={0:"0", 12:"12", 24:"24", 36:"36", 48:"48"})
            ], style={"width":"49%", "display":"inline-block", "verticalAlign":"top"}),
            html.Div([
                html.Label("Resample (opcional)"),
                dcc.Dropdown(
                    id="corr-resample",
                    options=[
                        {"label":"(sin resample)", "value":""},
                        {"label":"15 minutos", "value":"15T"},
                        {"label":"30 minutos", "value":"30T"},
                        {"label":"1 hora", "value":"H"},
                        {"label":"1 día", "value":"D"},
                    ],
                    value=""
                ),
                html.Label("Rango de fechas (opcional)"),
                dcc.DatePickerRange(id="corr-date-range")
            ], style={"width":"49%", "display":"inline-block", "paddingLeft":"8px", "verticalAlign":"top"}),
        ], style={"marginBottom":"8px"}),

        html.Button("Calcular correlación", id="corr-run-btn"),
        html.Div(id="corr-result", style={"marginTop":"10px", "fontWeight":"bold"}),
        dcc.Graph(id="corr-aligned-graph"),

    ], fluid=True)

#TAB 1 sub-tabs

# Función auxiliar para convertir base64 -> DataFrame
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'json' in filename.lower():
            df = pd.read_json(io.BytesIO(decoded))
        else:
            return None
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None

    return df

@app.callback(
    Output('file-data-store', 'data'),
    Output('file-upload-feedback', 'children'),
    Output('raw-data-store', 'data'),
    Input('upload-data', 'contents'),
    Input('db-table-selector', 'value'),
    State('upload-data', 'filename'),
    State('db-data-store', 'data'),
    prevent_initial_call=True
)
def manejar_fuente_datos(file_contents, tabla_elegida, filename, db_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # === Subida de archivo ===
    if trigger_id == 'upload-data':
        if file_contents is None:
            return no_update, no_update, no_update

        df = parse_contents(file_contents, filename)

        if df is None or df.empty:
            return {}, f"Error: el archivo '{filename}' no es válido o está vacío.", {}

        data_json = df.to_json(date_format='iso', orient='split')
        return data_json, f"Archivo '{filename}' cargado con {len(df)} filas.", data_json

    # === Selección desde base de datos ===
    elif trigger_id == 'db-table-selector':
        if not tabla_elegida or not db_data:
            return no_update, no_update, no_update

        json_data = db_data.get(tabla_elegida)
        if not json_data:
            return {}, "No se pudo obtener la tabla seleccionada.", {}

        return no_update, f"Tabla '{tabla_elegida}' cargada correctamente.", json_data

    return no_update, no_update, no_update


#Callbacks para cargar todas las tablas PostgreSQL
@app.callback(
    Output('db-connection-feedback', 'children'),
    Output('db-table-selector', 'options'),
    Output('db-data-store', 'data'),
    Input('connect-db-button', 'n_clicks'),
    State('db-host', 'value'),
    State('db-port', 'value'),
    State('db-name', 'value'),
    State('db-user', 'value'),
    State('db-pass', 'value'),
    State('db-schema', 'value'),
    prevent_initial_call=True
)
def conectar_y_listar_tablas(n_clicks, host, port, dbname, user, password, schema):
    from urllib.parse import quote_plus

    if not all([host, port, dbname, user, password]):
        return "Por favor completa todos los campos (el esquema es opcional).", [], {}

    password_safe = quote_plus(password.strip())
    schema = schema.strip() if schema else None  # Normaliza

    url = f"postgresql+psycopg2://{user}:{password_safe}@{host}:{port}/{dbname}"
    #url = "postgresql+psycopg2://uc3m-giaa:fKklRSXUFp7n%40bOkC%23BCZ?Yxwk0P6d@stowai-azuredb-postgres.postgres.database.azure.com:5432/stowaidb_gss_showroom_0001"

    try:
        engine = create_engine(url)
        
        # Consulta de tablas
        query_tablas = """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
        """
        if schema:
            query_tablas += f" AND table_schema = '{schema}'"

        tablas = pd.read_sql(query_tablas, engine)

        if tablas.empty:
            return f"No se encontraron tablas en el esquema '{schema or 'public'}'.", [], {}

        tablas_data = {}
        dropdown_options = []

        for _, row in tablas.iterrows():
            esquema = row["table_schema"]
            tabla = row["table_name"]
            try:
                query = f'SELECT * FROM "{esquema}"."{tabla}" LIMIT 1000;'
                df = pd.read_sql_query(query, engine)
                tablas_data[f"{esquema}.{tabla}"] = df.to_json(date_format='iso', orient='split')
                dropdown_options.append({"label": f"{esquema}.{tabla}", "value": f"{esquema}.{tabla}"})
            except Exception as e:
                print(f"Error leyendo {esquema}.{tabla}: {e}")

        return f"{len(dropdown_options)} tablas encontradas en el esquema '{schema or 'detectado'}'.", dropdown_options, tablas_data

    except Exception as e:
        return f"Error al conectar o leer tablas: {e}", [], {}


# ----------------------------------------------------------
#Callback mostrar preview
"""Sección | Qué muestra
 Tipos de datos | df.dtypes
 Dimensiones | df.shape
 Primeras y últimas filas | df.head(), df.tail()
 Resumen estadístico | df.describe()
 Valores nulos | df.isnull().sum()
 Valores únicos por columna | df.nunique()
 Duplicados | df.duplicated().sum()"""

# Versión simplificada del serializador
def serialize_df(df):
    return df.astype(str)

@app.callback(
    Output('data-exploration-preview', 'children'),
    Input('raw-data-store', 'data'),
    prevent_initial_call=True
)
def explorar_dataset(data_json):
    if not data_json:
        return html.Div("No hay datos cargados.")

    try:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    except Exception as e:
        return html.Div(f"Error al cargar datos: {e}")

    children = []

    # 1. Dimensiones
    children.append(html.Div([
        html.H5("Dimensiones"),
        html.P(f"{df.shape[0]} filas × {df.shape[1]} columnas")
    ], style={"marginTop": "30px"}))

    # 2. Tipos de datos
    tipos = pd.DataFrame(df.dtypes.astype(str), columns=["Tipo"]).reset_index().rename(columns={"index": "Columna"})
    tipos = serialize_df(tipos)
    children.append(html.Div([
        html.H5("Tipos de Datos"),
        dash_table.DataTable(
            data=tipos.to_dict('records'),
            columns=[{"name": i, "id": i} for i in tipos.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
    ], style={"marginTop": "30px"}))

    # 3. Primeras filas
    head_df = serialize_df(df.head(5).copy())
    children.append(html.Div([
        html.H5("Primeras filas"),
        dash_table.DataTable(
            data=head_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in head_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
    ], style={"marginTop": "30px"}))

    # 4. Últimas filas
    tail_df = serialize_df(df.tail(5).copy())
    children.append(html.Div([
        html.H5("Últimas filas"),
        dash_table.DataTable(
            data=tail_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in tail_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
    ], style={"marginTop": "30px"}))

    # 5. Resumen estadístico
    children.append(html.Div([
        html.H5("Resumen Estadístico (columnas numéricas)"),
    ], style={"marginTop": "30px"}))
    try:
        describe_df = df.describe().transpose().reset_index().rename(columns={"index": "Columna"})
        describe_df = serialize_df(describe_df.replace({np.nan: None}))
        children.append(html.Div([
            dash_table.DataTable(
                data=describe_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in describe_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'}
            )
        ]))
    except Exception as e:
        children.append(html.P(f"Error al calcular descripción: {e}"))

    # 6. Nulos por columna
    nulos = df.isnull().sum().reset_index()
    nulos.columns = ["Columna", "Nulos"]
    children.append(html.Div([
        html.H5("Valores Nulos por Columna"),
        dash_table.DataTable(
            data=nulos.to_dict('records'),
            columns=[{"name": i, "id": i} for i in nulos.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
    ], style={"marginTop": "30px"}))

    # 7. Valores únicos por columna
    unicos = df.nunique().reset_index()
    unicos.columns = ["Columna", "Valores Únicos"]
    children.append(html.Div([
        html.H5("Valores Únicos por Columna"),
        dash_table.DataTable(
            data=unicos.to_dict('records'),
            columns=[{"name": i, "id": i} for i in unicos.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
    ], style={"marginTop": "30px"}))

    # 8. Filas duplicadas
    duplicados = df.duplicated().sum()
    children.append(html.Div([
        html.H5("Filas Duplicadas"),
        html.P(f"{duplicados} fila(s) duplicada(s) encontradas.")
    ], style={"marginTop": "30px"}))

    return children


# ------------- VISTA PREVIA SERIES TEMPORALES -------------
# Callback para poblar el dropdown de columnas numéricas
@app.callback(
    Output("serie-suavizada-columna", "options"),
    Input("raw-data-store", "data"),
    Input("cleaned-data-store", "data")
)
def actualizar_columnas_numericas(raw_data, cleaned_data):
    if not raw_data and not cleaned_data:
        return []

    import pandas as pd, io
    source = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(source), orient="split")

    cols = df.select_dtypes(include="number").columns
    return [{"label": col, "value": col} for col in cols]

# Callback para generar la gráfica cruda vs suavizada
@app.callback(
    Output("grafico-suavizado", "figure"),
    Output("suavizado-feedback", "children"),
    Input("graficar-suavizado-btn", "n_clicks"),
    State("serie-suavizada-columna", "value"),
    State("suavizado-ventana", "value"),
    State("raw-data-store", "data"),
    State("cleaned-data-store", "data"),
    prevent_initial_call=True
)
def graficar_suavizado(n_clicks, columna, ventana, raw_data, cleaned_data):
    import pandas as pd, plotly.graph_objects as go, io

    if not columna or not ventana:
        return dash.no_update, "Debes seleccionar una columna y una ventana válida."

    data = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(data), orient="split")

    if "time" not in df.columns:
        return dash.no_update, "No se encontró columna 'time'. Asegúrate de que existe."

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    df = df[[columna, "time"]].dropna()

    df["suavizado"] = df[columna].rolling(window=ventana).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df[columna],
                             mode="lines", name="Cruda", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df["time"], y=df["suavizado"],
                             mode="lines", name=f"Suavizado ({ventana})", line=dict(color="blue")))

    fig.update_layout(title=f"Serie temporal: {columna} vs suavizado",
                      xaxis_title="Tiempo", yaxis_title=columna,
                      template="plotly_white")

    return fig, "Gráfico generado correctamente."

#callback enlazado al dropdown
@app.callback(
    Output("stl-columna", "options"),
    Input("raw-data-store", "data"),
    Input("cleaned-data-store", "data")
)
def actualizar_columnas_stl(raw_data, cleaned_data):
    if not raw_data and not cleaned_data:
        return []

    import pandas as pd, io
    source = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(source), orient="split")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    return [{"label": col, "value": col} for col in num_cols]
#Callback principal: aplica STL y genera gráfico
@app.callback(
    Output("stl-grafico", "figure"),
    Output("stl-feedback", "children"),
    Input("stl-boton", "n_clicks"),
    State("stl-columna", "value"),
    State("stl-periodo", "value"),
    State("raw-data-store", "data"),
    State("cleaned-data-store", "data"),
    prevent_initial_call=True
)
def aplicar_stl(n_clicks, columna, periodo, raw_data, cleaned_data):
    import pandas as pd, io
    from statsmodels.tsa.seasonal import STL
    import plotly.graph_objects as go

    if not columna or not periodo:
        return dash.no_update, "⚠️ Selecciona una columna y un periodo estacional."

    data = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(data), orient="split")

    if "time" not in df.columns:
        return dash.no_update, "⚠️ No se encontró la columna 'time'."

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    serie = df[[columna, "time"]].dropna()

    serie.set_index("time", inplace=True)
    # Resampleo a frecuencia horaria por defecto (puedes ajustar)
    serie = serie.resample("1H").mean().interpolate()

    try:
        resultado = STL(serie[columna], period=periodo).fit()
    except Exception as e:
        return dash.no_update, f"Error aplicando STL: {e}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=resultado.observed, name="Original"))
    fig.add_trace(go.Scatter(x=serie.index, y=resultado.trend, name="Tendencia"))
    fig.add_trace(go.Scatter(x=serie.index, y=resultado.seasonal, name="Estacionalidad"))
    fig.add_trace(go.Scatter(x=serie.index, y=resultado.resid, name="Residuo"))

    fig.update_layout(
        title=f"STL para {columna} (periodo={periodo})",
        xaxis_title="Tiempo", yaxis_title=columna,
        template="plotly_white"
    )

    return fig, "Descomposición STL realizada correctamente."
#desplegable 
@app.callback(
    Output("acf-pacf-columna", "options"),
    Input("raw-data-store", "data"),
    Input("cleaned-data-store", "data")
)
def actualizar_columnas_para_acf(raw_data, cleaned_data):
    import pandas as pd, io

    if not raw_data and not cleaned_data:
        return []

    source = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(source), orient="split")
    columnas_numericas = df.select_dtypes(include=["number"]).columns.tolist()

    return [{"label": col, "value": col} for col in columnas_numericas]



#Callback para generar los gráficos ACF y PACF
@app.callback(
    Output("acf-grafico", "figure"),
    Output("pacf-grafico", "figure"),
    Output("acf-pacf-feedback", "children"),
    Input("acf-pacf-boton", "n_clicks"),
    State("acf-pacf-columna", "value"),
    State("acf-pacf-lags", "value"),
    State("raw-data-store", "data"),
    State("cleaned-data-store", "data"),
    prevent_initial_call=True
)
def calcular_acf_pacf(n_clicks, columna, lags, raw_data, cleaned_data):
    import pandas as pd, io
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    if not columna or not lags:
        return dash.no_update, dash.no_update, "Selecciona una columna y número de lags."

    data = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(data), orient="split")

    if "time" not in df.columns:
        return dash.no_update, dash.no_update, "Columna 'time' no encontrada."

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    serie = df[[columna, "time"]].dropna()
    serie.set_index("time", inplace=True)
    serie = serie.resample("1H").mean().interpolate()

    # --- Cálculo ACF
    from statsmodels.tsa.stattools import acf, pacf

    valores = serie[columna].values

    try:
        acf_vals = acf(valores, nlags=lags)
        pacf_vals = pacf(valores, nlags=lags)
    except Exception as e:
        return dash.no_update, dash.no_update, f"Error: {e}"

    lags_range = list(range(len(acf_vals)))

    # --- Gráfico ACF
    acf_fig = go.Figure()
    acf_fig.add_trace(go.Bar(x=lags_range, y=acf_vals, name="ACF"))
    acf_fig.update_layout(title=f"Función de Autocorrelación (ACF) - {columna}",
                          xaxis_title="Lag (horas)", yaxis_title="Autocorrelación",
                          template="plotly_white")

    # --- Gráfico PACF
    pacf_fig = go.Figure()
    pacf_fig.add_trace(go.Bar(x=lags_range, y=pacf_vals, name="PACF", marker_color="green"))
    pacf_fig.update_layout(title=f"Función de Autocorrelación Parcial (PACF) - {columna}",
                           xaxis_title="Lag (horas)", yaxis_title="Autocorrelación parcial",
                           template="plotly_white")

    return acf_fig, pacf_fig, "ACF y PACF calculados correctamente."

#callback para poblar el dropdown de columnas
@app.callback(
    Output("boxplot-columna", "options"),
    Input("raw-data-store", "data"),
    Input("cleaned-data-store", "data")
)
def actualizar_columnas_boxplot(raw_data, cleaned_data):
    import pandas as pd, io

    if not raw_data and not cleaned_data:
        return []

    source = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(source), orient="split")

    columnas_numericas = df.select_dtypes(include=["number"]).columns.tolist()
    return [{"label": col, "value": col} for col in columnas_numericas]

#Callback para generar el boxplot
@app.callback(
    Output("boxplot-grafico", "figure"),
    Output("boxplot-feedback", "children"),
    Input("boxplot-boton", "n_clicks"),
    State("boxplot-columna", "value"),
    State("boxplot-agrupar", "value"),
    State("raw-data-store", "data"),
    State("cleaned-data-store", "data"),
    prevent_initial_call=True
)
def generar_boxplot(n_clicks, columna, agrupamiento, raw_data, cleaned_data):
    import pandas as pd, plotly.express as px, io

    if not columna or not agrupamiento:
        return dash.no_update, "Selecciona columna y agrupamiento."

    source = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(source), orient="split")

    if "time" not in df.columns:
        return dash.no_update, "La columna 'time' no está en el dataset."

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    df["month"] = df["time"].dt.month
    df["weekday"] = df["time"].dt.weekday
    df["hour"] = df["time"].dt.hour

    df_filtrado = df[[columna, agrupamiento]].dropna()

    fig = px.box(df_filtrado, x=agrupamiento, y=columna, points="outliers",
                 title=f"Distribución de {columna} por {agrupamiento.capitalize()}")

    # Etiquetas más claras
    if agrupamiento == "weekday":
        fig.update_xaxes(tickmode="array", tickvals=list(range(7)),
                         ticktext=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])

    return fig, "Boxplot generado correctamente."

#Callback para poblar el dropdown de columnas numéricas
@app.callback(
    Output("heatmap-columna", "options"),
    Input("raw-data-store", "data"),
    Input("cleaned-data-store", "data")
)
def actualizar_columnas_heatmap(raw_data, cleaned_data):
    import pandas as pd, io

    if not raw_data and not cleaned_data:
        return []

    source = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(source), orient="split")

    columnas_numericas = df.select_dtypes(include=["number"]).columns.tolist()
    return [{"label": col, "value": col} for col in columnas_numericas]

#Callback para generar el heatmap
@app.callback(
    Output("heatmap-grafico", "figure"),
    Output("heatmap-feedback", "children"),
    Input("heatmap-boton", "n_clicks"),
    State("heatmap-columna", "value"),
    State("raw-data-store", "data"),
    State("cleaned-data-store", "data"),
    prevent_initial_call=True
)
def generar_heatmap(n_clicks, columna, raw_data, cleaned_data):
    import pandas as pd, io
    import plotly.express as px
    import dash

    print("Botón clicado")  # debug básico

    if not columna:
        return dash.no_update, "Selecciona una columna."

    source = cleaned_data if cleaned_data else raw_data
    df = pd.read_json(io.StringIO(source), orient="split")

    if "time" not in df.columns:
        return dash.no_update, "La columna 'time' no está en el dataset."

    df["time"] = pd.to_datetime(df["time"])
    df["weekday"] = df["time"].dt.weekday
    df["hour"] = df["time"].dt.hour

    df_filtrado = df[[columna, "weekday", "hour"]].dropna()

    if df_filtrado.empty:
        return dash.no_update, "No hay datos suficientes para graficar."

    tabla = df_filtrado.pivot_table(index="weekday", columns="hour", values=columna, aggfunc="mean")

    if tabla.isnull().all().all():
        return dash.no_update, "Los valores están vacíos o son todos NaN."

    ticktext = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

    try:
        fig = px.imshow(
            tabla,
            labels=dict(x="Hora del Día", y="Día de la Semana", color=columna),
            x=tabla.columns,
            y=ticktext,
            aspect="auto",
            color_continuous_scale="Hot"
        )

        fig.update_layout(
            title=f"Mapa de Calor de {columna}",
            xaxis_title="Hora del Día",
            yaxis_title="Día de la Semana"
        )

        return fig, "Heatmap generado correctamente."
    except Exception as e:
        return dash.no_update, f"Error al generar heatmap: {e}"



def _simple_dtw(x: np.ndarray, y: np.ndarray, window: int | None = None) -> float:
    """
    DTW sencillo con banda de Sakoe-Chiba (ventana en número de lags).
    Complejidad O(n * window) aprox. si se limita la ventana.
    """
    n, m = len(x), len(y)
    if window is None or window <= 0:
        window = max(n, m)  # sin restricción efectiva
    window = max(window, abs(n - m))
    inf = float("inf")
    D = np.full((n + 1, m + 1), inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        xi = x[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(xi - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])

# 1) Poblar dropdowns con columnas numéricas
# --- helper ya existente en tu archivo ---
def get_latest_dataframe(cleaned_data, raw_data):
    import pandas as pd, io
    source = cleaned_data if cleaned_data else raw_data
    return pd.read_json(io.StringIO(source), orient="split")

# 1) Poblar dropdowns con columnas numéricas 
@app.callback(
    Output("corr-col-a", "options"),
    Output("corr-col-b", "options"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data"),
)
def _fill_corr_cols(cleaned_js, raw_js):
    import pandas as pd, io
    if not cleaned_js and not raw_js:
        return [], []
    df = get_latest_dataframe(cleaned_js, raw_js)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    opts = [{"label": c, "value": c} for c in num_cols]
    return opts, opts

# 2) Ejecutar correlación y graficar 
@app.callback(
    Output("corr-result", "children"),
    Output("corr-aligned-graph", "figure"),
    Input("corr-run-btn", "n_clicks"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    State("time-config-store", "data"),
    State("corr-col-a", "value"),
    State("corr-col-b", "value"),
    State("corr-method", "value"),
    State("dtw-window", "value"),
    State("corr-resample", "value"),
    State("corr-date-range", "start_date"),
    State("corr-date-range", "end_date"),
    prevent_initial_call=True
)
def _run_corr(n, cleaned_js, raw_js, time_cfg, col_a, col_b, method, dtw_window, resample_rule, start, end):
    import pandas as pd, plotly.graph_objects as go
    import numpy as np

    if (not cleaned_js and not raw_js) or not col_a or not col_b:
        return "Selecciona ambas series.", go.Figure()

    df = get_latest_dataframe(cleaned_js, raw_js)

    # Asegurar eje temporal
    if time_cfg and time_cfg.get("use_as_index", False) and isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        tcol = (time_cfg or {}).get("time_col")
        if tcol and tcol in df.columns:
            df = df.sort_values(tcol).set_index(tcol)
        else:
            return "Primero configura la columna temporal en Tab 2.", go.Figure()

    # Rango
    if start and end:
        df = df.loc[start:end]

    if col_a not in df.columns or col_b not in df.columns:
        return "Las columnas seleccionadas no existen en el dataset.", go.Figure()

    sA = df[col_a].astype(float)
    sB = df[col_b].astype(float)

    if resample_rule:
        sA = sA.resample(resample_rule).mean()
        sB = sB.resample(resample_rule).mean()

    aligned = pd.concat({"A": sA, "B": sB}, axis=1).dropna()
    if aligned.empty:
        return "Sin datos tras alinear/filtrar.", go.Figure()

    # Métrica
    if method in ("pearson", "spearman"):
        try:
            from scipy.stats import pearsonr, spearmanr
            r, p = (pearsonr if method=="pearson" else spearmanr)(aligned["A"], aligned["B"])
            result_text = f"Correlación {method.capitalize()}: {r:.3f}  |  p-valor: {p:.3g}"
        except Exception:
            r = aligned["A"].corr(aligned["B"], method=method)
            result_text = f"Correlación {method.capitalize()}: {r:.3f}"
    else:
        # DTW
        x = aligned["A"].to_numpy(); y = aligned["B"].to_numpy()
        result_text = f"DTW (ventana={int(dtw_window or 0)}): " + \
                      f"{_simple_dtw(x, y, window=int(dtw_window or 0)):.3f}"

    fig = go.Figure()
    fig.add_scatter(x=aligned.index, y=aligned["A"], mode="lines", name=col_a)
    fig.add_scatter(x=aligned.index, y=aligned["B"], mode="lines", name=col_b)
    ttl = f"Series alineadas en el tiempo ({'resample: '+resample_rule if resample_rule else 'sin resample'})"
    fig.update_layout(title=ttl, xaxis_title="Tiempo", yaxis_title="Valor", legend=dict(orientation="h"))
    return result_text, fig


# ----------------------------------------------------------
# ----------------------------------------------------------
## TAB 2: Preprocesamiento
# ----------------------------------------------------------
# ----------------------------------------------------------

def layout_tab2():
    return dbc.Container([

        html.H3("2. Preprocesamiento de Datos"),

        html.H5("Guía rápida de preprocesamiento de datos"),
        dash_table.DataTable(
            data=[
                {"Etapa": "1. Eliminar duplicados",
                "Descripción": "Remueve filas idénticas para evitar redundancia en el análisis.",
                "Cuándo usar": "Cuando hay filas exactas repetidas. Especialmente tras un join o concatenación.",
                "Precauciones": "Puede eliminar información si los duplicados no son errores."},

                {"Etapa": "2. Manejo de nulos",
                "Descripción": "Trata valores faltantes eliminándolos o imputándolos.",
                "Cuándo usar": "Siempre que haya columnas con valores NaN.",
                "Precauciones": "Eliminar muchas filas puede dañar el dataset. Imputar mal puede introducir sesgo."},

                {"Etapa": "3. Codificación",
                "Descripción": "Convierte variables categóricas en numéricas para modelos ML.",
                "Cuándo usar": "Con cualquier columna 'object' o 'category'.",
                "Precauciones": "LabelEncoder introduce orden implícito, OneHot puede crear muchas columnas."},

                {"Etapa": "4. Escalado",
                "Descripción": "Transforma variables numéricas a un rango comparable.",
                "Cuándo usar": "Antes de modelos sensibles a escala (regresión, SVM, etc).",
                "Precauciones": "No escalar columnas categóricas codificadas."},

                {"Etapa": "5. Selección de columnas",
                "Descripción": "Elimina columnas irrelevantes o con alta correlación.",
                "Cuándo usar": "Con muchas variables o redundancia.",
                "Precauciones": "Evitar eliminar variables clave sin análisis."},

                {"Etapa": "6. Feature engineering",
                "Descripción": "Crear nuevas variables útiles a partir de las existentes.",
                "Cuándo usar": "Cuando hay conocimiento del dominio o se identifican patrones.",
                "Precauciones": "Puede complicar el modelo si se abusa."},

                {"Etapa": "7. Balanceo",
                "Descripción": "Ajusta la cantidad de clases en clasificación.",
                "Cuándo usar": "Cuando hay clases desbalanceadas (ej. 90%-10%).",
                "Precauciones": "SMOTE puede crear ejemplos artificiales irreales."},

                {"Etapa": "8. Conversión de tipos",
                "Descripción": "Transforma tipos de datos para análisis correcto.",
                "Cuándo usar": "Fechas como texto, categóricas como numéricas.",
                "Precauciones": "Conversión errónea puede ocultar errores."},

                {"Etapa": "9. Validación de datos",
                "Descripción": "Comprueba la coherencia del dataset tras los cambios.",
                "Cuándo usar": "Siempre después de cualquier transformación.",
                "Precauciones": "Paso obligatorio antes de modelar."}
            ],
            columns=[{"name": i, "id": i} for i in ["Etapa", "Descripción", "Cuándo usar", "Precauciones"]],
            style_table={"overflowX": "auto"},
            style_cell={'textAlign': 'left', 'font_size': '14px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
        ),
        
        html.Hr(),
        dbc.Card([
            dbc.CardHeader(html.H5("1. Manejo de valores nulos")),

            dbc.CardBody([

                html.P("Esta herramienta te permite tratar valores faltantes (NaN) en columnas seleccionadas. Puedes eliminarlos o imputarlos con la media o mediana."),

                html.Div(id="null-columns-preview"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Selecciona columnas con nulos:"),
                        dcc.Dropdown(id="null-columns-dropdown", multi=True),
                    ]),
                    dbc.Col([
                        html.Label("Método de imputación:"),
                        dcc.RadioItems(
                            id="null-handling-method",
                            options=[
                                {"label": "Eliminar filas con nulos", "value": "drop"},
                                {"label": "Imputar con media", "value": "mean"},
                                {"label": "Imputar con mediana", "value": "median"}
                            ],
                            value="drop",
                            inline=False
                        )
                    ])
                ]),

                html.Br(),
                dbc.Button("Aplicar tratamiento de nulos", id="apply-null-handling", color="primary"),
                html.Div(id="null-handling-feedback", style={"marginTop": "10px", "color": "green"}),
                html.Div(id="null-handling-preview", style={"marginTop": "20px"}),

            ])
        ], className="mb-4"),

        dbc.Card([
            dbc.CardHeader(html.H5("2. Eliminación de duplicados")),

            dbc.CardBody([

                html.P("Las filas duplicadas son registros exactamente iguales (en todas o algunas columnas). Esta herramienta te permite eliminarlas para evitar redundancia y errores en el análisis."),

                html.Div(id="duplicates-info"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Selecciona columnas para buscar duplicados (opcional):"),
                        dcc.Dropdown(id="duplicate-columns-dropdown", multi=True),
                    ]),
                    dbc.Col([
                        dbc.Checklist(
                            options=[{"label": "Eliminar duplicados", "value": "drop"}],
                            value=[],
                            id="drop-duplicates-checklist",
                            switch=True,
                            inline=True
                        ),
                    ], width=3, style={"marginTop": "30px"})
                ]),

                html.Br(),
                dbc.Button("Aplicar eliminación de duplicados", id="apply-duplicates-btn", color="primary"),
                html.Div(id="duplicates-feedback", style={"marginTop": "10px", "color": "green"}),
                html.Div(id="duplicates-preview", style={"marginTop": "20px"})

            ])
        ], className="mb-4"),

        dbc.Card([
            dbc.CardHeader(html.H5("3. Codificación de variables categóricas")),

            dbc.CardBody([

                html.P("Los modelos de Machine Learning requieren datos numéricos. Aquí puedes convertir variables categóricas en números mediante codificación."),
                html.Div(id="categorical-columns-info"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Selecciona columnas categóricas:"),
                        dcc.Dropdown(id="categorical-columns-dropdown", multi=True),
                    ]),
                    dbc.Col([
                        html.Label("Tipo de codificación:"),
                        dcc.RadioItems(
                            id="encoding-method",
                            options=[
                                {"label": "Label Encoding", "value": "label"},
                                {"label": "One-Hot Encoding", "value": "onehot"},
                            ],
                            value="onehot",
                            inline=True
                        )
                    ])
                ]),

                html.Br(),
                dbc.Button("Aplicar codificación", id="apply-encoding-btn", color="primary"),
                html.Div(id="encoding-feedback", style={"marginTop": "10px", "color": "green"}),
                html.Div(id="encoding-preview", style={"marginTop": "20px"})

            ])
        ], className="mb-4"),

        dbc.Card([
            dbc.CardHeader(html.H5("4. Escalado de variables numéricas")),

            dbc.CardBody([

                html.P("Los modelos de Machine Learning sensibles a la escala requieren normalizar los datos. Aquí puedes aplicar distintas técnicas de escalado a columnas numéricas."),

                html.Div(id="scaling-columns-info"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Selecciona columnas numéricas:"),
                        dcc.Dropdown(id="scaling-columns-dropdown", multi=True),
                    ]),
                    dbc.Col([
                        html.Label("Método de escalado:"),
                        dcc.RadioItems(
                            id="scaling-method",
                            options=[
                                {"label": "StandardScaler", "value": "standard"},
                                {"label": "MinMaxScaler", "value": "minmax"},
                                {"label": "RobustScaler", "value": "robust"}
                            ],
                            value="standard",
                            inline=False
                        )
                    ])
                ]),

                html.Br(),
                dbc.Button("Aplicar escalado", id="apply-scaling-btn", color="primary"),
                html.Div(id="scaling-feedback", style={"marginTop": "10px", "color": "green"}),
                html.Div(id="scaling-preview", style={"marginTop": "20px"})

            ])
        ], className="mb-4"),

        dbc.Card([
            dbc.CardHeader(html.H5("5. Selección y transformación de características")),

            dbc.CardBody([

                html.P("En esta sección puedes eliminar columnas manualmente o aplicar métodos automáticos para seleccionar o transformar las variables más relevantes."),

                html.Label("Eliminar columnas manualmente:"),
                dcc.Dropdown(id="column-dropper-dropdown", multi=True),

                html.Br(),

                html.Label("Técnicas automáticas de selección/transf. (opcional):"),
                dcc.RadioItems(
                    id="feature-selection-method",
                    options=[
                        {"label": "Ninguna", "value": "none"},
                        {"label": "SelectKBest", "value": "kbest"},
                        {"label": "PCA", "value": "pca"},
                        {"label": "VarianceThreshold", "value": "variance"}
                    ],
                    value="none",
                    inline=True
                ),

                html.Div(id="feature-selection-params"),
                
                html.Br(),
                html.Label("Selecciona la variable objetivo (target):"),
                dcc.Dropdown(id="target-column-dropdown"),
                html.Br(),
                dbc.Button("Aplicar selección de características", id="apply-feature-selection-btn", color="primary"),
                html.Div(id="feature-selection-feedback", style={"marginTop": "10px", "color": "green"}),
                html.Div(id="feature-selection-preview", style={"marginTop": "20px"})

            ])
        ], className="mb-4"),

        dbc.Card([
            dbc.CardHeader(html.H5("6. Balanceo de clases (para clasificación)")),

            dbc.CardBody([

                html.P("Si tu variable objetivo está desbalanceada, puedes aplicar técnicas para equilibrar las clases y mejorar el rendimiento del modelo."),

                html.Label("Selecciona la variable objetivo:"),
                dcc.Dropdown(id="balancing-target-dropdown"),

                html.Br(),

                html.Div(id="class-distribution-preview"),

                html.Label("Técnica de balanceo:"),
                dcc.RadioItems(
                    id="balancing-method",
                    options=[
                        {"label": "SMOTE (sobre-muestreo)", "value": "smote"},
                        {"label": "Random UnderSampler", "value": "undersample"}
                    ],
                    value="smote",
                    inline=True
                ),

                html.Br(),
                dbc.Button("Aplicar balanceo", id="apply-balancing-btn", color="primary"),
                html.Div(id="balancing-feedback", style={"marginTop": "10px", "color": "green"}),
                html.Div(id="balancing-preview", style={"marginTop": "20px"})

            ])
        ], className="mb-4"),


        dbc.Card([
            dbc.CardHeader(html.H5("7. Detección y tratamiento de outliers")),

            dbc.CardBody([

                html.H6("Paso 1: Detección de outliers"),

                html.Label("Selecciona columnas numéricas:"),
                dcc.Dropdown(id="outlier-detection-columns", multi=True),

                html.Br(),

                html.Label("Método de detección:"),
                dcc.RadioItems(
                    id="outlier-detection-method",
                    options=[
                        {"label": "IQR", "value": "iqr"},
                        {"label": "Z-Score", "value": "zscore"}
                    ],
                    value="iqr"
                ),

                html.Div([
                    html.Label("Umbral Z-Score:"),
                    dcc.Input(id="zscore-threshold", type="number", value=3.0, step=0.1, min=0.5)
                ], id="zscore-threshold-container", style={"display": "none"}),

                html.Br(),
                dbc.Button("Detectar Outliers", id="detect-outliers-btn", color="primary"),
                html.Div(id="outlier-detection-feedback", className="mt-2"),
                html.Div(id="outlier-boxplot-preview", className="mt-3"),

                html.Hr(),

                html.H6("Paso 2: Tratamiento de outliers"),
                html.Label("Método de tratamiento:"),
                dcc.RadioItems(
                    id="outlier-treatment-method",
                    options=[
                        {"label": "Eliminar filas", "value": "remove"},
                        {"label": "Reemplazar por media", "value": "mean"},
                        {"label": "Reemplazar por mediana", "value": "median"},
                        {"label": "Marcar con columna binaria", "value": "mark"}
                    ],
                    value="remove"
                ),

                html.Br(),
                dbc.Button("Aplicar tratamiento", id="apply-outlier-treatment-btn", color="success"),
                html.Div(id="outlier-treatment-feedback", className="mt-2"),
                html.Div(id="outlier-treatment-preview", className="mt-3")
            ])
        ]),
        dcc.Store(id="outlier-index-store"),

                dbc.Card([
            dbc.CardHeader(html.H5("8. Conversión de tipos y validación de columnas")),

            dbc.CardBody([

                html.P("Revisa y corrige los tipos de datos de tus columnas para asegurar que sean tratados correctamente por los modelos."),

                html.Label("Columnas disponibles:"),
                dcc.Dropdown(id="type-columns-dropdown", multi=True),

                html.Br(),

                html.Label("Tipo al que convertir:"),
                dcc.RadioItems(
                    id="conversion-type",
                    options=[
                        {"label": "Numérico", "value": "numeric"},
                        {"label": "Texto (string)", "value": "string"},
                        {"label": "Categoría", "value": "category"},
                        {"label": "Fecha (datetime)", "value": "datetime"}
                    ],
                    value="string"
                ),

                html.Br(),
                dbc.Button("Aplicar conversión de tipo", id="apply-type-conversion-btn", color="primary"),
                html.Div(id="type-conversion-feedback", style={"marginTop": "10px", "color": "green"}),

                html.Hr(),
                html.Div(id="type-info-preview")

            ])
        ], className="mb-4"),
        html.Hr(),
        html.H3("Índice temporal"),
        html.Div([
            dcc.Dropdown(id="timeindex-col", placeholder="Selecciona columna temporal", options=[]),
            dcc.RadioItems(
                id="timeindex-tz",
                options=[
                    {"label": "UTC", "value": "UTC"},
                    {"label": "Europe/Madrid", "value": "Europe/Madrid"},
                ],
                value="Europe/Madrid",
                inline=True
            ),
            dcc.Checklist(
                id="timeindex-use",
                options=[{"label": "Usar como índice temporal para análisis", "value": "on"}],
                value=["on"]
            ),
            html.Button("Aplicar", id="timeindex-apply"),
            html.Div(id="timeindex-msg", style={"marginTop": "6px"})
        ]),
        html.Hr(),
        html.H3("Ingeniería temporal — Resample + Agregaciones"),
        html.Div([
            html.Div([
                html.Label("Frecuencia de remuestreo"),
                dcc.Dropdown(
                    id="resample-freq",
                    options=[
                        {"label": "15 minutos", "value": "15T"},
                        {"label": "30 minutos", "value": "30T"},
                        {"label": "1 hora", "value": "H"},
                        {"label": "1 día", "value": "D"},
                    ],
                    value="D",
                    clearable=False
                ),
            ], style={"marginBottom":"8px"}),

            html.Div([
                html.Label("Funciones de agregación"),
                dcc.Checklist(
                    id="agg-funcs",
                    options=[
                        {"label":"mean","value":"mean"},
                        {"label":"max","value":"max"},
                        {"label":"min","value":"min"},
                        {"label":"std","value":"std"},
                        {"label":"median","value":"median"},
                    ],
                    value=["mean","max"],
                    inline=True
                ),
            ], style={"marginBottom":"8px"}),

            html.Div([
                html.Label("Agrupar por (opcional, p.ej. sensor_id)"),
                dcc.Dropdown(id="groupby-col", placeholder="Selecciona columna de agrupación (opcional)"),
            ], style={"marginBottom":"8px"}),

            html.Div([
                html.Button("Generar agregados", id="build-agg-btn"),
                html.Span(id="agg-msg", style={"marginLeft":"10px"})
            ], style={"marginBottom":"10px"}),

            html.Div([
                html.Small("El resultado se guarda en 'agg-store' y se usará en el etiquetado y clasificación.")
            ])
        ]),
        html.Hr(),
        html.H3("Etiquetado automático por umbral"),
        html.Div([
            html.Div([
                html.Label("Columna numérica"),
                dcc.Dropdown(id="label-col", placeholder="Selecciona columna"),
            ], style={"marginBottom": "8px"}),

            html.Div([
                html.Label("Operador"),
                dcc.Dropdown(
                    id="label-operator",
                    options=[
                        {"label": ">=", "value": ">="},
                        {"label": ">",  "value": ">"},
                        {"label": "<=", "value": "<="},
                        {"label": "<",  "value": "<"},
                        {"label": "==", "value": "=="},
                        {"label": "between [min,max]", "value": "between"}
                    ],
                    value=">=",
                    clearable=False
                ),
            ], style={"marginBottom": "8px"}),

            html.Div(id="threshold-block", children=[
                dcc.Input(id="label-threshold", type="number", placeholder="Umbral", style={"width":"200px"}),
            ], style={"marginBottom": "8px"}),

            html.Div(
                id="threshold-between-block",
                children=[
                    dcc.Input(id="label-threshold-min", type="number", placeholder="Mín", style={"width":"120px", "marginRight":"6px"}),
                    dcc.Input(id="label-threshold-max", type="number", placeholder="Máx", style={"width":"120px"}),
                ],
                style={"display":"none", "marginBottom":"8px"}
            ),


            html.Div([
                html.Label("Nombre de la columna etiqueta"),
                dcc.Input(id="label-name", type="text", placeholder="ej: etiqueta_binaria", style={"width":"260px"}),
            ], style={"marginBottom": "8px"}),

            html.Div([
                dcc.Checklist(
                    id="label-invert",
                    options=[{"label": "Invertir (1 si NO se cumple la condición)", "value": "invert"}],
                    value=[]
                )
            ], style={"marginBottom": "8px"}),

            html.Button("Crear etiqueta", id="build-label-btn"),
            html.Span(id="label-msg", style={"marginLeft": "10px"})
        ]),


    ], fluid=True)


# TAB 2 sub-tabs
def get_latest_dataframe(cleaned_data, raw_data):
    import pandas as pd
    source = cleaned_data if cleaned_data else raw_data
    return pd.read_json(source, orient="split")

##Fase 1: Tratamiento de nulos
#Callback para actualizar el dropdown de columnas con nulos
@app.callback(
    Output("null-columns-dropdown", "options"),
    Output("null-columns-preview", "children"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data"),
)
def update_null_columns(cleaned_data, raw_data):
    if not cleaned_data and not raw_data:
        return [], html.P("No hay datos cargados.", style={"color": "red"})

    try:
        data_source = cleaned_data if cleaned_data else raw_data
        df = pd.read_json(io.StringIO(data_source), orient="split")
    except Exception as e:
        return [], html.P(f"Error al leer los datos: {e}", style={"color": "red"})

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if cols_with_nulls.empty:
        return [], html.P("No hay columnas con valores nulos.", style={"color": "green"})

    options = [{"label": f"{col} ({cnt} nulos)", "value": col} for col, cnt in cols_with_nulls.items()]
    fig = go.Figure([go.Bar(x=cols_with_nulls.index, y=cols_with_nulls.values)])
    fig.update_layout(title="Valores nulos por columna")

    return options, dcc.Graph(figure=fig)



#Callback para aplicar tratamiento de nulos
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("null-handling-feedback", "children"),
    Output("null-handling-preview", "children"),
    Input("apply-null-handling", "n_clicks"),
    State("null-columns-dropdown", "value"),
    State("null-handling-method", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True
)
def handle_nulls(n_clicks, selected_cols, method, cleaned_data, raw_data):
    if not raw_data or not selected_cols:
        raise PreventUpdate

    try:
        df = get_latest_dataframe(cleaned_data, raw_data)  
    except Exception as e:
        return dash.no_update, f"Error al leer los datos: {e}", html.Div()

    if method == "drop":
        df = df.dropna(subset=selected_cols)
        msg = f"Se eliminaron las filas con nulos en: {', '.join(selected_cols)}"

    elif method == "mean":
        for col in selected_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
        msg = f"Se imputaron los nulos con la media en: {', '.join(selected_cols)}"

    elif method == "median":
        for col in selected_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
        msg = f"Se imputaron los nulos con la mediana en: {', '.join(selected_cols)}"

    else:
        return dash.no_update, "Método de imputación no válido", html.Div()

    # --- Vista previa
    preview = dash_table.DataTable(
        data=df.head().to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=5,
        style_table={"overflowX": "auto"}
    )

    # --- Exportar nuevamente en formato split
    df_json = df.to_json(date_format='iso', orient='split')

    return df_json, msg, preview

##  FASE 2: Eliminación de duplicados
#Callback Actualiza columnas disponibles y muestra cuántos duplicados hay
@app.callback(
    Output("duplicate-columns-dropdown", "options"),
    Output("duplicates-info", "children"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def update_duplicate_columns(cleaned_data, raw_data):
    if not cleaned_data and not raw_data:
        return [], html.P("No hay datos cargados.", style={"color": "red"})

    try:
        data_source = cleaned_data if cleaned_data else raw_data
        df = pd.read_json(io.StringIO(data_source), orient="split")
    except Exception as e:
        return [], html.P(f"Error al leer los datos: {e}", style={"color": "red"})

    dup_total = df.duplicated().sum()
    info = html.P(f"🔁 Se encontraron {dup_total} filas duplicadas (basado en todas las columnas).")

    options = [{"label": col, "value": col} for col in df.columns]
    return options, info


#Callback Aplicar eliminación de duplicados
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("duplicates-feedback", "children"),
    Output("duplicates-preview", "children"),
    Input("apply-duplicates-btn", "n_clicks"),
    State("drop-duplicates-checklist", "value"),
    State("duplicate-columns-dropdown", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True
)
def handle_duplicates(n_clicks, drop_checklist, selected_columns, cleaned_data, raw_data):
    if not raw_data or "drop" not in drop_checklist:
        raise PreventUpdate

    try:
        df = get_latest_dataframe(cleaned_data, raw_data)
    except Exception as e:
        return dash.no_update, f"Error al leer los datos: {e}", html.Div()

    initial_rows = df.shape[0]

    if selected_columns:
        df = df.drop_duplicates(subset=selected_columns)
        cols_str = ", ".join(selected_columns)
        msg = f"Se eliminaron duplicados considerando las columnas: {cols_str}."
    else:
        df = df.drop_duplicates()
        msg = "Se eliminaron duplicados en todas las columnas."

    final_rows = df.shape[0]
    removed = initial_rows - final_rows
    msg += f" 🔍 Filas eliminadas: {removed}."

    preview = dash_table.DataTable(
        data=df.head().to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=5,
        style_table={"overflowX": "auto"}
    )

    df_json = df.to_json(date_format="iso", orient="split")
    return df_json, msg, preview

## Fase 3: Codificación de variables categóricas
#CALLBACK Mostrar columnas categóricas
@app.callback(
    Output("categorical-columns-dropdown", "options"),
    Output("categorical-columns-info", "children"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def update_categorical_columns(cleaned_data, raw_data):
    if not cleaned_data and not raw_data:
        return [], html.P("No hay datos cargados.", style={"color": "red"})

    try:
        data_source = cleaned_data if cleaned_data else raw_data
        df = pd.read_json(io.StringIO(data_source), orient="split")
    except Exception as e:
        return [], html.P(f"Error al leer los datos: {e}", style={"color": "red"})

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        return [], html.P("No se detectaron columnas categóricas.", style={"color": "green"})

    options = [{"label": col, "value": col} for col in categorical_cols]
    msg = f"📦 Columnas categóricas detectadas: {', '.join(categorical_cols)}"

    return options, html.P(msg)

#CALLBACK Aplicar codificación
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("encoding-feedback", "children"),
    Output("encoding-preview", "children"),
    Input("apply-encoding-btn", "n_clicks"),
    State("categorical-columns-dropdown", "value"),
    State("encoding-method", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True
)
def apply_encoding(n_clicks, selected_columns, encoding_method, cleaned_data, raw_data):
    if not raw_data or not selected_columns:
        raise PreventUpdate

    try:
        df = get_latest_dataframe(cleaned_data, raw_data)
    except Exception as e:
        return dash.no_update, f"Error al leer los datos: {e}", html.Div()

    try:
        if encoding_method == "label":
            for col in selected_columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            msg = f"Se aplicó Label Encoding a: {', '.join(selected_columns)}"

        elif encoding_method == "onehot":
            df = pd.get_dummies(df, columns=selected_columns)
            msg = f"Se aplicó One-Hot Encoding a: {', '.join(selected_columns)}"

        else:
            return dash.no_update, "Método de codificación no válido.", html.Div()

    except Exception as e:
        return dash.no_update, f"Error durante la codificación: {e}", html.Div()

    # Vista previa
    preview = dash_table.DataTable(
        data=df.head().to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=5,
        style_table={"overflowX": "auto"}
    )

    return df.to_json(date_format="iso", orient="split"), msg, preview

##Fase4: : Escalado de variables numéricas
#Callback Mostrar columnas numéricas disponibles
@app.callback(
    Output("scaling-columns-dropdown", "options"),
    Output("scaling-columns-info", "children"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def update_numeric_columns(cleaned_data, raw_data):
    import io
    import pandas as pd

    if not cleaned_data and not raw_data:
        return [], html.P("No hay datos cargados.", style={"color": "red"})

    try:
        data_source = cleaned_data if cleaned_data else raw_data
        df = pd.read_json(io.StringIO(data_source), orient="split")
    except Exception as e:
        return [], html.P(f"Error al leer los datos: {e}", style={"color": "red"})

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not num_cols:
        return [], html.P("No se detectaron columnas numéricas.", style={"color": "green"})

    options = [{"label": col, "value": col} for col in num_cols]
    msg = f"Columnas numéricas detectadas: {', '.join(num_cols)}"

    return options, html.P(msg)

#Callback Aplicar escalado
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("scaling-feedback", "children"),
    Output("scaling-preview", "children"),
    Input("apply-scaling-btn", "n_clicks"),
    State("scaling-columns-dropdown", "value"),
    State("scaling-method", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True
)
def apply_scaling(n_clicks, selected_columns, method, cleaned_data, raw_data):
    if not selected_columns:
        raise PreventUpdate

    try:
        data_source = cleaned_data if cleaned_data else raw_data
        df = pd.read_json(io.StringIO(data_source), orient="split")
    except Exception as e:
        return dash.no_update, f"Error al leer los datos: {e}", html.Div()

    try:
        scaler = None
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return dash.no_update, "Método de escalado no válido.", html.Div()

        df[selected_columns] = scaler.fit_transform(df[selected_columns])
        msg = f"Se aplicó {method} a: {', '.join(selected_columns)}"

    except Exception as e:
        return dash.no_update, f"Error durante el escalado: {e}", html.Div()

    preview = dash_table.DataTable(
        data=df.head().to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=5,
        style_table={"overflowX": "auto"}
    )

    return df.to_json(date_format="iso", orient="split"), msg, preview

##Fase 5: Selección de características y transformación avanzada
#Callback Mostrar columnas del dataset
@app.callback(
    Output("column-dropper-dropdown", "options"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def update_column_dropdown(cleaned_data, raw_data):
    import io
    import pandas as pd

    if not cleaned_data and not raw_data:
        return []

    try:
        data_source = cleaned_data if cleaned_data else raw_data
        df = pd.read_json(io.StringIO(data_source), orient="split")
    except Exception:
        return []

    return [{"label": col, "value": col} for col in df.columns]

#Callback Mostrar columnas del target
@app.callback(
    Output("target-column-dropdown", "options"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def update_target_dropdown(cleaned_data, raw_data):
    if not cleaned_data and not raw_data:
        return []

    df = pd.read_json(io.StringIO(cleaned_data if cleaned_data else raw_data), orient="split")
    return [{"label": col, "value": col} for col in df.columns]

#Callback Mostrar parámetros según el método elegido
@app.callback(
    Output("feature-selection-params", "children"),
    Input("feature-selection-method", "value")
)
def show_feature_selection_params(method):
    children = []

    if method == "kbest":
        children.append(html.Div([
            html.Label("Número de mejores características a seleccionar (k):"),
            dcc.Input(id="kbest-k", type="number", min=1, step=1, value=5)
        ]))

    elif method == "pca":
        children.append(html.Div([
            html.Label("Número de componentes principales (k):"),
            dcc.Input(id="pca-k", type="number", min=1, step=1, value=2)
        ]))

    return children


#Callback Aplicar selección o transformación
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("feature-selection-feedback", "children" , allow_duplicate=True),
    Output("feature-selection-preview", "children"),
    Input("apply-feature-selection-btn", "n_clicks"),
    State("column-dropper-dropdown", "value"),
    State("feature-selection-method", "value"),
    State("kbest-k", "value"),
    State("pca-k", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    State("target-column-dropdown", "value"),
    prevent_initial_call=True
)
def apply_feature_selection(n_clicks, drop_cols, method, k_kbest, k_pca, target_col, cleaned_data, raw_data):

    try:
        data_source = cleaned_data if cleaned_data else raw_data
        df = pd.read_json(io.StringIO(data_source), orient="split")
    except Exception as e:
        return dash.no_update, f"Error al leer los datos: {e}", html.Div()

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    try:
        if method == "kbest":
            if k_kbest is None or not target_col or target_col not in df.columns:
                return "Debes ingresar un valor válido para k y seleccionar una variable objetivo.", dash.no_update, html.Div()
            
            if not target_col or target_col not in df.columns:
                return dash.no_update, "Debes seleccionar una variable objetivo válida para usar SelectKBest.", html.Div()

            X = df.drop(columns=[target_col])
            y = df[target_col]

            selector = SelectKBest(score_func=f_classif, k=min(k_kbest or 5, X.shape[1]))
            X_new = selector.fit_transform(X, y)

            selected_features = X.columns[selector.get_support()]
            df = pd.DataFrame(X_new, columns=selected_features)
            df[target_col] = y.values
            msg = f"SelectKBest aplicado. Variables seleccionadas: {', '.join(selected_features)}"

        elif method == "pca":
            if k_pca is None:
                return "Debes ingresar un valor válido para k en PCA.", dash.no_update, html.Div()
            X = df.select_dtypes(include=["number"])
            pca = PCA(n_components=min(k_pca, X.shape[1]))
            X_pca = pca.fit_transform(X)
            df = pd.DataFrame(X_pca, columns=[f"PCA_{i+1}" for i in range(X_pca.shape[1])])

        elif method == "variance":
            selector = VarianceThreshold(threshold=0.0)
            X = df.select_dtypes(include=["number"])
            X_new = selector.fit_transform(X)
            cols = X.columns[selector.get_support()]
            df = pd.DataFrame(X_new, columns=cols)

        msg = "Transformación aplicada correctamente."

    except Exception as e:
        return dash.no_update, f"Error durante la transformación: {e}", html.Div()

    preview = dash_table.DataTable(
        data=df.head().to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=5,
        style_table={"overflowX": "auto"}
    )

    return df.to_json(date_format="iso", orient="split"), msg, preview

##Fase 6: Balanceo de clases
#Callback Llenar dropdown y mostrar balance
@app.callback(
    Output("balancing-target-dropdown", "options"),
    Output("class-distribution-preview", "children"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def update_balancing_target(cleaned_data, raw_data):
    import io
    import pandas as pd
    import plotly.graph_objects as go

    if not cleaned_data and not raw_data:
        return [], html.P("No hay datos cargados.", style={"color": "red"})

    df = pd.read_json(io.StringIO(cleaned_data if cleaned_data else raw_data), orient="split")

    options = [{"label": col, "value": col} for col in df.columns]

    fig = go.Figure()

    # Mostrar gráfico solo si la variable objetivo tiene <= 20 clases (evitar problemas con valores únicos)
    previews = []
    for col in df.columns:
        if df[col].nunique() <= 20:
            fig.add_trace(go.Bar(x=df[col].value_counts().index.astype(str),
                                 y=df[col].value_counts().values,
                                 name=col))
            previews.append(col)

    if previews:
        fig.update_layout(title="Distribución de clases (solo columnas con ≤ 20 valores únicos)",
                          barmode="group")
        return options, dcc.Graph(figure=fig)

    return options, html.P("No se detectaron columnas con clases discretas para previsualizar.")

#Callback aplicar balanceo
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("balancing-feedback", "children"),
    Output("balancing-preview", "children"),
    Input("apply-balancing-btn", "n_clicks"),
    State("balancing-target-dropdown", "value"),
    State("balancing-method", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True
)
def apply_balancing(n_clicks, target_col, method, cleaned_data, raw_data):

    if not target_col:
        raise PreventUpdate

    try:
        df = pd.read_json(io.StringIO(cleaned_data if cleaned_data else raw_data), orient="split")
    except Exception as e:
        return dash.no_update, f"Error al leer los datos: {e}", html.Div()

    if target_col not in df.columns:
        return dash.no_update, "Columna objetivo no válida.", html.Div()

    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convertir categorías si hace falta
        if y.dtype == "object":
            y = y.astype("category").cat.codes

        if method == "smote":
            balancer = SMOTE()
        elif method == "undersample":
            balancer = RandomUnderSampler()
        else:
            return dash.no_update, "Método de balanceo no reconocido.", html.Div()

        X_bal, y_bal = balancer.fit_resample(X, y)

        df_bal = pd.DataFrame(X_bal, columns=X.columns)
        df_bal[target_col] = y_bal

        msg = f"Balanceo aplicado con {method.upper()}. Nuevas clases: {dict(pd.Series(y_bal).value_counts().to_dict())}"

    except Exception as e:
        return dash.no_update, f"Error durante el balanceo: {e}", html.Div()

    preview = dash_table.DataTable(
        data=df_bal.head().to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_bal.columns],
        page_size=5
    )

    return df_bal.to_json(date_format="iso", orient="split"), msg, preview


##Fase 7: Detección y tratamiento de outliers
#Callback para mostrar campo de Z-Score solo si se selecciona
@app.callback(
    Output("zscore-threshold-container", "style"),
    Input("outlier-detection-method", "value")
)
def toggle_zscore_input(method):
    return {"display": "block"} if method == "zscore" else {"display": "none"}

#Callback Mostrar columnas numéricas
@app.callback(
    Output("outlier-detection-columns", "options"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def load_outlier_columns(cleaned_data, raw_data):
    if not cleaned_data and not raw_data:
        return []
    df = pd.read_json(io.StringIO(cleaned_data if cleaned_data else raw_data), orient="split")
    num_cols = df.select_dtypes(include=["number"]).columns
    return [{"label": col, "value": col} for col in num_cols]


#Callback para detectar outliers (visualización + guardar índices)
@app.callback(
    Output("outlier-boxplot-preview", "children"),
    Output("outlier-detection-feedback", "children"),
    Output("outlier-index-store", "data"),
    Input("detect-outliers-btn", "n_clicks"),
    State("outlier-detection-columns", "value"),
    State("outlier-detection-method", "value"),
    State("zscore-threshold", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True
)
def detectar_outliers(n_clicks, columnas, metodo, z_th, cleaned_data, raw_data):
    if not columnas:
        raise dash.exceptions.PreventUpdate

    df = pd.read_json(io.StringIO(cleaned_data if cleaned_data else raw_data), orient="split")
    outlier_idx = set()

    for col in columnas:
        if metodo == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        elif metodo == "zscore":
            z = (df[col] - df[col].mean()) / df[col].std()
            mask = z.abs() > z_th
        else:
            continue

        outlier_idx.update(df[mask].index)

    fig = px.box(df, y=columnas, points="outliers", title="Visualización de Outliers")
    return dcc.Graph(figure=fig), f"Detectados {len(outlier_idx)} outliers en total.", list(outlier_idx)

#Callback para aplicar tratamiento a los outliers
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("outlier-treatment-feedback", "children"),
    Output("outlier-treatment-preview", "children"),
    Input("apply-outlier-treatment-btn", "n_clicks"),
    State("outlier-index-store", "data"),
    State("outlier-treatment-method", "value"),
    State("outlier-detection-columns", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True
)
def tratar_outliers(n_clicks, outlier_idx, metodo, columnas, cleaned_data, raw_data):
    if not outlier_idx or not columnas:
        raise dash.exceptions.PreventUpdate

    df = pd.read_json(io.StringIO(cleaned_data if cleaned_data else raw_data), orient="split")
    df_out = df.copy()
    idx = list(map(int, outlier_idx))

    if metodo == "remove":
        df_out.drop(index=idx, inplace=True)
        msg = f"Se eliminaron {len(idx)} filas con outliers."
    elif metodo == "mean":
        for col in columnas:
            df_out.loc[idx, col] = df[col].mean()
        msg = f"Reemplazados con la media en {len(columnas)} columnas."
    elif metodo == "median":
        for col in columnas:
            df_out.loc[idx, col] = df[col].median()
        msg = f"Reemplazados con la mediana en {len(columnas)} columnas."
    elif metodo == "mark":
        for col in columnas:
            df_out[f"{col}_is_outlier"] = df.index.isin(idx).astype(int)
        msg = f"Columnas de marcado añadidas para {len(columnas)} columnas."

    preview = dash_table.DataTable(
        data=df_out.head().to_dict("records"),
        columns=[{"name": i, "id": i} for i in df_out.columns],
        page_size=5,
        style_table={"overflowX": "auto"}
    )

    return df_out.to_json(date_format="iso", orient="split"), msg, preview


##Fase 8: Conversión de tipos y limpieza de columnas 
#Callback Llenar columnas con sus tipos
@app.callback(
    Output("type-columns-dropdown", "options"),
    Output("type-info-preview", "children"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def update_type_columns(cleaned_data, raw_data):
    if not cleaned_data and not raw_data:
        return [], html.P("No hay datos cargados.", style={"color": "red"})

    df = pd.read_json(io.StringIO(cleaned_data if cleaned_data else raw_data), orient="split")
    options = [{"label": f"{col} ({str(df[col].dtype)})", "value": col} for col in df.columns]

    dtypes_table = dash_table.DataTable(
        data=[{"Columna": col, "Tipo": str(df[col].dtype)} for col in df.columns],
        columns=[{"name": "Columna", "id": "Columna"}, {"name": "Tipo", "id": "Tipo"}],
        page_size=10,
        style_table={"overflowX": "auto"}
    )

    return options, dtypes_table


#Callback Aplicar conversión de tipo
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("type-conversion-feedback", "children"),
    Input("apply-type-conversion-btn", "n_clicks"),
    State("type-columns-dropdown", "value"),
    State("conversion-type", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    prevent_initial_call=True
)
def apply_type_conversion(n_clicks, selected_cols, target_type, cleaned_data, raw_data):
    if not selected_cols:
        raise dash.exceptions.PreventUpdate

    df = pd.read_json(io.StringIO(cleaned_data if cleaned_data else raw_data), orient="split")
    errors = []

    for col in selected_cols:
        try:
            if target_type == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type == "string":
                df[col] = df[col].astype(str)
            elif target_type == "category":
                df[col] = df[col].astype("category")
            elif target_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception as e:
            errors.append(f"{col}: {str(e)}")

    msg = f"Columnas convertidas a {target_type}: {', '.join(selected_cols)}"
    if errors:
        msg += f"Errores: {', '.join(errors)}"

    return df.to_json(date_format="iso", orient="split"), msg


#callback Aplicar índice temporal
# Opciones de columna temporal 
@app.callback(
    Output("timeindex-col", "options"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def _fill_timeindex_cols(cleaned_js, raw_js):
    import pandas as pd, io
    if not cleaned_js and not raw_js:
        return []
    df = get_latest_dataframe(cleaned_js, raw_js)
    pref = [c for c in df.columns if any(k in c.lower() for k in ["time","date","fecha","datetime","timestamp"])]
    cols = pref + [c for c in df.columns if c not in pref]
    return [{"label": c, "value": c} for c in cols]

# Aplicar índice temporal → ESCRIBE en cleaned-data-store
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("time-config-store", "data"),
    Output("timeindex-msg", "children"),
    Input("timeindex-apply", "n_clicks"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    State("timeindex-col", "value"),
    State("timeindex-tz", "value"),
    State("timeindex-use", "value"),
    prevent_initial_call=True
)
def _apply_timeindex(n, cleaned_js, raw_js, time_col, tz, use_flag):
    import pandas as pd, io, dash
    if not (cleaned_js or raw_js):
        return dash.no_update, dash.no_update, "No hay datos cargados."
    if not time_col:
        return dash.no_update, dash.no_update, "Selecciona una columna temporal."

    df = get_latest_dataframe(cleaned_js, raw_js)

    if time_col not in df.columns and df.index.name != time_col:
        return dash.no_update, dash.no_update, f"La columna '{time_col}' no existe."

    if df.index.name == time_col:
        df[time_col] = df.index

    parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    n_total = len(parsed); n_nat = int(parsed.isna().sum())
    if n_nat == n_total:
        return dash.no_update, dash.no_update, f"No se pudo convertir '{time_col}' a datetime."

    df[time_col] = parsed
    if tz and tz != "UTC":
        try:
            df[time_col] = df[time_col].dt.tz_convert(tz)
        except Exception as e:
            return dash.no_update, dash.no_update, f"Error al convertir a {tz}: {e}"

    df = df.sort_values(time_col)
    use_as_index = "on" in (use_flag or [])
    if use_as_index:
        df = df.set_index(time_col)

    msg = f"Índice temporal aplicado sobre '{time_col}' [{tz}]. Convertidos: {n_total - n_nat} / {n_total}"
    if n_nat: msg += f" (NaT: {n_nat})"
    msg += " | Usado como índice." if use_as_index else " | **No** se estableció como índice."

    time_cfg = {"time_col": time_col, "timezone": tz, "use_as_index": use_as_index}
    return df.to_json(date_format='iso', orient='split'), time_cfg, msg



# Opciones de groupby (SIN df-store)
@app.callback(
    Output("groupby-col", "options"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def _fill_groupby_options(cleaned_js, raw_js):
    import pandas as pd, io
    if not cleaned_js and not raw_js:
        return []
    df = get_latest_dataframe(cleaned_js, raw_js)
    return [{"label": c, "value": c} for c in df.columns]

# Generar agregados → lee cleaned/raw y guarda en agg-store (OK)
@app.callback(
    Output("agg-store", "data"),
    Output("agg-msg", "children"),
    Input("build-agg-btn", "n_clicks"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    State("time-config-store", "data"),
    State("resample-freq", "value"),
    State("agg-funcs", "value"),
    State("groupby-col", "value"),
    prevent_initial_call=True
)
def _build_aggregations(n_clicks, cleaned_js, raw_js, time_cfg, freq, funcs, gcol):
    import pandas as pd, io, dash
    if not (cleaned_js or raw_js):
        return dash.no_update, "No hay datos cargados."
    if not funcs:
        return dash.no_update, "Selecciona al menos una función de agregación."

    df = get_latest_dataframe(cleaned_js, raw_js)

    # Asegurar índice temporal
    if time_cfg and time_cfg.get("use_as_index") and isinstance(df.index, pd.DatetimeIndex):
        tcol_name = df.index.name or "time"
        df = df.sort_index()
    else:
        tcol = (time_cfg or {}).get("time_col")
        if not tcol or tcol not in df.columns:
            return dash.no_update, "Configura primero la columna temporal en Tab 2."
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
        if df[tcol].isna().all():
            return dash.no_update, f"No se pudo convertir '{tcol}' a datetime."
        df = df.sort_values(tcol).set_index(tcol)
        tcol_name = tcol

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return dash.no_update, "No hay columnas numéricas para agregar."

    try:
        if gcol and gcol in df.columns:
            agg_df = (
                df.groupby(gcol)[num_cols]
                  .resample(freq)
                  .agg(funcs)
                  .reset_index()
            )
        else:
            agg_df = df[num_cols].resample(freq).agg(funcs).reset_index()

        if isinstance(agg_df.columns, pd.MultiIndex):
            agg_df.columns = ["_".join([str(l) for l in tup if l]) for tup in agg_df.columns.values]

        if tcol_name not in agg_df.columns:
            time_candidates = [c for c in agg_df.columns if pd.api.types.is_datetime64_any_dtype(agg_df[c])]
            if time_candidates:
                agg_df = agg_df.rename(columns={time_candidates[0]: tcol_name})
        if tcol_name in agg_df.columns:
            agg_df = agg_df.sort_values(tcol_name)

    except Exception as e:
        return dash.no_update, f"Error al generar agregados: {e}"

    msg = f"Agregados generados: freq={freq}, funcs={funcs}."
    if gcol and gcol in df.columns:
        msg += f" (Agrupado por '{gcol}')"
    return agg_df.to_json(date_format="iso", orient="split"), msg


def _df_from_json(js):
    return pd.read_json(io.StringIO(js), orient="split")

# Mostrar/ocultar controles según operador (>=, >, <=, <, == vs between)
@app.callback(
    Output("threshold-block", "style"),
    Output("threshold-between-block", "style"),
    Input("label-operator", "value")
)
def toggle_threshold_inputs(op):
    if op == "between":
        return {"display":"none"}, {"display":"block"}
    return {"display":"block"}, {"display":"none"}
# Poblar columnas numéricas desde CLEANED (o RAW si aún no hay cleaned)
@app.callback(
    Output("label-col", "options"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def fill_label_cols(cleaned_js, raw_js):
    import pandas as pd
    if not cleaned_js and not raw_js:
        return []
    df = _df_from_json(cleaned_js) if cleaned_js else _df_from_json(raw_js)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return [{"label": c, "value": c} for c in num_cols]

# Crear etiqueta binaria y ESCRIBIR en cleaned-data-store 
@app.callback(
    Output("cleaned-data-store", "data", allow_duplicate=True),
    Output("label-msg", "children"),
    Input("build-label-btn", "n_clicks"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data"),
    State("label-col", "value"),
    State("label-operator", "value"),
    State("label-threshold", "value"),
    State("label-threshold-min", "value"),
    State("label-threshold-max", "value"),
    State("label-name", "value"),
    State("label-invert", "value"),
    prevent_initial_call=True
)
def build_label_into_cleaned(n, cleaned_js, raw_js, col, op, thr, thr_min, thr_max, lbl_name, invert_opt):
    import dash, numpy as np, pandas as pd

    if not (cleaned_js or raw_js):
        return dash.no_update, "No hay datos cargados."
    if not col:
        return dash.no_update, "Selecciona la columna sobre la que crear la etiqueta."

    # Base: CLEANED si existe; sino RAW
    df = _df_from_json(cleaned_js) if cleaned_js else _df_from_json(raw_js)
    if col not in df.columns:
        return dash.no_update, f"La columna '{col}' no existe en el dataset actual."

    s = pd.to_numeric(df[col], errors="coerce")

    # Construir máscara según operador
    if op in (">=", ">", "<=", "<", "=="):
        if thr is None:
            return dash.no_update, "Indica el umbral numérico."
        cond_txt = f"{col} {op} {thr}"
        if op == ">=": mask = s >= thr
        elif op == ">": mask = s > thr
        elif op == "<=": mask = s <= thr
        elif op == "<": mask = s < thr
        else:           mask = s == thr
    elif op == "between":
        if thr_min is None or thr_max is None:
            return dash.no_update, "Indica los dos umbrales (mín y máx)."
        lo, hi = (thr_min, thr_max) if thr_min <= thr_max else (thr_max, thr_min)
        mask = s.between(lo, hi, inclusive="both")
        cond_txt = f"{col} entre [{lo}, {hi}]"
    else:
        return dash.no_update, "Operador no reconocido."

    label_series = mask.astype(int)
    if "invert" in (invert_opt or []):
        label_series = (1 - label_series).astype(int)
        cond_txt = f"NO ({cond_txt})"

    # Nombre de la nueva columna (evitar sobreescritura accidental)
    name = (lbl_name or "etiqueta_binaria").strip()
    if name in df.columns:
        k, base = 2, name
        while name in df.columns:
            name = f"{base}_{k}"
            k += 1

    df[name] = label_series

    pos = int(df[name].sum()); total = int(df[name].count())
    msg = f"Etiqueta '{name}' creada con condición: {cond_txt}. Positivos: {pos}/{total} ({(pos/total*100):.1f}%)"

    # Promocionamos SIEMPRE a cleaned-data-store (como pediste)
    return df.to_json(date_format="iso", orient="split"), msg

# ----------------------------------------------------------
# ----------------------------------------------------------
## TAB 3: Entrenamiento y Evaluación de Modelos
# ----------------------------------------------------------
# ----------------------------------------------------------

def layout_tab3():
    return dbc.Container([
        html.H3("3. Entrenamiento y Evaluación de Modelos"),


        html.H5("Guía rápida de selección de modelos según el tipo de problema"),
        dash_table.DataTable(
            data=[
                {
                    "Tipo": "Clasificación",
                    "Modelos recomendados": "LogisticRegression, RandomForestClassifier, KNeighborsClassifier, SVC, GradientBoostingClassifier, XGBoost, LightGBM",
                    "¿Cuándo usar?": "Cuando la variable objetivo tiene clases (0/1, A/B, etc).",
                    "Ventajas / Desventajas": "Logistic: simple y rápida pero lineal. RF/XGBoost: alta precisión, toleran ruido. KNN: intuitivo pero lento con muchos datos. SVC: bueno para margen claro, no escalable."
                },
                {
                    "Tipo": "Regresión",
                    "Modelos recomendados": "LinearRegression, RandomForestRegressor, SVR, GradientBoostingRegressor, XGBoostRegressor",
                    "¿Cuándo usar?": "Cuando la variable objetivo es continua (precios, cantidades, etc).",
                    "Ventajas / Desventajas": "Linear: interpretabilidad alta, pero lineal. RF/XGBR: buena precisión y tolerancia a outliers. SVR: preciso pero lento y difícil de tunear."
                },
                {
                    "Tipo": "Clustering",
                    "Modelos recomendados": "KMeans, DBSCAN, AgglomerativeClustering",
                    "¿Cuándo usar?": "Cuando no hay variable objetivo y se busca agrupar observaciones similares.",
                    "Ventajas / Desventajas": "KMeans: rápido pero sensible a outliers. DBSCAN: detecta formas arbitrarias, no requiere K, pero sensible a escala. Agglomerative: jerárquico pero lento en grandes datasets."
                },
                {
                    "Tipo": "Series temporales",
                    "Modelos recomendados": "ARIMA, Prophet, LSTM, RandomForest (sliding window)",
                    "¿Cuándo usar?": "Cuando hay una variable dependiente del tiempo (predicción de demanda, precios, etc).",
                    "Ventajas / Desventajas": "ARIMA: buena para series estables. Prophet: fácil de usar, incorpora estacionalidad. LSTM: captura secuencias largas, requiere mucho dato. RF: útil con ventana móvil, no modela secuencia directa."
                }
            ],
            columns=[
                {"name": "Tipo", "id": "Tipo"},
                {"name": "Modelos recomendados", "id": "Modelos recomendados"},
                {"name": "¿Cuándo usar?", "id": "¿Cuándo usar?"},
                {"name": "Ventajas / Desventajas", "id": "Ventajas / Desventajas"},
            ],
            style_table={"overflowX": "auto", "marginBottom": "40px"},
            style_cell={'textAlign': 'left', 'font_size': '14px', 'whiteSpace': 'normal', 'height': 'auto'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
        ),

        html.H5("Seleccionar variable objetivo (target)"),
        dcc.Dropdown(
            id="target-selector-tab3",
            placeholder="Selecciona la columna objetivo...",
            style={"width": "60%"}
        ),

        html.Br(),
        html.Div(id="series-column-container"),

        html.Div(id="exog-variable-container", children=[
            html.Label("Selecciona variables exógenas (opcional):"),
            dcc.Dropdown(
                id="exog-variable-selector",
                options=[],  # se actualiza dinámicamente
                multi=True,
                placeholder="Selecciona columnas adicionales que pueden ayudar a predecir"
            )
        ], style={"marginTop": "20px", "width": "60%", "display": "none"}),

        html.Hr(),
        
        html.H5("1. Separación de datos"),
        dbc.Row([
            dbc.Col([
                html.Label("Tamaño del conjunto de prueba (test size sobre 1) :"),
                dcc.Input(id="test-size-input", type="number", value=0.2, min=0.05, max=0.5, step=0.05),
            ], width=3),
            dbc.Col([
                html.Label("¿Estratificar? (solo clasificación)"),
                dcc.Checklist(
                    options=[{'label': 'Sí', 'value': 'stratify'}],
                    id='stratify-checklist',
                    value=[]
                )
            ], width=3)
        ]),

        html.Hr(),

        html.H5("2. Tipo de problema"),
        dcc.RadioItems(
            id='problem-type-radio',
            options=[
                {'label': 'Clasificación', 'value': 'clasificacion'},
                {'label': 'Regresión', 'value': 'regresion'},
                {'label': 'Clustering', 'value': 'clustering'},
                {'label': 'Series temporales', 'value': 'series_temporales'}
            ],
            value='clasificacion',
            labelStyle={'display': 'inline-block', 'marginRight': '15px'}
        ),

        html.Hr(),

        html.Div(id="ts-model-category-container", children=[
            html.H5("Selecciona una categoría de modelo de series temporales"),
            dcc.RadioItems(
                id="ts-category-selector",
                options=[
                    {"label": "(a) Modelos estadísticos clásicos", "value": "estadisticos"},
                    {"label": "(b) ML adaptados", "value": "ml"},
                    {"label": "(c) Deep Learning", "value": "dl"},
                    {"label": "(d) Transformers", "value": "transformers"},
                    {"label": "(e) AutoML", "value": "automl"}
                ],
                labelStyle={"display": "block", "marginBottom": "5px"},
                style={"marginBottom": "20px"}
            )
        ], style={"display": "none"}),

        html.H5("3. Selección de modelo"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[],  # Se actualiza dinámicamente
            placeholder="Selecciona un modelo",
            style={'width': '60%'}
        ),

        html.Br(),

        html.Div(id='model-params-container'),  # Inputs dinámicos según modelo

        html.Hr(),

        html.H5("4. Entrenamiento"),
        html.Label("Selecciona una o más métricas:"),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[],  # se actualiza dinámicamente
            value=[],    # se define en callback
            multi=True,
            placeholder="Selecciona una o más métricas",
            style={'width': '60%'}
        ),

        html.Br(),
        dbc.Button("Entrenar modelo", id='train-model-button', color='primary'),
        html.Div(id='training-feedback', style={'marginTop': '10px', 'color': 'green'}),

        html.Hr(),

        html.H5("5. Resultados de entrenamiento"),
        html.Div(id='training-results'),




        html.Hr(),
        html.Div([
            dbc.Button("Descargar modelo entrenado", id="download-model-button", color="secondary"),
            dcc.Download(id="download-model")
        ]),
        html.Div(
            id="series-column-container",
            children=[
                html.Label("Selecciona la columna de la serie temporal:"),
                dcc.Dropdown(
                    id="series-column-selector",
                    options=[],  # se rellena dinámicamente
                    placeholder="Selecciona una columna numérica",
                    style={"width": "60%"}
                )
            ],
            style={"display": "none"}  # inicial oculto
        ),

    ], fluid=True)


# TAB 4 sub-tabs
#Callback para actualizar el dropdown con las columnas del dataset para seleccionar target
@app.callback(
    Output("target-selector-tab3", "options"),
    Input("cleaned-data-store", "data"),
    Input("raw-data-store", "data")
)
def actualizar_target_options(cleaned_data_json, raw_data_json):
    if not cleaned_data_json and not raw_data_json:
        return []
    df_json = cleaned_data_json if cleaned_data_json else raw_data_json
    df = pd.read_json(io.StringIO(df_json), orient="split")
    return [{"label": col, "value": col} for col in df.columns]

#Callback Mostrar/ocultar y rellenar el selector de variables exógenas
@app.callback(
    Output("exog-variable-container", "style"),
    Output("exog-variable-selector", "options"),
    Input("model-dropdown", "value"),
    State("cleaned-data-store", "data"),
    State("raw-data-store", "data")
)
def mostrar_selector_exogenas(modelo, cleaned_data_json, raw_data_json):
    modelos_que_usar_exog = ["arima", "prophet", "random_forest_reg", "xgb_reg", "lgbm_reg"]
    if modelo not in modelos_que_usar_exog:
        return {"display": "none"}, []

    data_json = cleaned_data_json if cleaned_data_json else raw_data_json
    if not data_json:
        return {"display": "block"}, []

    df = pd.read_json(io.StringIO(data_json), orient="split")
    opciones = [{"label": col, "value": col} for col in df.select_dtypes(include=['number', 'category', 'object']).columns]

    return {"display": "block"}, opciones


#Callback para actualizar modelos disponible
MODEL_OPTIONS = {
    "clasificacion": [
        {"label": "Regresión Logística", "value": "logistic_regression"},
        {"label": "Random Forest Classifier", "value": "random_forest_clf"},
        {"label": "K-Nearest Neighbors", "value": "knn"},
        {"label": "SVC (Máquinas de Vectores de Soporte)", "value": "svc"},
        {"label": "Gradient Boosting Classifier", "value": "gbc"},
        {"label": "XGBoost Classifier", "value": "xgb_clf"},
        {"label": "LightGBM Classifier", "value": "lgbm_clf"},
        {"label": "LSTM Classifier", "value": "lstm_classifier"}

    ],
    "regresion": [
        {"label": "Regresión Lineal", "value": "linear_regression"},
        {"label": "Random Forest Regressor", "value": "random_forest_reg"},
        {"label": "SVR (Support Vector Regression)", "value": "svr"},
        {"label": "Gradient Boosting Regressor", "value": "gbr"},
        {"label": "XGBoost Regressor", "value": "xgb_reg"},
        {"label": "LightGBM Regressor", "value": "lgbm_reg"}
    ],
    "clustering": [
        {"label": "KMeans", "value": "kmeans"},
        {"label": "DBSCAN", "value": "dbscan"},
        {"label": "Agglomerative Clustering", "value": "agglo"}
    ],
}

SERIES_TEMPORALES_MODELOS = {
    "estadisticos": [
        {"label": "ARIMA", "value": "arima"},
        {"label": "Prophet", "value": "prophet"}
    ],
    "ml": [
        {"label": "Random Forest (sliding window)", "value": "rf_sw"},
        {"label": "XGBoost (sliding window)", "value": "xgb_sw"}
    ],
    "dl": [
        {"label": "LSTM", "value": "lstm"},
        {"label": "TCN", "value": "tcn"}

    ],
    "transformers": [
        {"label": "Informer", "value": "informer"},
        {"label": "TFT", "value": "tft"}
    ],
    "automl": [
        {"label": "Nixtla Auto (StatsForecast)", "value": "nixtla"},
        {"label": "AutoTS", "value": "autots"}
    ]
}


@app.callback(
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    Input("problem-type-radio", "value"),
    Input("ts-category-selector", "value")
)
def actualizar_modelos_disponibles(tipo_problema, categoria_ts):
    if tipo_problema != "series_temporales":
        opciones = MODEL_OPTIONS.get(tipo_problema, [])
    else:
        opciones = SERIES_TEMPORALES_MODELOS.get(categoria_ts, [])
    
    valor_defecto = opciones[0]["value"] if opciones else None
    return opciones, valor_defecto



#callback para mostrar subcategorías si series temporales
@app.callback(
    Output("ts-model-category-container", "style"),
    Input("problem-type-radio", "value")
)
def mostrar_categoria_series_temporales(tipo):
    if tipo == "series_temporales":
        return {"display": "block"}
    return {"display": "none"}
#Callback para actualizar métricas disponibles (metric-dropdown)
METRIC_OPTIONS = {
    "clasificacion": [
        {"label": "Accuracy", "value": "accuracy"},
        {"label": "F1-score", "value": "f1"},
        {"label": "ROC AUC", "value": "roc_auc"},
        {"label": "Precisión", "value": "precision"},
        {"label": "Recall", "value": "recall"}
    ],
    "regresion": [
        {"label": "R²", "value": "r2"},
        {"label": "MAE (Error Absoluto Medio)", "value": "mae"},
        {"label": "MSE (Error Cuadrático Medio)", "value": "mse"},
        {"label": "RMSE (Raíz del Error Cuadrático)", "value": "rmse"}
    ],
    "clustering": [
        {"label": "Silhouette Score", "value": "silhouette"},
        {"label": "Davies-Bouldin Index", "value": "dbi"},
        {"label": "Calinski-Harabasz Index", "value": "chi"}
    ],
    "series_temporales": [
        {"label": "MAE", "value": "mae"},
        {"label": "MSE", "value": "mse"},
        {"label": "RMSE", "value": "rmse"},
        {"label": "MAPE", "value": "mape"}
    ]
}

@app.callback(
    Output("metric-dropdown", "options"),
    Output("metric-dropdown", "value"),
    Input("problem-type-radio", "value")
)
def actualizar_metricas(tipo_problema):
    if tipo_problema not in METRIC_OPTIONS:
        return [], []

    opciones = METRIC_OPTIONS[tipo_problema]
    # seleccionamos todas por defecto 
    valores_por_defecto = [opt["value"] for opt in opciones]
    return opciones, valores_por_defecto



MODEL_PARAMS = {
    "random_forest_clf": [
        {"id": "n_estimators", "label": "Número de árboles", "type": "number", "value": 100, "min": 10, "step": 10},
        {"id": "max_depth", "label": "Máxima profundidad", "type": "number", "value": 5, "min": 1, "step": 1},
    ],
    "random_forest_reg": [
        {"id": "n_estimators", "label": "Número de árboles", "type": "number", "value": 100, "min": 10, "step": 10},
        {"id": "max_depth", "label": "Máxima profundidad", "type": "number", "value": 5, "min": 1, "step": 1},
    ],
    "xgb_clf": [
        {"id": "n_estimators", "label": "n_estimators", "type": "number", "value": 100, "min": 10, "step": 10},
        {"id": "max_depth", "label": "max_depth", "type": "number", "value": 3, "min": 1, "step": 1},
    ],
    "xgb_reg": [
        {"id": "n_estimators", "label": "n_estimators", "type": "number", "value": 100, "min": 10, "step": 10},
        {"id": "max_depth", "label": "max_depth", "type": "number", "value": 3, "min": 1, "step": 1},
    ],
    "logistic_regression": [
        {"id": "C", "label": "Regularización (C)", "type": "number", "value": 1.0, "min": 0.01, "step": 0.1}
    ],
    "knn": [
        {"id": "n_neighbors", "label": "Vecinos (K)", "type": "number", "value": 5, "min": 1, "step": 1}
    ],
    "svc": [
        {"id": "C", "label": "Regularización (C)", "type": "number", "value": 1.0, "min": 0.01, "step": 0.1}
    ],
    "linear_regression": [],
    "lgbm_clf": [
        {"id": "n_estimators", "label": "n_estimators", "type": "number", "value": 100},
        {"id": "learning_rate", "label": "learning_rate", "type": "number", "value": 0.1, "min": 0.01, "step": 0.01}
    ],
    "lgbm_reg": [
        {"id": "n_estimators", "label": "n_estimators", "type": "number", "value": 100},
        {"id": "learning_rate", "label": "learning_rate", "type": "number", "value": 0.1, "min": 0.01, "step": 0.01}
    ],
    "gbr": [
        {"id": "n_estimators", "label": "n_estimators", "type": "number", "value": 100},
        {"id": "learning_rate", "label": "learning_rate", "type": "number", "value": 0.1}
    ],
    "gbc": [
        {"id": "n_estimators", "label": "n_estimators", "type": "number", "value": 100},
        {"id": "learning_rate", "label": "learning_rate", "type": "number", "value": 0.1}
    ],
    "svr": [
        {"id": "C", "label": "C (Regularización)", "type": "number", "value": 1.0, "step": 0.1},
        {"id": "epsilon", "label": "epsilon", "type": "number", "value": 0.1, "step": 0.1}
    ],
    "kmeans": [
    {"id": "n_clusters", "label": "Número de clusters", "type": "number", "value": 3, "min": 1}
    ],
    "dbscan": [
        {"id": "eps", "label": "Distancia máxima (eps)", "type": "number", "value": 0.5, "min": 0.1, "step": 0.1},
        {"id": "min_samples", "label": "Mínimo puntos por grupo", "type": "number", "value": 5, "min": 1}
    ],
    "agglo": [
        {"id": "n_clusters", "label": "Número de clusters", "type": "number", "value": 3, "min": 1}
    ],
    "arima": [
        {"id": "p", "label": "AR (p)", "type": "number", "value": 1},
        {"id": "d", "label": "I (d)", "type": "number", "value": 1},
        {"id": "q", "label": "MA (q)", "type": "number", "value": 0},
        {"id": "n_steps", "label": "Horizonte de predicción (n pasos)", "type": "number", "value": 10, "min": 1, "step": 1}
    ],
    "prophet": [
        {"id": "n_steps", "label": "Horizonte de predicción (n pasos)", "type": "number", "value": 10, "min": 1, "step": 1}
    ],  # Prophet se autoconfigura
     "rf_sw": [
        {"id": "n_lags", "label": "Nº de rezagos (sliding window)", "type": "number", "value": 5, "min": 1, "step": 1},
        {"id": "n_estimators", "label": "Número de árboles", "type": "number", "value": 100},
        {"id": "max_depth", "label": "Profundidad máxima", "type": "number", "value": 5}
    ],
    "xgb_sw": [
        {"id": "n_lags", "label": "Nº de rezagos (sliding window)", "type": "number", "value": 5, "min": 1, "step": 1},
        {"id": "n_estimators", "label": "Número de árboles", "type": "number", "value": 100},
        {"id": "max_depth", "label": "Profundidad máxima", "type": "number", "value": 5}
    ],
    "lstm": [
        {"id": "n_lags", "label": "Timesteps (ventana)", "type": "number", "value": 10, "min": 1, "step": 1},
        {"id": "units", "label": "Nº de neuronas (units)", "type": "number", "value": 50, "min": 1},
        {"id": "epochs", "label": "Épocas", "type": "number", "value": 20, "min": 1},
        {"id": "n_steps", "label": "Horizonte de predicción", "type": "number", "value": 10, "min": 1}
    ],
    "tcn": [
        {"id": "n_lags", "label": "Timesteps (ventana)", "type": "number", "value": 10, "min": 1},
        {"id": "n_filters", "label": "Nº de filtros", "type": "number", "value": 32, "min": 1},
        {"id": "n_steps", "label": "Horizonte de predicción", "type": "number", "value": 10, "min": 1},
        {"id": "epochs", "label": "Épocas", "type": "number", "value": 20, "min": 1}
    ],
    "nixtla": [
        {"id": "n_steps", "label": "Horizonte de predicción", "type": "number", "value": 10, "min": 1}
    ],
    "autots": [
        {"id": "n_steps", "label": "Horizonte de predicción", "type": "number", "value": 10, "min": 1},
        {"id": "exhaustividad", "label": "Nivel de exhaustividad", "type": "select", "value": "medio",
        "options": ["rápido", "medio", "exhaustivo"]}
    ],
    "lstm_classifier": [
    {"id": "window",     "label": "Longitud de ventana", "type": "number", "value": 14, "min": 2, "max": 200, "step": 1},
    {"id": "epochs",     "label": "Épocas",               "type": "number", "value": 20, "min": 1},
    {"id": "batch_size", "label": "Batch size",           "type": "number", "value": 16, "min": 1}
],

}

#Callback llenar series temporales
@app.callback(
    Output("series-column-container", "style"),
    Output("series-column-selector", "options"),
    Input("problem-type-radio", "value"),
    State('cleaned-data-store', 'data'),
    State('raw-data-store', 'data')
)
def mostrar_selector_serie(problema, cleaned_data_json, raw_data_json):
    if problema != "series_temporales":
        return {"display": "none"}, []

    data_json = cleaned_data_json if cleaned_data_json else raw_data_json
    if not data_json:
        return {"display": "block"}, []

    df = pd.read_json(io.StringIO(data_json), orient="split")
    opciones = [{"label": col, "value": col} for col in df.select_dtypes(include='number').columns]

    return {"display": "block"}, opciones


#Callback que genera los inputs en model-params-container
@app.callback(
    Output("model-params-container", "children"),
    Input("model-dropdown", "value")
)
def actualizar_parametros_modelo(modelo_seleccionado):
    if not modelo_seleccionado:
        return html.Div("Selecciona un modelo para ver sus hiperparámetros.")

    params = MODEL_PARAMS.get(modelo_seleccionado, [])
    inputs = []

    # Añadir advertencia para modelos DL
    if modelo_seleccionado in ["lstm", "tcn", "lstm_classifier"]:
        inputs.append(
            html.Div("Este modelo es más adecuado para series largas (varios cientos de observaciones).",
                     style={"color": "orange", "marginBottom": "10px"})
        )

    if not params:
        inputs.append(html.Div("Este modelo no requiere hiperparámetros configurables."))
    else:
        for p in params:
            if p["type"] == "select":
                input_component = dcc.Dropdown(
                    id={"type": "model-param", "param": p["id"]},
                    options=[{"label": o.capitalize(), "value": o} for o in p["options"]],
                    value=p["value"],
                    style={"width": "100%"}
                )
            else:
                input_component = dcc.Input(
                    id={"type": "model-param", "param": p["id"]},
                    type=p["type"],
                    value=p.get("value"),
                    min=p.get("min"),
                    max=p.get("max"),
                    step=p.get("step"),
                    debounce=True
                )

            inputs.append(
                html.Div([html.Label(p["label"]), input_component],
                         style={"marginBottom": "10px"})
            )

    return dbc.Col(inputs, width=6)



#Callback de entrenamiento
@app.callback(
    Output('trained-model-store', 'data'),
    Output('training-feedback', 'children'),
    Output('training-results', 'children'),
    Input('train-model-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('test-size-input', 'value'),
    State('stratify-checklist', 'value'),
    State('metric-dropdown', 'value'),
    State({'type': 'model-param', 'param': ALL}, 'value'),
    State({'type': 'model-param', 'param': ALL}, 'id'),
    State('cleaned-data-store', 'data'),
    State('raw-data-store', 'data'),
    State("target-selector-tab3", "value"),
    State("series-column-selector", "value"),
    State("exog-variable-selector", "value"),
    prevent_initial_call=True
)
def entrenar_modelo_interactivo(n_clicks, model_name, test_size, stratify_val, metric,
                                param_values, param_ids, cleaned_data_json, raw_data_json, target_col, series_col, exog_vars):

    import traceback
    from sklearn.model_selection import cross_val_score

    try:
        # Selección de datos
        data_json = cleaned_data_json if cleaned_data_json else raw_data_json
        if not data_json:
            return {}, "No hay datos disponibles.", html.Div()

        if not model_name:
            return {}, "Error: Debes seleccionar un modelo.", html.Div("No se ha seleccionado ningún modelo.")

        df = pd.read_json(io.StringIO(data_json), orient="split")

        # Identificación del tipo de problema
        problem_type = None
        if model_name in ["kmeans", "dbscan", "agglo"]:
            problem_type = "clustering"
        elif model_name in ["arima", "prophet"]:
            problem_type = "series_temporales"
        else:
            if "clf" in model_name or model_name in ["logistic_regression", "svc", "knn"]:
                problem_type = "clasificacion"
            elif "reg" in model_name or model_name in ["linear_regression", "svr"]:
                problem_type = "regresion"

        # -----------------------------------------------
        # Entrenamiento CLUSTERING
        if problem_type == "clustering":
            from sklearn.decomposition import PCA
            X = df.copy()
            params_dict = {param['param']: value for param, value in zip(param_ids, param_values)}

            model = {
                "kmeans": KMeans,
                "dbscan": DBSCAN,
                "agglo": AgglomerativeClustering
            }[model_name](**params_dict)

            model.fit(X)
            y_pred = model.labels_

            resultados = [html.H5("Clustering completado")]

            if metric:
                metric_val = None
                if metric[0] == "silhouette":
                    metric_val = silhouette_score(X, y_pred)
                elif metric[0] == "dbi":
                    metric_val = davies_bouldin_score(X, y_pred)
                elif metric[0] == "chi":
                    metric_val = calinski_harabasz_score(X, y_pred)
                if metric_val is not None:
                    resultados.append(html.P(f"{metric[0]} Score: {metric_val:.4f}"))

            # Visualización Clusters
            if X.shape[1] > 2:
                X_vis = PCA(n_components=2).fit_transform(X)
            else:
                X_vis = X.values

            df_vis = pd.DataFrame(X_vis, columns=["Componente 1", "Componente 2"])
            df_vis["Cluster"] = y_pred

            fig = px.scatter(df_vis, x="Componente 1", y="Componente 2", color=df_vis["Cluster"].astype(str))
            resultados.append(dcc.Graph(figure=fig))

            model_id = str(uuid.uuid4())
            MODELS_MEMORY[model_id] = model

            return (
                {"model_id": model_id, "model_name": model_name},
                "Modelo de clustering entrenado correctamente.",
                dbc.Card(dbc.CardBody(resultados))
            )

        # -----------------------------------------------
        # Entrenamiento SERIES TEMPORALES
        if problem_type == "series_temporales":
            if not series_col or series_col not in df.columns:
                return {}, "Debes seleccionar columna válida.", html.Div()

            serie = df[series_col].dropna()
            pasos = int(params_dict.get("n_steps", 10))
            params_dict = {param['param']: value for param, value in zip(param_ids, param_values)}

            if model_name == "arima":
                exog_df = df[exog_vars] if exog_vars else None
                model = ARIMA(serie, order=(params_dict.get("p", 1), params_dict.get("d", 1), params_dict.get("q", 0)), exog=exog_df).fit()
                pred_index = pd.RangeIndex(len(serie), len(serie) + pasos)
                forecast = model.forecast(steps=pasos, exog=exog_df.tail(pasos) if exog_vars else None)
                x_vals = pred_index
                y_vals = forecast
            elif model_name == "prophet":
                prophet_df = pd.DataFrame({"ds": df.index[:len(serie)], "y": serie})
                if exog_vars:
                    for var in exog_vars:
                        prophet_df[var] = df[var]
                        model.add_regressor(var)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=pasos)
                forecast_df = model.predict(future)
                x_vals = forecast_df["ds"].tail(pasos)
                y_vals = forecast_df["yhat"].tail(pasos)
            elif model_name in ["rf_sw", "xgb_sw"]:
                n_lags = params_dict.get("n_lags", 5)
                #validaciones tempranas antes de entrenar:
                if len(serie) < n_lags + pasos + 10:
                    return {}, "La serie temporal es demasiado corta para esta configuración de lags y pasos.", html.Div()
                if exog_vars:
                    for var in exog_vars:
                        if var not in df.columns:
                            return {}, f"Variable exógena '{var}' no encontrada en el DataFrame.", html.Div()

                X, y = [], []
                #Asegurar preprocesamiento de variables exógenas?
                if exog_vars:
                    for col in exog_vars:
                        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                            df = pd.get_dummies(df, columns=[col], drop_first=True)

                for i in range(n_lags, len(serie) - pasos):
                    lags = serie.iloc[i - n_lags:i].values
                    row = list(lags)
                    if exog_vars:
                        row += df[exog_vars].iloc[i].values.tolist()
                    X.append(row)
                    y.append(serie.iloc[i])
                
                X = np.array(X)
                y = np.array(y)

                model_cls = RandomForestRegressor if model_name == "rf_sw" else xgb.XGBRegressor
                model = model_cls(
                    n_estimators=params_dict.get("n_estimators", 100),
                    max_depth=params_dict.get("max_depth", 5)
                )
                model.fit(X, y)

                # Predicción n pasos adelante (simplificada)
                ultimos = list(serie.iloc[-n_lags:].values)
                if exog_vars:
                    ult_exog = df[exog_vars].iloc[-pasos:].values
                    preds = []
                    for step in range(pasos):
                        input_row = ultimos[-n_lags:]
                        input_row = list(input_row) + list(ult_exog[step])
                        pred = model.predict([input_row])[0]
                        preds.append(pred)
                        ultimos.append(pred)
                    y_vals = preds
                    x_vals = pd.RangeIndex(len(serie), len(serie) + pasos)
                else:
                    # Sin exógenas
                    y_vals = []
                    current = list(serie.iloc[-n_lags:].values)
                    for _ in range(pasos):
                        pred = model.predict([current[-n_lags:]])[0]
                        y_vals.append(pred)
                        current.append(pred)
                    x_vals = pd.RangeIndex(len(serie), len(serie) + pasos)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=serie, mode="lines", name="Serie original"))
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name="Predicción"))

                model_id = str(uuid.uuid4())
                MODELS_MEMORY[model_id] = model

                return (
                    {"model_id": model_id, "model_name": model_name},
                    "Modelo ML (ventana móvil) entrenado correctamente.",
                    dbc.Card(dbc.CardBody([html.H5("Predicción con ventana móvil"), dcc.Graph(figure=fig)]))
                )
            elif model_name == "lstm":
                n_lags = params_dict.get("n_lags", 10)
                n_steps = params_dict.get("n_steps", 10)
                units = params_dict.get("units", 50)
                epochs = params_dict.get("epochs", 20)

                if len(serie) < n_lags + n_steps + 20:
                    return {}, "La serie es demasiado corta para entrenar una LSTM con esta configuración.", html.Div()

                # Escalado (esencial para redes neuronales)
                scaler = MinMaxScaler()
                serie_scaled = scaler.fit_transform(serie.values.reshape(-1, 1))

                X, y = [], []
                for i in range(n_lags, len(serie_scaled) - n_steps):
                    X.append(serie_scaled[i - n_lags:i, 0])
                    y.append(serie_scaled[i:i + n_steps, 0])  # multi-step forecast

                X = np.array(X)
                y = np.array(y)
                X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

                # Red LSTM
                model = Sequential()
                model.add(LSTM(units, activation='tanh', input_shape=(n_lags, 1)))
                model.add(Dense(n_steps))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

                early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
                model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])

                # Predicción
                ultimos = serie_scaled[-n_lags:].reshape(1, n_lags, 1)
                pred_scaled = model.predict(ultimos)[0]
                pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                x_vals = pd.RangeIndex(len(serie), len(serie) + n_steps)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=serie, mode="lines", name="Serie original"))
                fig.add_trace(go.Scatter(x=x_vals, y=pred, mode="lines+markers", name="Predicción LSTM"))

                model_id = str(uuid.uuid4())
                MODELS_MEMORY[model_id] = model

                return (
                    {"model_id": model_id, "model_name": model_name},
                    "LSTM entrenada correctamente.",
                    dbc.Card(dbc.CardBody([
                        html.H5("Predicción con LSTM"),
                        dcc.Graph(figure=fig),
                        html.P("Nota: LSTM requiere series largas. Si el rendimiento es bajo, revisa el tamaño del dataset.")
                    ]))
                )
            elif model_name == "tcn":
                n_lags = params_dict.get("n_lags", 10)
                n_filters = params_dict.get("n_filters", 32)
                n_steps = params_dict.get("n_steps", 10)
                epochs = params_dict.get("epochs", 20)

                if len(serie) < n_lags + n_steps + 20:
                    return {}, "La serie es demasiado corta para entrenar un modelo TCN con esta configuración.", html.Div()
                if serie.isnull().any():
                    return {}, "La serie contiene valores nulos. Por favor, limpia los datos en Tab 2.", html.Div()

                scaler = MinMaxScaler()
                serie_scaled = scaler.fit_transform(serie.values.reshape(-1, 1))

                X, y = [], []
                for i in range(n_lags, len(serie_scaled) - n_steps):
                    X.append(serie_scaled[i - n_lags:i, 0])
                    y.append(serie_scaled[i:i + n_steps, 0])

                X = np.array(X).reshape(-1, n_lags, 1)
                y = np.array(y)

                model = Sequential()
                model.add(TCN(nb_filters=n_filters, kernel_size=2, dilations=[1, 2, 4], input_shape=(n_lags, 1)))
                model.add(Dense(n_steps))
                model.compile(optimizer='adam', loss='mse')

                early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
                model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])

                # Predicción
                ultimos = serie_scaled[-n_lags:].reshape(1, n_lags, 1)
                pred_scaled = model.predict(ultimos)[0]
                pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                x_vals = pd.RangeIndex(len(serie), len(serie) + n_steps)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=serie, mode="lines", name="Serie original"))
                fig.add_trace(go.Scatter(x=x_vals, y=pred, mode="lines+markers", name="Predicción TCN"))

                model_id = str(uuid.uuid4())
                MODELS_MEMORY[model_id] = model

                return (
                    {"model_id": model_id, "model_name": model_name},
                    "TCN entrenada correctamente.",
                    dbc.Card(dbc.CardBody([
                        html.H5("Predicción con TCN"),
                        dcc.Graph(figure=fig),
                        html.P("Nota: Las TCN capturan dependencias de largo plazo de forma eficiente.")
                    ]))
                )
            elif model_name == "nixtla":

                n_steps = params_dict.get("n_steps", 10)
                serie = df[[series_col]].copy()
                serie['ds'] = df.index
                serie['unique_id'] = "serie"

                if len(serie) < 30:
                    return {}, "La serie es demasiado corta para que Nixtla funcione bien.", html.Div()

                train_df = serie.iloc[:-n_steps]
                test_df = serie.iloc[-n_steps:]

                sf = StatsForecast(df=train_df[['unique_id', 'ds', series_col]], models=[AutoARIMA(), AutoETS(), AutoTheta()], freq="D")
                pred = sf.forecast(h=n_steps)

                y_vals = pred["AutoARIMA"].values if "AutoARIMA" in pred.columns else pred.iloc[:, 1].values
                x_vals = test_df["ds"].values

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=serie["ds"], y=serie[series_col], mode="lines", name="Serie original"))
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name="Predicción Nixtla"))

                model_id = str(uuid.uuid4())
                MODELS_MEMORY[model_id] = sf

                return (
                    {"model_id": model_id, "model_name": model_name},
                    "Nixtla entrenado correctamente.",
                    dbc.Card(dbc.CardBody([
                        html.H5("Predicción con Nixtla StatsForecast"),
                        dcc.Graph(figure=fig)
                    ]))
                )
            elif model_name == "autots":
                n_steps = params_dict.get("n_steps", 10)
                nivel = params_dict.get("exhaustividad", "medio")
                series = df[[series_col]].copy()

                if len(series) < 50:
                    return {}, "AutoTS requiere series de tamaño moderado (mínimo ~50 registros).", html.Div()

                forecast_length = n_steps
                model_config = {
                    "rápido": {"generations": 1, "max_generations": 2},
                    "medio": {"generations": 3, "max_generations": 5},
                    "exhaustivo": {"generations": 5, "max_generations": 10}
                }[nivel]

                model = AutoTS(
                    forecast_length=forecast_length,
                    frequency='infer',
                    prediction_interval=0.9,
                    model_list="superfast",
                    max_generations=model_config["max_generations"],
                    num_validations=2,
                    verbose=0
                )

                model = model.fit(series, date_col=None, value_col=series_col, id_col=None)
                prediction = model.predict()
                forecast_df = prediction.forecast

                y_vals = forecast_df[series_col].values
                x_vals = pd.RangeIndex(len(series), len(series) + n_steps)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=series[series_col], mode="lines", name="Serie original"))
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name="Predicción AutoTS"))

                model_id = str(uuid.uuid4())
                MODELS_MEMORY[model_id] = model

                return (
                    {"model_id": model_id, "model_name": model_name},
                    "AutoTS entrenado correctamente.",
                    dbc.Card(dbc.CardBody([
                        html.H5("Predicción con AutoTS"),
                        dcc.Graph(figure=fig)
                    ]))
                )
            

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=serie, mode="lines", name="Serie original"))
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name="Predicción"))
            fig.update_layout(title="Predicción de serie temporal")

            model_id = str(uuid.uuid4())
            MODELS_MEMORY[model_id] = model

            return (
                {"model_id": model_id, "model_name": model_name},
                "Serie temporal entrenada correctamente.",
                dbc.Card(dbc.CardBody([
                    html.H5("Predicción de series temporales"),
                    dcc.Graph(figure=fig)
                ]))
            )


        # -----------------------------------------------
        # Entrenamiento CLASIFICACIÓN o REGRESIÓN
        if not target_col or target_col not in df.columns:
            return {}, "Debes seleccionar una variable objetivo.", html.Div()

        X = df.drop(columns=[target_col])
        y = df[target_col]

        stratify = y if 'stratify' in stratify_val else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=42
        )
        # --- Branch específico: LSTM Classifier (binaria) ---
        if model_name == "lstm_classifier":
            # Hiperparámetros desde la UI
            params_dict = {param['param']: value for param, value in zip(param_ids, param_values)}
            window     = int(params_dict.get("window", 14))
            epochs     = int(params_dict.get("epochs", 20))
            batch_size = int(params_dict.get("batch_size", 16))

            # Construimos dataset secuencial 3D (N, window, n_features) usando TODAS las features salvo el target
            X_cols = [c for c in df.columns if c != target_col]
            # Nota: ya tienes helpers para esto:
            # _build_seq_dataset(df, X_cols, y_col, window)
            X_seq, y_seq = _build_seq_dataset(df, X_cols, target_col, window)

            # Entrenamos la LSTM binaria y calculamos métricas
            # Si prefieres usar el helper, lo tienes ya implementado (_train_lstm_binary)
            # Aquí lo haremos explícito para además guardar el modelo y mostrar CM:
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            import tensorflow as tf
            from tensorflow.keras import layers, models

            stratify_arr = y_seq if len(np.unique(y_seq)) == 2 else None
            Xtr, Xte, ytr, yte = train_test_split(X_seq, y_seq, test_size=test_size, random_state=42, stratify=stratify_arr)

            model = models.Sequential([
                layers.Input(shape=(X_seq.shape[1], X_seq.shape[2])),
                layers.LSTM(32),
                layers.Dense(1, activation="sigmoid")
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

            y_score = model.predict(Xte, verbose=0).ravel()
            y_pred  = (y_score >= 0.5).astype(int)

            # Métricas seleccionadas por el usuario
            resultados_texto = []
            for m in metric:
                try:
                    if m == "accuracy":  resultados_texto.append(f"Accuracy: {accuracy_score(yte, y_pred):.4f}")
                    if m == "precision": resultados_texto.append(f"Precisión: {precision_score(yte, y_pred, zero_division=0):.4f}")
                    if m == "recall":    resultados_texto.append(f"Recall: {recall_score(yte, y_pred, zero_division=0):.4f}")
                    if m == "f1":        resultados_texto.append(f"F1-score: {f1_score(yte, y_pred, zero_division=0):.4f}")
                    if m == "roc_auc":
                        try:
                            resultados_texto.append(f"ROC AUC: {roc_auc_score(yte, y_score):.4f}")
                        except Exception:
                            pass
                except Exception as e:
                    resultados_texto.append(f"Error calculando {m}: {str(e)}")

            # Visuales: matriz de confusión
            import plotly.express as px
            import plotly.graph_objects as go
            cm = confusion_matrix(yte, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')

            cards = [
                dbc.Card(dbc.CardBody([html.H5("Métricas"), html.P(" | ".join(resultados_texto))])),
                dbc.Card(dbc.CardBody([html.H5("Matriz de Confusión"), dcc.Graph(figure=fig_cm)])),
                dbc.Card(dbc.CardBody([html.P("Nota: La LSTM de clasificación requiere que los datos estén ordenados temporalmente y suficientes filas para formar ventanas.")]))
            ]

            # Guardamos el modelo en memoria como haces con el resto
            model_id = str(uuid.uuid4())
            MODELS_MEMORY[model_id] = model

            return (
                {"model_id": model_id, "model_name": model_name},
                "LSTM Classifier entrenada correctamente.",
                html.Div(cards)
            )

        # Modelo
        params_dict = {param['param']: value for param, value in zip(param_ids, param_values)}
        modelos = {
            'random_forest_clf': RandomForestClassifier,
            'random_forest_reg': RandomForestRegressor,
            'logistic_regression': LogisticRegression,
            'xgb_clf': xgb.XGBClassifier,
            'xgb_reg': xgb.XGBRegressor,
            'svc': SVC,
            'knn': KNeighborsClassifier,
            'linear_regression': LinearRegression,
            'svr': SVR,
            'lgbm_clf': lgb.LGBMClassifier,
            'lgbm_reg': lgb.LGBMRegressor,
            'gbc': GradientBoostingClassifier,
            'gbr': GradientBoostingRegressor
        }

        model = modelos[model_name](**params_dict)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        is_clf = "clf" in model_name or model_name in ["logistic_regression", "svc", "knn", "gbc", "lgbm_clf", "xgb_clf"]

        # Resultados métricas
        resultados_texto = []
        for m in metric:
            try:
                if m == "r2" and not is_clf:
                    resultados_texto.append(f"R²: {r2_score(y_test, y_pred):.4f}")
                elif m == "mae" and not is_clf:
                    resultados_texto.append(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
                elif m == "mse" and not is_clf:
                    resultados_texto.append(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
                elif m == "rmse" and not is_clf:
                    resultados_texto.append(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
                elif m == "accuracy" and is_clf:
                    resultados_texto.append(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                elif m == "f1" and is_clf:
                    resultados_texto.append(f"F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
                elif m == "precision" and is_clf:
                    resultados_texto.append(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
                elif m == "recall" and is_clf:
                    resultados_texto.append(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
                elif m == "roc_auc" and is_clf and hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_test)
                    if probas.shape[1] == 2:
                        auc = roc_auc_score(y_test, probas[:, 1])
                    else:
                        auc = roc_auc_score(y_test, probas, multi_class='ovr')
                    resultados_texto.append(f"ROC AUC: {auc:.4f}")
            except Exception as e:
                resultados_texto.append(f"Error calculando {m}: {str(e)}")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy' if is_clf else 'r2')
        cv_summary = f"Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"

        # Cards visuales
        cards = [
            dbc.Card(dbc.CardBody([html.H5("Métricas"), html.Ul([html.Li(t) for t in resultados_texto])])),
            dbc.Card(dbc.CardBody([html.H5("Cross-Validation"), html.P(cv_summary)]))
        ]

        if is_clf:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
            cards.append(dbc.Card(dbc.CardBody([html.H5("Matriz de Confusión"), dcc.Graph(figure=fig_cm)])))
        else:
            fig_disp = px.scatter(x=y_test, y=y_pred)
            cards.append(dbc.Card(dbc.CardBody([html.H5("Real vs Predicho"), dcc.Graph(figure=fig_disp)])))

        if hasattr(model, 'feature_importances_'):
            labels = [f"lag_{i}" for i in range(n_lags)]
            if exog_vars:
                exog_labels = []
                for col in df.columns:
                    if col.startswith(tuple(exog_vars)):
                        exog_labels.append(col)
                labels += exog_labels
            fig_feat = px.bar(x=labels, y=model.feature_importances_)
            resultados.append(dbc.Card(dbc.CardBody([html.H5("Importancia de variables"), dcc.Graph(figure=fig_feat)])))


        model_id = str(uuid.uuid4())
        MODELS_MEMORY[model_id] = model

        return (
            {"model_id": model_id, "model_name": model_name},
            "Modelo entrenado correctamente.",
            html.Div(cards)
        )

    except Exception as e:
        print(traceback.format_exc())
        return {}, "Error inesperado.", html.Div(str(e))



#callback para descargar el modelo
@app.callback(
    Output("download-model", "data"),
    Input("download-model-button", "n_clicks"),
    State("trained-model-store", "data"),
    prevent_initial_call=True
)
def descargar_modelo(n_clicks, modelo_guardado):
    if not modelo_guardado or "model_id" not in modelo_guardado:
        raise PreventUpdate

    model_id = modelo_guardado["model_id"]
    model = MODELS_MEMORY.get(model_id)

    if model is None:
        raise PreventUpdate

    # Serializar el modelo como pickle
    pickle_bytes = pickle.dumps(model)

    return dcc.send_bytes(
        pickle_bytes,
        filename=f"modelo_{modelo_guardado['model_name']}.pkl"
    )

"""El archivo puede ser cargado más tarde así:
import pickle
with open("modelo_nombre.pkl", "rb") as f:
    modelo = pickle.load(f)
"""
def _ensure_time_index(df, time_cfg):
    """Devuelve df ordenado temporalmente con índice datetime usando time-config-store."""
    import pandas as pd
    if time_cfg and time_cfg.get("use_as_index") and isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    tcol = (time_cfg or {}).get("time_col")
    if tcol and tcol in df.columns:
        tmp = df.copy()
        tmp[tcol] = pd.to_datetime(tmp[tcol], errors="coerce", utc=True)
        tmp = tmp.sort_values(tcol).set_index(tcol)
        return tmp
    return df

def _build_seq_dataset(df, X_cols, y_col, window):
    """Construye X (N, window, n_features) e y (N,) con ventanas pasadas; etiqueta en t."""
    import numpy as np
    df_ = df.dropna(subset=X_cols + [y_col]).copy()
    Xmat = df_[X_cols].astype(float).to_numpy()
    yvec = df_[y_col].astype(int).to_numpy()
    if len(Xmat) <= window:
        raise ValueError("No hay suficientes filas para formar secuencias con esa ventana.")
    X_seqs, y_out = [], []
    for t in range(window, len(Xmat)):
        X_seqs.append(Xmat[t-window:t, :])
        y_out.append(yvec[t])
    X = np.asarray(X_seqs, dtype="float32")
    y = np.asarray(y_out, dtype="int32")
    return X, y

def _train_lstm_binary(X, y, epochs=20, batch_size=16, random_state=42):
    """Entrena LSTM binario sobre X 3D e y binario. Devuelve métricas tipo sklearn."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as e:
        raise RuntimeError(f"TensorFlow/Keras no disponible: {e}")

    strat = y if len(np.unique(y)) == 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=strat)

    model = models.Sequential([
        layers.Input(shape=(X.shape[1], X.shape[2])),
        layers.LSTM(32),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, epochs=int(epochs), batch_size=int(batch_size), validation_split=0.2, verbose=0)

    y_score = model.predict(Xte, verbose=0).ravel()
    y_pred  = (y_score >= 0.5).astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(yte, y_pred)),
        "precision": float(precision_score(yte, y_pred, zero_division=0)),
        "recall":    float(recall_score(yte, y_pred, zero_division=0)),
        "f1":        float(f1_score(yte, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(yte, y_score))
    except Exception:
        metrics["roc_auc"] = None

    return metrics


# ----------------------------------------------------------
# ----------------------------------------------------------
## TAB 4: optimización
# ----------------------------------------------------------
# ----------------------------------------------------------


def layout_tab4():
    return dbc.Container([
        html.H2("4. Optimización de Modelos (Plan Futuro)", className="text-center my-4"),

        dbc.Card([
            dbc.CardBody([
                html.H4("Prioridad de Desarrollo", className="card-title"),
                html.P("""
                Durante la fase de diseño y desarrollo de este proyecto, se priorizó la implementación de un flujo completo
                que permitiera a usuarios sin conocimientos de programación realizar tareas de machine learning de manera
                guiada, intuitiva y efectiva.
                """),
            ])
        ], className="mb-4"),

        dbc.Card([
            dbc.CardBody([
                html.H4("Motivo de no Implementación", className="card-title"),
                html.P("""
                Si bien se contempla la inclusión de procesos de optimización automática de hiperparámetros como un paso
                natural para mejorar el rendimiento de los modelos, esta funcionalidad no ha sido incorporada en la versión actual
                por motivos de alcance y tiempo de desarrollo.
                """),
                html.P("""
                La optimización de hiperparámetros implica procesos de búsqueda exhaustiva (como Grid Search o Random Search),
                que requieren múltiples ciclos de entrenamiento, un manejo avanzado de errores y un considerable tiempo de ejecución.
                Integrarla de manera robusta habría incrementado significativamente la complejidad del sistema, afectando la estabilidad
                general del flujo actual, especialmente en una plataforma orientada a usuarios no técnicos.
                """),
            ])
        ], className="mb-4"),

        dbc.Card([
            dbc.CardBody([
                html.H4("Propuesta de Futuras Mejoras", className="card-title"),
                html.Ul([
                    html.Li("Implementar optimización automática mediante técnicas de búsqueda de hiperparámetros ajustadas al tipo de problema (clasificación, regresión, clustering o series temporales)."),
                    html.Li("Ofrecer opciones de optimización sencilla para parámetros críticos, con control sobre la profundidad de la búsqueda."),
                    html.Li("Permitir a los usuarios seleccionar entre un entrenamiento estándar o un entrenamiento optimizado según sus necesidades y recursos disponibles."),
                ]),
                html.P("""
                Esta evolución permitirá no solo mejorar el rendimiento de los modelos generados, sino también ofrecer una
                experiencia más completa a los usuarios, sin sacrificar la simplicidad ni la usabilidad del sistema actual.
                """),
            ])
        ], className="mb-4"),
    ], fluid=True)


# ----------------------------------------------------------
# CALLBACK PARA RENDERIZAR CADA TAB PRINCIPAL
# ----------------------------------------------------------
@app.callback(
    Output("main-tab-content", "children"),
    Input("main-tabs", "value")
)

def render_main_tab(tab):
    if tab == 'tab-1':
        return layout_tab1()
    elif tab == 'tab-2':
        return layout_tab2()
    elif tab == 'tab-3':
        return layout_tab3()
    elif tab == 'tab-4':
        return layout_tab4()

    return html.Div("Tab principal no encontrada.")


# ----------------------------------------------------------
# EJECUCIÓN
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)