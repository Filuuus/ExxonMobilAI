"""
SARIMA para series de humedad de suelo SMAP (TxSON)
Proceso a las 6 estaciones TXS01-TXS06:
 Carga y filtra por estacion
 Remuestrea a frecuencia semanal (regulariza el tiempo)
 Revisa estacionariedad (ADF test) y estacionalidad (decomposicion)
 Divide en train/test (reserva el ultimo anio como test)
 Busca los mejores parametros SARIMA por AIC (grid search simple)
 Pronostica el periodo de test y compara contra valores reales
 Guarda metricas (RMSE, MAE, MAPE) y graficos por estacion

"""

import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")  # ignore warnings


# CONFIGURACION
RUTA_CSV = "../SMAP_Cleaned.csv"      
FRECUENCIA = "W"                    # "W" = semanal 
PERIODO_ESTACIONAL = 52             # 52 semanas ~ 1 año 
PROP_TEST_SEMANAS = 52              # reservamos el ultimo año (52 semanas) como test
CARPETA_SALIDA = "resultados_sarima"

import os
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# 1. CARGA Y FILTRADO POR ESTACION

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def filtrar_estacion(df, station_id):
    serie = df[df["Station_ID"] == station_id].copy()
    serie = serie.sort_values("Date").set_index("Date")
    return serie["SMAP_Moisture"]



# 2. REMUESTREO

def remuestrear(serie, frecuencia=FRECUENCIA):
    serie_remuestreada = serie.resample(frecuencia).mean()
    # Interpolacion lineal para huecos donde no hubo ninguna medicion esa semana
    serie_remuestreada = serie_remuestreada.interpolate(method="linear")
    # Por si quedan NaN al inicio/final (antes de la primera o despues de la ultima medicion real)
    serie_remuestreada = serie_remuestreada.bfill().ffill()
    return serie_remuestreada



# 3. ESTACIONARIEDAD Y ESTACIONALIDAD
def prueba_estacionariedad(serie, nombre_estacion):
    resultado = adfuller(serie.dropna())
    print(f"\n[{nombre_estacion}] Prueba Dickey-Fuller aumentada (ADF)")
    print(f"  Estadistico ADF : {resultado[0]:.4f}")
    print(f"  p-value         : {resultado[1]:.4f}")
    if resultado[1] < 0.05:
        print("  -> La serie es estacionaria (rechazamos H0). d=0 probablemente basta.")
    else:
        print("  -> La serie NO es estacionaria. Se necesita diferenciacion (d>=1).")
    return resultado[1]


def graficar_decomposicion(serie, nombre_estacion, periodo=PERIODO_ESTACIONAL):
    decomposicion = seasonal_decompose(serie, model="additive", period=periodo)
    fig = decomposicion.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle(f"Descomposicion estacional - {nombre_estacion}", y=1.02)
    fig.savefig(f"{CARPETA_SALIDA}/decomposicion_{nombre_estacion}.png", bbox_inches="tight")
    plt.close(fig)



# 4. TRAIN / TEST SPLIT

def dividir_train_test(serie, n_test=PROP_TEST_SEMANAS):
    train = serie.iloc[:-n_test]
    test = serie.iloc[-n_test:]
    return train, test



# 5. BUSQUEDA DE PARAMETROS SARIMA POR AIC (grid search simple)
def buscar_mejor_sarima(train, periodo_estacional=PERIODO_ESTACIONAL):
    """
    Prueba combinaciones pequenas de (p,d,q)(P,D,Q,s) y se queda con
    la de menor AIC. Grid reducido para que no tarde horas;
    se puede ampliar segun se necesite.
    """
    p = d = q = range(0, 2)       # 0 o 1
    P = D = Q = range(0, 2)       # 0 o 1
    s = periodo_estacional

    mejor_aic = np.inf
    mejor_orden = None
    mejor_orden_estacional = None
    mejor_modelo = None

    combinaciones = list(itertools.product(p, d, q, P, D, Q))
    print(f"\nProbando {len(combinaciones)} combinaciones de parametros...")

    for (pi, di, qi, Pi, Di, Qi) in combinaciones:
        try:
            modelo = SARIMAX(
                train,
                order=(pi, di, qi),
                seasonal_order=(Pi, Di, Qi, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            resultado = modelo.fit(disp=False)
            if resultado.aic < mejor_aic:
                mejor_aic = resultado.aic
                mejor_orden = (pi, di, qi)
                mejor_orden_estacional = (Pi, Di, Qi, s)
                mejor_modelo = resultado
        except Exception:
            continue

    print(f"Mejor orden: {mejor_orden} x {mejor_orden_estacional}  (AIC={mejor_aic:.2f})")
    return mejor_modelo, mejor_orden, mejor_orden_estacional


# 6. PRONOSTICO Y METRICAS DE VALIDACION

def evaluar_pronostico(modelo_ajustado, test):
    pronostico = modelo_ajustado.get_forecast(steps=len(test))
    valores_pronosticados = pronostico.predicted_mean
    intervalo_confianza = pronostico.conf_int()

    rmse = np.sqrt(mean_squared_error(test, valores_pronosticados))
    mae = mean_absolute_error(test, valores_pronosticados)
    mape = np.mean(np.abs((test.values - valores_pronosticados.values) / test.values)) * 100

    metricas = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE_%": round(mape, 2)}
    return valores_pronosticados, intervalo_confianza, metricas


def graficar_comparacion(train, test, pronostico, intervalo_confianza, nombre_estacion):
    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train.values, label="Entrenamiento", color="steelblue")
    plt.plot(test.index, test.values, label="Observado (real)", color="black")
    plt.plot(test.index, pronostico.values, label="Pronosticado (SARIMA)", color="crimson", linestyle="--")
    plt.fill_between(
        test.index,
        intervalo_confianza.iloc[:, 0],
        intervalo_confianza.iloc[:, 1],
        color="crimson", alpha=0.15, label="Intervalo de confianza 95%"
    )
    plt.title(f"SARIMA vs Observado - {nombre_estacion}")
    plt.xlabel("Fecha")
    plt.ylabel("Humedad de suelo (SMAP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDA}/comparacion_{nombre_estacion}.png")
    plt.close()



# 7. PIPELINE COMPLETO POR ESTACION

def procesar_estacion(df, station_id):
    print(f"\n{'='*60}\nProcesando estacion: {station_id}\n{'='*60}")

    serie = filtrar_estacion(df, station_id)
    serie_remuestreada = remuestrear(serie)

    prueba_estacionariedad(serie_remuestreada, station_id)
    graficar_decomposicion(serie_remuestreada, station_id)

    train, test = dividir_train_test(serie_remuestreada)

    modelo_ajustado, orden, orden_estacional = buscar_mejor_sarima(train)

    pronostico, intervalo_confianza, metricas = evaluar_pronostico(modelo_ajustado, test)
    graficar_comparacion(train, test, pronostico, intervalo_confianza, station_id)

    metricas["Station_ID"] = station_id
    metricas["orden_sarima"] = f"{orden}x{orden_estacional}"
    return metricas



# 8. EJECUCION PARA LAS 6 ESTACIONES

def main():
    df = cargar_datos(RUTA_CSV)
    estaciones = sorted(df["Station_ID"].unique())
    print(f"Estaciones encontradas: {estaciones}")

    resultados = []
    for station_id in estaciones:
        metricas = procesar_estacion(df, station_id)
        resultados.append(metricas)

    tabla_resumen = pd.DataFrame(resultados)
    tabla_resumen = tabla_resumen[["Station_ID", "orden_sarima", "RMSE", "MAE", "MAPE_%"]]
    print("\n\nRESUMEN FINAL - Metricas por estacion")
    print(tabla_resumen.to_string(index=False))

    tabla_resumen.to_csv(f"{CARPETA_SALIDA}/metricas_resumen.csv", index=False)
    print(f"\nResultados guardados en la carpeta '{CARPETA_SALIDA}/'")


if __name__ == "__main__":
    main()