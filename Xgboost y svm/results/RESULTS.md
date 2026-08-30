# Resultados — XGBoost & SVM para predicción de humedad de suelo

Pipeline: `Xgboost y svm/soil_moisture_models.py`  (correr: `python "Xgboost y svm/soil_moisture_models.py"` desde la raíz del repo)
Datos: 6 estaciones TxSON (in-situ, horario → diario) + NASA SMAP.
Panel: 14,616 estación-día, 2015-01-01 → 2021-09-01.
Hold-out temporal: entrenamiento < 2020-01-01, prueba 2020-2021.
Objetivo: `SWC_5` (contenido de agua del suelo a 5 cm) a h = 1, 7 y 30 días.

## 1. Skill de pronóstico — RMSE (m³/m³)

| Modelo | h=1 | h=7 | h=30 |
|---|---|---|---|
| Persistencia (baseline) | 0.0176 | 0.0462 | 0.0654 |
| Climatología (baseline) | 0.0664 | 0.0659 | 0.0659 |
| **SVR (SVM, kernel RBF)** | 0.0210 | 0.0507 | 0.0709 |
| **XGBoost** | **0.0142** | **0.0430** | **0.0599** |

- XGBoost es el mejor en todos los horizontes. R² = 0.95 (h=1), 0.59 (h=7), 0.20 (h=30).
- A 1 día XGBoost supera a la persistencia en **19 %** de RMSE; a 7 días en **7 %**.
- A 30 días todos los modelos convergen al nivel de la climatología → límite de predictibilidad
  usando solo variables locales (mensaje honesto para la presentación).
- SVR queda **por debajo de la persistencia**: SVR sirve como comparación pero XGBoost
  es claramente el modelo a llevar a producción.

## 2. Validación SMAP vs in-situ

Global: **R = 0.66**, ubRMSE = 0.057 m³/m³, sesgo = +0.013 (SMAP ligeramente húmedo).
Por estación R = 0.76–0.87 (la correlación es buena; el sesgo varía por sitio: TXS06 +0.071, TXS05 −0.060).
→ Justifica una corrección de sesgo por estación antes de fusionar SMAP con el modelo.

## 3. Drivers de la humedad de suelo (XGBoost gain, h=7)

1. `SWC_5` actual (0.45) · 2. `SWC_5` día anterior (0.14) · 3. humedad a 10 cm ·
4. media móvil 30 días · 5. humedad a 50 cm · luego radiación (`Srad`), estacionalidad
(`month`, `doy`) y `SMAP_ff`.
→ La memoria del propio suelo domina; los forzamientos meteorológicos aportan en segundo orden.

## 4. Predicción en sitios SIN sensor (Leave-One-Station-Out, h=7)

Modelo entrenado con 5 estaciones, evaluado en la 6ª, usando **solo meteorología + SMAP + estación
del año** (sin histórico de sensores locales):

| Estación excluida | RMSE | R | RMSE climatología |
|---|---|---|---|
| TXS01 | 0.052 | 0.61 | 0.061 |
| TXS02 | 0.042 | 0.74 | 0.054 |
| TXS03 | 0.040 | 0.68 | 0.043 |
| TXS04 | 0.056 | 0.72 | 0.063 |
| TXS05 | 0.095 | 0.78 | 0.100 |
| TXS06 | 0.075 | 0.69 | 0.082 |

→ Incluso sin sensor local, el modelo le gana a la climatología en las 6 estaciones (R = 0.61–0.78).
Este es el resultado que respalda la aplicación en México (donde hay SMAP pero no red densa de sensores).

## Figuras (etiquetas en español)

- `fig_skill_vs_horizon.png` — error (RMSE) de cada modelo a 1, 7 y 30 días.
- `fig_obs_vs_pred_TXS01_h7.png` — humedad real vs predicha (TXS01, 7 días de anticipación).
- `fig_feature_importance.png` — qué variables usa XGBoost para predecir.
- `fig_smap_vs_insitu.png` — comparación satélite (SMAP) vs sensores en tierra.
- `fig_sitios_sin_sensores.png` — predicción en estaciones sin sensores locales (Leave-One-Station-Out).

## Limitaciones a declarar

- 6 estaciones, misma región (Hill Country, TX) → no valida climas/suelos mexicanos.
- SWC ya venía "filled" (imputado); no se conoce el método de relleno.
- Horizonte útil real ≈ 1–10 días; a 30 días es prácticamente climatología.
- Transferencia a México = demostración vía SMAP, pendiente de validar con datos locales.
