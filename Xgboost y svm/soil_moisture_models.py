from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE = Path(__file__).resolve().parent          # .../ExxonMobilAI/Xgboost y svm
ROOT = HERE.parent                              # .../ExxonMobilAI
DATA = ROOT / "datasets"                        # input data (shared, not in this folder)
OUT = HERE / "results"                          # all outputs live next to this script
OUT.mkdir(exist_ok=True)

STATIONS = [1, 2, 3, 4, 5, 6]                       # TxSON station file numbers
SID = {i: f"TXS0{i}" for i in STATIONS}             # -> SMAP Station_ID
HORIZONS = [1, 7, 30]                               # forecast lead times (days)
TEST_START = "2020-01-01"                           # temporal hold-out
TARGET = "SWC_5"


# 1. Load + daily aggregation
def load_station_daily(i: int) -> pd.DataFrame:
    f = DATA / "TxSON-Station-Files" / f"Station{i}_filled_Data.csv"
    df = pd.read_csv(f)
    df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    # fill short gaps in the meteo channels that carry NaNs
    for c in ["RH", "Srad", "Wind speed", "Wind direction"]:
        df[c] = df[c].interpolate(limit=6).ffill().bfill()

    agg = {
        "SWC_5": "mean", "SWC_10": "mean", "SWC_20": "mean", "SWC_50": "mean",
        "T_5": "mean", "T_10": "mean", "T_20": "mean", "T_50": "mean",
        "Tair": "mean", "RH": "mean", "Srad": "mean",
        "Wind speed": "mean", "Ppt": "sum",
        "Flag": lambda s: float((s == 0).mean()),   # fraction of good-quality hours
    }
    d = df.resample("1D").agg(agg)
    d = d.rename(columns={"Flag": "qc_good_frac", "Wind speed": "wind"})
    d["Station_ID"] = SID[i]
    return d.reset_index().rename(columns={"datetime": "Date"})


def load_smap() -> pd.DataFrame:
    s = pd.read_csv(DATA / "SMAP_Cleaned.csv", parse_dates=["Date"])
    return s[["Station_ID", "Date", "SMAP_Moisture"]]


def build_panel() -> pd.DataFrame:
    frames = []
    smap = load_smap()
    for i in STATIONS:
        d = load_station_daily(i)
        sm = smap[smap.Station_ID == SID[i]].set_index("Date").sort_index()

        full_idx = pd.date_range(d.Date.min(), d.Date.max(), freq="1D")
        sm_daily = sm["SMAP_Moisture"].reindex(full_idx)
        smap_ff = sm_daily.ffill(limit=6)
        days_since = (~sm_daily.isna()).astype(int)
        days_since = days_since.groupby(days_since.cumsum()).cumcount()

        d = d.set_index("Date")
        d["SMAP_raw"] = sm_daily          # only on satellite-overpass days
        d["SMAP_ff"] = smap_ff            # carried forward for modelling
        d["days_since_smap"] = days_since
        frames.append(d.reset_index())
    return pd.concat(frames, ignore_index=True)



# 2. Feature engineering
def api(precip: pd.Series, k: float = 0.92) -> pd.Series:
    """Antecedent Precipitation Index (exponential decay)."""
    out = np.zeros(len(precip))
    acc = 0.0
    p = precip.fillna(0).to_numpy()
    for t in range(len(p)):
        acc = acc * k + p[t]
        out[t] = acc
    return pd.Series(out, index=precip.index)


def add_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    doy = g.Date.dt.dayofyear
    g["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    g["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    g["month"] = g.Date.dt.month

    for lag in [1, 2, 3, 7]:
        g[f"SWC_5_lag{lag}"] = g.SWC_5.shift(lag)
    g["SWC_10_lag1"] = g.SWC_10.shift(1)
    g["SWC_20_lag1"] = g.SWC_20.shift(1)
    g["SWC_50_lag1"] = g.SWC_50.shift(1)

    for w in [3, 7, 15, 30]:
        g[f"ppt_{w}d"] = g.Ppt.rolling(w, min_periods=1).sum()
    g["api"] = api(g.Ppt)
    g["swc5_roll7"] = g.SWC_5.rolling(7, min_periods=1).mean().shift(1)
    g["swc5_roll30"] = g.SWC_5.rolling(30, min_periods=1).mean().shift(1)
    g["srad_roll7"] = g.Srad.rolling(7, min_periods=1).mean()
    g["swc5_trend7"] = g.SWC_5.shift(1) - g.SWC_5.shift(8)
    return g


FEATURES_FULL = [
    "SWC_5", "SWC_10_lag1", "SWC_20_lag1", "SWC_50_lag1",
    "SWC_5_lag1", "SWC_5_lag2", "SWC_5_lag3", "SWC_5_lag7",
    "swc5_roll7", "swc5_roll30", "swc5_trend7",
    "T_5", "Tair", "RH", "Srad", "srad_roll7", "wind",
    "Ppt", "ppt_3d", "ppt_7d", "ppt_15d", "ppt_30d", "api",
    "SMAP_ff", "days_since_smap",
    "doy_sin", "doy_cos", "month",
]

# "no local sensor history" feature set -> can run where only weather + SMAP exist
FEATURES_NOHIST = [
    "T_5", "Tair", "RH", "Srad", "srad_roll7", "wind",
    "Ppt", "ppt_3d", "ppt_7d", "ppt_15d", "ppt_30d", "api",
    "SMAP_ff", "days_since_smap", "doy_sin", "doy_cos", "month",
]


# 3. Metrics
def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    bias = float(np.mean(yp - yt))
    ub = float(np.sqrt(np.mean(((yp - yp.mean()) - (yt - yt.mean())) ** 2)))
    r = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 else np.nan
    return {
        "n": int(m.sum()),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "ubRMSE": ub,
        "MAE": float(mean_absolute_error(yt, yp)),
        "bias": bias,
        "R": r,
        "R2": float(r2_score(yt, yp)),
    }



# 4. Forecast experiment
def make_target(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    parts = []
    for sid, g in panel.groupby("Station_ID"):
        g = add_features(g)
        g["y"] = g[TARGET].shift(-h)           # SWC_5 h days ahead
        g["y_persist"] = g[TARGET]             # persistence baseline
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def climatology_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    clim = train.groupby(train.Date.dt.dayofyear)["y"].mean()
    return test.Date.dt.dayofyear.map(clim).fillna(train.y.mean()).to_numpy()


def run_forecast(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    keep_preds = {}
    for h in HORIZONS:
        df = make_target(panel, h)
        df = df.dropna(subset=FEATURES_FULL + ["y"])
        tr = df[df.Date < TEST_START]
        te = df[df.Date >= TEST_START]

        Xtr, Xte = tr[FEATURES_NOHIST], te[FEATURES_NOHIST]
        ytr, yte = tr.y.to_numpy(), te.y.to_numpy()

        preds = {
            "persistence": te.y_persist.to_numpy(),
            "climatology": climatology_pred(tr, te),
        }

        # --- SVM (SVR, RBF kernel) ---
        scaler = StandardScaler().fit(Xtr)
        svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.005)
        svr.fit(scaler.transform(Xtr), ytr)
        preds["SVR"] = svr.predict(scaler.transform(Xte))

        # --- XGBoost ---
        model = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, random_state=42, n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        preds["XGBoost"] = model.predict(Xte)

        for name, p in preds.items():
            rows.append({"horizon_d": h, "model": name, **metrics(yte, p)})

        keep_preds[h] = {
            "dates": te.Date.to_numpy(),
            "station": te.Station_ID.to_numpy(),
            "y": yte, **preds,
        }
        if h == 7:
            imp = pd.Series(model.feature_importances_, index=FEATURES_FULL)
            imp.sort_values().to_frame("gain").to_csv(OUT / "feature_importance_h7.csv")
            keep_preds["imp7"] = imp.sort_values()

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics_forecast.csv", index=False)
    return res, keep_preds


# 5. SMAP vs in-situ validation
def run_smap_validation(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in panel.groupby("Station_ID"):
        v = g.dropna(subset=["SMAP_raw", "SWC_5"])
        rows.append({"Station_ID": sid, **metrics(v.SWC_5, v.SMAP_raw)})
    v = panel.dropna(subset=["SMAP_raw", "SWC_5"])
    rows.append({"Station_ID": "ALL", **metrics(v.SWC_5, v.SMAP_raw)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "metrics_smap_validation.csv", index=False)
    return out


# 6. Leave-One-Station-Out  (no in-situ history -> "sites without sensors")
def run_loso(panel: pd.DataFrame, h: int = 7) -> pd.DataFrame:
    df = make_target(panel, h).dropna(subset=FEATURES_NOHIST + ["y"])
    rows = []
    for sid in df.Station_ID.unique():
        tr = df[df.Station_ID != sid]
        te = df[df.Station_ID == sid]
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        model.fit(tr[FEATURES_NOHIST], tr.y)
        p = model.predict(te[FEATURES_NOHIST])
        rows.append({"held_out_station": sid, "horizon_d": h,
                     **metrics(te.y, p),
                     "RMSE_climatology": metrics(te.y, climatology_pred(tr, te))["RMSE"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "metrics_loso.csv", index=False)
    return out


# 7. Figures
# etiquetas en español para las gráficas
ES_MODEL = {
    "persistence": "Persistencia",
    "climatology": "Climatología",
    "SVR": "SVM (SVR)",
    "XGBoost": "XGBoost",
}
ES_FEAT = {
    "SWC_5": "Humedad 5 cm (hoy)",
    "SWC_5_lag1": "Humedad 5 cm (ayer)",
    "SWC_5_lag2": "Humedad 5 cm (hace 2 días)",
    "SWC_5_lag3": "Humedad 5 cm (hace 3 días)",
    "SWC_5_lag7": "Humedad 5 cm (hace 7 días)",
    "SWC_10_lag1": "Humedad 10 cm (ayer)",
    "SWC_20_lag1": "Humedad 20 cm (ayer)",
    "SWC_50_lag1": "Humedad 50 cm (ayer)",
    "swc5_roll7": "Humedad 5 cm (promedio 7 días)",
    "swc5_roll30": "Humedad 5 cm (promedio 30 días)",
    "swc5_trend7": "Tendencia humedad (últimos 7 días)",
    "T_5": "Temp. del suelo 5 cm",
    "Tair": "Temp. del aire",
    "RH": "Humedad relativa del aire",
    "Srad": "Radiación solar",
    "srad_roll7": "Radiación solar (promedio 7 días)",
    "wind": "Viento",
    "Ppt": "Lluvia (día)",
    "ppt_3d": "Lluvia acumulada 3 días",
    "ppt_7d": "Lluvia acumulada 7 días",
    "ppt_15d": "Lluvia acumulada 15 días",
    "ppt_30d": "Lluvia acumulada 30 días",
    "api": "Índice de lluvia antecedente (API)",
    "SMAP_ff": "Humedad satélite (SMAP)",
    "days_since_smap": "Días desde última medición SMAP",
    "doy_sin": "Estación del año (sen)",
    "doy_cos": "Estación del año (cos)",
    "month": "Mes",
}


def figures(panel, fc_res, preds, smap_val, loso):
    # habilidad del pronóstico vs horizonte
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, sub in fc_res.groupby("model"):
        sub = sub.sort_values("horizon_d")
        ax.plot(sub.horizon_d, sub.RMSE, marker="o", label=ES_MODEL.get(name, name))
    ax.set_xlabel("Horizonte de predicción (días a futuro)")
    ax.set_ylabel("Error del pronóstico – RMSE (m³/m³)\n(más abajo = mejor)")
    ax.set_title("Qué tan bien predice cada modelo la humedad a 5 cm\n(prueba en 2020–2021, años nunca vistos por el modelo)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(OUT / "fig_skill_vs_horizon.png", dpi=140); plt.close(fig)

    # observado vs predicho, una estación, h=7
    p = preds[7]
    mask = p["station"] == "TXS01"
    order = np.argsort(p["dates"][mask])
    d = p["dates"][mask][order]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(d, p["y"][mask][order], "k", lw=1.8, label="Humedad real (medida)")
    ax.plot(d, p["XGBoost"][mask][order], "tab:red", lw=1.3, label="Predicción XGBoost")
    ax.plot(d, p["SVR"][mask][order], "tab:blue", lw=1, alpha=.8, label="Predicción SVM")
    ax.plot(d, p["persistence"][mask][order], "tab:gray", lw=.8, alpha=.6,
            label="Suponer 'igual que hoy' (persistencia)")
    ax.set_title("Estación TXS01 – humedad del suelo a 5 cm, predicción con 7 días de anticipación")
    ax.set_ylabel("Humedad del suelo (m³/m³)")
    ax.set_xlabel("Fecha")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_obs_vs_pred_TXS01_h7.png", dpi=140); plt.close(fig)

    # importancia de variables
    if "imp7" in preds:
        imp = preds["imp7"].tail(15)
        fig, ax = plt.subplots(figsize=(8, 5.8))
        ax.barh([ES_FEAT.get(k, k) for k in imp.index], imp.values, color="tab:green")
        ax.set_xlabel("Importancia relativa (cuánto usa el modelo esa variable)")
        ax.set_title("¿Qué información usa XGBoost para predecir la humedad?\n(predicción a 7 días)")
        fig.tight_layout(); fig.savefig(OUT / "fig_feature_importance.png", dpi=140); plt.close(fig)

    # dispersión SMAP vs sensores en tierra
    v = panel.dropna(subset=["SMAP_raw", "SWC_5"])
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(v.SWC_5, v.SMAP_raw, s=6, alpha=.25)
    lim = [0, max(v.SWC_5.max(), v.SMAP_raw.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="Coincidencia perfecta")
    row = smap_val[smap_val.Station_ID == "ALL"].iloc[0]
    ax.set_title("¿Coincide el satélite con los sensores en tierra?\n"
                 f"Correlación R = {row.R:.2f}    ·    Error típico = {row.ubRMSE:.3f}\n"
                 f"Sesgo = {row.bias:+.3f} (el satélite marca un poco de más)",
                 fontsize=10)
    ax.set_xlabel("Humedad medida por sensores en tierra – TxSON (m³/m³)")
    ax.set_ylabel("Humedad estimada por satélite – SMAP (m³/m³)")
    ax.legend()
    ax.set_xlim(lim); ax.set_ylim(lim); fig.tight_layout()
    fig.savefig(OUT / "fig_smap_vs_insitu.png", dpi=140); plt.close(fig)

    # predicción en sitios sin sensores (Leave-One-Station-Out)
    lo = loso.copy()
    x = np.arange(len(lo)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - w / 2, lo.RMSE_climatology, w, label="Promedio histórico (referencia simple)",
           color="tab:gray")
    ax.bar(x + w / 2, lo.RMSE, w, label="Modelo XGBoost sin sensores locales",
           color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(lo.held_out_station)
    ax.set_ylabel("Error del pronóstico – RMSE (m³/m³)\n(más abajo = mejor)")
    ax.set_xlabel("Estación 'oculta' (el modelo nunca vio sus sensores)")
    ax.set_title("¿Se puede predecir donde NO hay sensores?\n"
                 "Modelo entrenado en 5 estaciones y probado en la 6ª (predicción a 7 días)")
    ax.legend(); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(OUT / "fig_sitios_sin_sensores.png", dpi=140); plt.close(fig)


def main():
    print("Building daily panel ...")
    panel = build_panel()
    panel.to_csv(OUT / "daily_panel.csv", index=False)
    print(f"  {len(panel):,} station-days | {panel.Date.min().date()} -> {panel.Date.max().date()}")

    print("\nSMAP vs in-situ validation ...")
    smap_val = run_smap_validation(panel)
    print(smap_val.to_string(index=False))

    print("\nForecast experiment (persistence / climatology / SVR / XGBoost) ...")
    fc_res, preds = run_forecast(panel)
    print(fc_res.pivot_table(index="model", columns="horizon_d",
                             values="RMSE").round(4).to_string())

    print("\nLeave-One-Station-Out (no in-situ history, h=7) ...")
    loso = run_loso(panel, h=7)
    print(loso[["held_out_station", "RMSE", "ubRMSE", "R", "RMSE_climatology"]]
          .round(4).to_string(index=False))

    figures(panel, fc_res, preds, smap_val, loso)
    print(f"\nDone. Tables + figures in: {OUT}")


if __name__ == "__main__":
    main()
