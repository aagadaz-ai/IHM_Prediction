from __future__ import annotations
import numpy as np, pandas as pd, joblib
from pathlib import Path
from typing import List, Union, BinaryIO, Any
import warnings

# Chemins modèles / scalers 
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

SK_MODELS = {
    "SVR":          MODEL_DIR / "model_svr.save",
    "RandomForest": MODEL_DIR / "model_rf.save",
}
TF_MODELS = {
    "CNN":          MODEL_DIR / "cnn_model.keras",
}

#Scalers séparés pour chaque modèle
SCALER_X_SVR_PATH   = MODEL_DIR / "scaler_X_svr.save"
SCALER_Y_SVR_PATH   = MODEL_DIR / "scaler_y_svr.save"
SCALER_Y_CNN_PATH   = MODEL_DIR / "scaler_y_cnn.save"
SCALER_AUX_CNN_PATH = MODEL_DIR / "scaler_aux_cnn.save"

#  Utilitaires Signal 
def pad_270(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(270, dtype="float32")
    out[:min(len(arr), 270)] = arr[:270]
    return out

def adapt_len(batch: np.ndarray, target_len: int) -> np.ndarray:
    cur = batch.shape[1]
    if cur > target_len:
        return batch[:, :target_len, :]
    if cur < target_len:
        pad = np.zeros((batch.shape[0], target_len - cur, batch.shape[2]), dtype=batch.dtype)
        return np.concatenate([batch, pad], axis=1)
    return batch

def _signal_length(model) -> int:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[1])

# Extraction d’un onglet 
def _find_col(df: pd.DataFrame, keys: List[str]) -> str:
    for c in df.columns:
        if any(k.lower() in str(c).lower() for k in keys):
            return c
    raise KeyError(keys)

def _process_sheet(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = df[_find_col(df, ["temps", "time"])].astype(float).values * 1e9
    a = df[_find_col(df, ["amplitude"])   ].astype(float).values
    p = df[_find_col(df, ["permittivité"])].astype(float).values
    s = df[_find_col(df, ["écart", "std"])].astype(float).values

    mask = (t >= 3.0) & (t <= 4.0)
    sig_vec = pad_270(a[mask])
    feat_vec = np.array([
        np.mean(a[mask]) if mask.any() else 0.0,
        np.mean(p[mask]) if mask.any() else 0.0,
        np.mean(s[mask]) if mask.any() else 0.0
    ], dtype="float32")
    return sig_vec, feat_vec

def _iter_flat(db: pd.DataFrame):
    for (fic, ong), g in db.groupby(["fichier_source", "onglet_source"]):
        sig, feat = _process_sheet(g)
        yield f"{fic} | {ong}", sig, feat

# Fonction publique 
def predict_file(
    file_like: Union[BinaryIO, Any, None],
    model_name: str = "SVR",
    model=None,
    scaler=None,
    scaler_aux=None,
    df_override: pd.DataFrame = None  
) -> pd.DataFrame:

    #Choix entre fichier ou base consolidée
    if df_override is not None:
        df = df_override
    else:
        df = (pd.read_csv(file_like)
              if file_like.name.lower().endswith(".csv")
              else pd.read_excel(file_like))

    labels, sigs, feats = [], [], []

    if {"fichier_source", "onglet_source"}.issubset(df.columns):
        for lab, sig, feat in _iter_flat(df):
            labels.append(lab)
            sigs.append(sig)
            feats.append(feat)
    else:
        sig, feat = _process_sheet(df)
        labels.append(Path(file_like.name).stem)
        sigs.append(sig)
        feats.append(feat)

    X_sig = np.stack(sigs)
    X_aux = np.stack(feats)

    #  CNN 
    if model_name == "CNN":
        import tensorflow as tf
        model = model or tf.keras.models.load_model(TF_MODELS["CNN"])

        L = _signal_length(model)
        X_sig_adj = adapt_len(X_sig[..., np.newaxis], L)

        scaler_aux = scaler_aux or joblib.load(SCALER_AUX_CNN_PATH)
        X_aux_scaled = scaler_aux.transform(X_aux)

        y_scaled = model.predict([X_sig_adj, X_aux_scaled], verbose=0).ravel()

        if SCALER_Y_CNN_PATH.exists():
            scaler_y = joblib.load(SCALER_Y_CNN_PATH)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        else:
            warnings.warn("scaler_y_cnn.save introuvable : valeurs non inversées")
            y_pred = y_scaled

        

    # ─────────── SVR / RF ─────────────
    else:
        model = model or joblib.load(SK_MODELS[model_name])
        scaler = scaler or joblib.load(SCALER_X_SVR_PATH)

        X_cat = np.hstack([X_sig, X_aux])
        X_cat = X_cat[:, :scaler.n_features_in_]
        X_scaled = scaler.transform(X_cat)
        y_scaled = model.predict(X_scaled)

        if model_name == "SVR" and SCALER_Y_SVR_PATH.exists():
            scaler_y = joblib.load(SCALER_Y_SVR_PATH)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_scaled

    return pd.DataFrame({
        "onglet_source": labels,
        f"Épaisseur_prévue_mm_{model_name}": y_pred
    })

# ─────────── Exports ─────────────
__all__ = ["predict_file", "SK_MODELS", "TF_MODELS"]
