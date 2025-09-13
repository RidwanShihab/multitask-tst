import os, warnings, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.set_printoptions(edgeitems=3, linewidth=120)

MODEL_DIR        = "/mnt/data/deeptcn_MTL_cv_models"
OUTDIR           = "/mnt/data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

SCATTER_KIND = "hexbin"

FEATURES = [
    'EDA',
    'Left_Openness','Right_Openness',
    'Left_Diameter','Right_Diameter',
    'Left_PupilSensor_X','Left_PupilSensor_Y',
    'Right_PupilSensor_X','Right_PupilSensor_Y',
    'Left_GazeDir_X','Left_GazeDir_Y','Left_GazeDir_Z',
    'Right_GazeDir_X','Right_GazeDir_Y','Right_GazeDir_Z'
]
ID_COL    = "participant_id"
COND_COL  = "condition"
TIME_COL  = "Timestamp"

Y1_COL = "fms"
Y2_COL = "blur_effectiveness"
Y3_COL = "blur_label"

DURATION_SEC = 3
FT_HZ  = 5
WINDOW_FT  = DURATION_SEC * FT_HZ
STRIDE_FT  = max(1, WINDOW_FT  // 4)

FILTERS=64; KERNEL=6; DILATIONS=(1,2,4,8,16,32); NB_STACKS=3
DROPOUT=0.2; EMBED_DIM=128

K_FOLDS=10

EPOCHS = 90
BATCH  = 64

VAL_SIZE=0.15
SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED)

def eval_regression_with_plots(y_true, y_pred, title_prefix, outdir=OUTDIR, scatter_kind="scatter"):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    pearson = np.corrcoef(y_true, y_pred)[0,1] if (np.std(y_true)>0 and np.std(y_pred)>0) else np.nan
    print(f"{title_prefix}: MAE={mae:.4f} | RMSE={rmse:.4f} | R^2={r2:.4f} | r={pearson:.4f}")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    if scatter_kind == "hexbin":
        hb = ax.hexbin(y_true, y_pred, gridsize=40, bins='log')
        cb = fig.colorbar(hb, ax=ax); cb.set_label('log(count)')
    else:
        ax.scatter(y_true, y_pred, s=8, alpha=0.25)
    ax.plot([mn, mx], [mn, mx], lw=1)
    ax.set_title(f"{title_prefix} — y vs ŷ")
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    plt.tight_layout()
    p_scatter = os.path.join(outdir, f"{title_prefix}_scatter.png")
    plt.savefig(p_scatter, dpi=150); plt.show()

    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.hist(resid, bins=40)
    ax.set_title(f"{title_prefix} — Residuals (ŷ - y)")
    ax.set_xlabel("Residual"); ax.set_ylabel("Count")
    plt.tight_layout()
    p_hist = os.path.join(outdir, f"{title_prefix}_residuals.png")
    plt.savefig(p_hist, dpi=150); plt.show()

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": pearson,
            "scatter_path": p_scatter, "residuals_path": p_hist}

def plot_cm(cm, labels, title, save_path=None):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()

def drop_quasi_constant(df, cols, thr=1e-6):
    keep = []
    for c in cols:
        v = df[c].var(skipna=True)
        if pd.isna(v) or v < thr:
            continue
        keep.append(c)
    return keep

def zclip_fit(X):
    mu = X.mean(axis=(0,1), keepdims=True)
    sd = X.std(axis=(0,1), keepdims=True) + 1e-8
    def f(Z):
        Z = (Z - mu)/sd
        return np.clip(Z, -3.0, 3.0)
    return f

def minmax_fit(X):
    mn = X.min(axis=(0,1), keepdims=True)
    mx = X.max(axis=(0,1), keepdims=True)
    rng = np.maximum(mx - mn, 1e-12)
    return lambda Z: (Z - mn)/rng

def windowize_with_meta(df, features, labels, window, stride,
                        id_col=ID_COL, cond_col=COND_COL, time_col=None,
                        block_len=None):
    if cond_col not in df.columns:
        df = df.copy(); df[cond_col] = "default"

    sort_cols = [c for c in [id_col, time_col] if c in df.columns and c is not None]
    dd = df.sort_values(sort_cols) if sort_cols else df.copy()

    gkeys = [c for c in [id_col, cond_col] if c in dd.columns]
    iterator = dd.groupby(gkeys, sort=False) if gkeys else [(('none','none'), dd)]

    X, Ys = [], [[] for _ in labels]
    meta_rows = []
    for key, g in iterator:
        pid, cond = (key if isinstance(key, tuple) and len(key)==2 else (None, None))
        arr = g[features].to_numpy(dtype=np.float32)

        labs = []
        for l in labels:
            if l is None or l not in g.columns:
                labs.append(np.full(len(g), np.nan, dtype=float))
            else:
                labs.append(pd.to_numeric(g[l], errors="coerce").to_numpy(dtype=float))

        n = len(g)
        if n < window: continue

        series_id = f"{pid}|{cond}" if (pid is not None or cond is not None) else "none|none"
        block_len_eff = block_len if (block_len and block_len>0) else window

        for s in range(0, n - window + 1, stride):
            e = s + window
            X.append(arr[s:e])
            for i, lab in enumerate(labs):
                Ys[i].append(lab[e-1])
            meta_rows.append({
                "series_id": series_id,
                "block_id": f"{series_id}#blk{int(s // block_len_eff)}",
                "start_idx": int(s),
                "end_idx": int(e),
                "participant_id": pid,
                "condition": cond,
            })

    X = np.stack(X, axis=0) if len(X) else np.empty((0, window, len(features)), dtype=np.float32)
    Ys = [np.asarray(y, dtype=np.float32) for y in Ys]
    meta_df = pd.DataFrame(meta_rows)
    return X, Ys, meta_df

def drop_nan_windows(X, *ys, meta=None):
    m = np.ones(len(X), bool)
    for y in ys:
        m &= ~np.isnan(y)
    out = [X[m]]
    for y in ys:
        out.append(y[m])
    if meta is not None:
        meta = meta.loc[m].reset_index(drop=True)
        out.append(meta)
    return tuple(out)

def _fallback_tcn_stack(x, nb_filters, kernel_size, dilations, nb_stacks, dropout):
    def TCNBlock(x, filters, k, d, drop):
        skip = x
        x = layers.Conv1D(filters, k, padding="causal", dilation_rate=d)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.SpatialDropout1D(drop)(x)
        x = layers.Conv1D(filters, k, padding="causal", dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        if skip.shape[-1] != filters:
            skip = layers.Conv1D(filters, 1, padding="same")(skip)
        x = layers.Add()([x, skip]); x = layers.ReLU()(x)
        return x
    for _ in range(nb_stacks):
        for d in dilations:
            x = TCNBlock(x, nb_filters, kernel_size, d, dropout)
    return x

def build_backbone(input_shape,
                   nb_filters=FILTERS,
                   kernel_size=KERNEL,
                   dilations=DILATIONS,
                   nb_stacks=NB_STACKS,
                   dropout_rate=DROPOUT,
                   embed_dim=EMBED_DIM):
    inp = layers.Input(shape=input_shape, name="input")
    try:
        from tcn import TCN
        x = TCN(
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=list(dilations),
            nb_stacks=NB_STACKS,
            use_skip_connections=True,
            use_batch_norm=True,
            dropout_rate=dropout_rate,
            return_sequences=False,
            name="backbone_tcn",
        )(inp)
    except Exception:
        x = _fallback_tcn_stack(inp, nb_filters, kernel_size, list(dilations), nb_stacks, dropout_rate)
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(EMBED_DIM, activation="relu", name="embed")(x)
    x = layers.Dropout(0.2)(x)
    return keras.Model(inp, x, name="DeepTCN")

def build_multitask_model(backbone):
    x  = backbone.output
    y1 = layers.Dense(1, activation="linear", name="y1")(x)
    y2 = layers.Dense(2, activation="softmax", name="y2")(x)
    y3 = layers.Dense(1, activation="linear", name="y3")(x)
    return keras.Model(backbone.input, [y1, y2, y3], name="MTL_y1reg_y2bin_y3reg")

ft_df = fdf.copy()

FEATURES = drop_quasi_constant(ft_df, FEATURES, thr=1e-6)
print("Using FEATURES:", FEATURES)

if TIME_COL in ft_df.columns:
    ft_df[TIME_COL] = pd.to_datetime(ft_df[TIME_COL], errors="coerce")

X_all, [y1_all, y2_all, y3_all], meta = windowize_with_meta(
    ft_df, FEATURES, [Y1_COL, Y2_COL, Y3_COL],
    window=WINDOW_FT, stride=STRIDE_FT,
    id_col=ID_COL, cond_col=COND_COL, time_col=(TIME_COL if TIME_COL in ft_df.columns else None),
    block_len=WINDOW_FT
)
X_all, y1_all, y2_all, y3_all, meta = drop_nan_windows(X_all, y1_all, y2_all, y3_all, meta=meta)

y1_all = np.clip(y1_all, 0.0, 10.0)
y3_all = np.clip(y3_all, 0.0, 10.0)
y2_all = y2_all.astype(int)

y3_scale   = 10.0
y3_all_raw = y3_all.copy()
y3_all     = y3_all / y3_scale

gkf = GroupKFold(n_splits=K_FOLDS)
groups = meta["block_id"].values
fold_metrics = {
    "y1_MAE": [], "y1_RMSE": [], "y1_R2": [], "y1_r": [],
    "y3_MAE": [], "y3_RMSE": [], "y3_R2": [], "y3_r": [],
    "y2_ACC": []
}

def class_weights_binary(y):
    counts = np.array([(y==0).sum(), (y==1).sum()], dtype=float)
    counts[counts==0] = 1.0
    total = counts.sum()
    w = total / (2.0 * counts)
    return {0: float(w[0]), 1: float(w[1])}

def summarize(name, vals):
    vals = np.array(vals, dtype=float)
    return f"{name}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}  (per-fold: {', '.join(f'{v:.4f}' for v in vals)})"

fold_id = 0
for tr_index, te_index in gkf.split(X_all, y1_all, groups=groups):
    fold_id += 1
    print(f"\n===================== Fold {fold_id}/{K_FOLDS} =====================")
    tr_groups = groups[tr_index]
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED+fold_id)
    tr_sub_idx, va_sub_idx = next(gss_inner.split(tr_index, groups=tr_groups))
    tr_idx = tr_index[tr_sub_idx]
    va_idx = tr_index[va_sub_idx]
    te_idx = te_index

    X_tr, X_va, X_te = X_all[tr_idx], X_all[va_idx], X_all[te_idx]
    y1_tr, y1_va, y1_te = y1_all[tr_idx], y1_all[va_idx], y1_all[te_idx]
    y2_tr, y2_va, y2_te = y2_all[tr_idx], y2_all[va_idx], y2_all[te_idx]
    y3_tr, y3_va, y3_te = y3_all[tr_idx], y3_all[va_idx], y3_all[te_idx]
    y3_te_raw = y3_all_raw[te_idx]

    y1_tr = np.clip(y1_tr, 0.0, 10.0); y1_va = np.clip(y1_va, 0.0, 10.0); y1_te = np.clip(y1_te, 0.0, 10.0)
    y3_tr = np.clip(y3_tr, 0.0, 1.0);  y3_va = np.clip(y3_va, 0.0, 1.0);  y3_te = np.clip(y3_te, 0.0, 1.0)

    zf = zclip_fit(X_tr); X_tr = zf(X_tr); X_va = zf(X_va); X_te = zf(X_te)
    mm = minmax_fit(X_tr);  X_tr = mm(X_tr); X_va = mm(X_va); X_te = mm(X_te)

    keras.backend.clear_session()
    gc.collect()
    input_shape = (X_tr.shape[1], X_tr.shape[2])
    bb = build_backbone(input_shape)
    for L in bb.layers: L.trainable = True
    mt = build_multitask_model(bb)

    mt.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={'y1': keras.losses.Huber(delta=1.0),
              'y2': 'sparse_categorical_crossentropy',
              'y3': keras.losses.Huber(delta=1.0)},
        loss_weights={'y1': 1.0, 'y2': 1.0, 'y3': 1.0},
        metrics={'y1': [keras.metrics.MeanAbsoluteError(name="MAE")],
                 'y2': ['sparse_categorical_accuracy'],
                 'y3': [keras.metrics.MeanAbsoluteError(name="MAE")]}
    )

    cw_y2 = class_weights_binary(y2_tr)
    sw_y2_tr = np.array([cw_y2[int(v)] for v in y2_tr], dtype=np.float32)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    ]

    _ = mt.fit(
        X_tr, {'y1': y1_tr, 'y2': y2_tr, 'y3': y3_tr},
        sample_weight={'y2': sw_y2_tr},
        validation_data=(X_va, {'y1': y1_va, 'y2': y2_va, 'y3': y3_va}),
        epochs=EPOCHS, batch_size=BATCH, verbose=1, callbacks=callbacks
    )

    model_path_fold = os.path.join(MODEL_DIR, f"deeptcn_MTL_fold{fold_id}.keras")
    mt.save(model_path_fold)
    print("Saved fold model:", model_path_fold)

    y1p, y2p, y3p = mt.predict(X_te, batch_size=BATCH, verbose=0)

    y1_pred = np.clip(y1p.ravel(), 0.0, 10.0)
    y2_hat  = np.argmax(y2p, axis=1).astype(int)
    y3_pred = np.clip(y3p.ravel() * y3_scale, 0.0, 10.0)

    _ = eval_regression_with_plots(
        y1_te, y1_pred,
        title_prefix=f"Fold{fold_id}_y1_fms",
        outdir=OUTDIR, scatter_kind=SCATTER_KIND
    )
    _ = eval_regression_with_plots(
        y3_te_raw, y3_pred,
        title_prefix=f"Fold{fold_id}_y3_blur_level",
        outdir=OUTDIR, scatter_kind=SCATTER_KIND
    )

    cm_y2 = confusion_matrix(y2_te, y2_hat, labels=[0,1])
    plot_cm(cm_y2, labels=[0,1], title=f"Fold{fold_id} y2 — Confusion Matrix",
            save_path=os.path.join(OUTDIR, f"cm_fold{fold_id}_y2.png"))

    mae1 = mean_absolute_error(y1_te, y1_pred)
    rmse1 = np.sqrt(mean_squared_error(y1_te, y1_pred))
    r21   = r2_score(y1_te, y1_pred)
    r1    = np.corrcoef(y1_te, y1_pred)[0,1] if (np.std(y1_te)>0 and np.std(y1_pred)>0) else np.nan

    mae3 = mean_absolute_error(y3_te_raw, y3_pred)
    rmse3 = np.sqrt(mean_squared_error(y3_te_raw, y3_pred))
    r23   = r2_score(y3_te_raw, y3_pred)
    r3    = np.corrcoef(y3_te_raw, y3_pred)[0,1] if (np.std(y3_te_raw)>0 and np.std(y3_pred)>0) else np.nan

    acc2 = accuracy_score(y2_te, y2_hat)

    print(f"[Fold {fold_id}] y1  MAE={mae1:.4f} RMSE={rmse1:.4f} R2={r21:.4f} r={r1:.4f}")
    print(f"[Fold {fold_id}] y3  MAE={mae3:.4f} RMSE={rmse3:.4f} R2={r23:.4f} r={r3:.4f}")
    print(f"[Fold {fold_id}] y2  ACC={acc2:.4f}")

    fold_metrics["y1_MAE"].append(mae1); fold_metrics["y1_RMSE"].append(rmse1); fold_metrics["y1_R2"].append(r21); fold_metrics["y1_r"].append(r1)
    fold_metrics["y3_MAE"].append(mae3); fold_metrics["y3_RMSE"].append(rmse3); fold_metrics["y3_R2"].append(r23); fold_metrics["y3_r"].append(r3)
    fold_metrics["y2_ACC"].append(acc2)

def summarize(name, vals):
    vals = np.array(vals, dtype=float)
    return f"{name}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}  (per-fold: {', '.join(f'{v:.4f}' for v in vals)})"

print("\n===================== {K_FOLDS}-fold CV Summary (finetune dataset) =====================")
print(summarize("y1 MAE",  fold_metrics["y1_MAE"]))
print(summarize("y1 RMSE", fold_metrics["y1_RMSE"]))
print(summarize("y1 R2",   fold_metrics["y1_R2"]))
print(summarize("y1 r",    fold_metrics["y1_r"]))
print(summarize("y3 MAE",  fold_metrics["y3_MAE"]))
print(summarize("y3 RMSE", fold_metrics["y3_RMSE"]))
print(summarize("y3 R2",   fold_metrics["y3_R2"]))
print(summarize("y3 r",    fold_metrics["y3_r"]))
print(summarize("y2 ACC",  fold_metrics["y2_ACC"]))