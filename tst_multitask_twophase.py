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

BACKBONE_WEIGHTS = "/mnt/data/transformer_backbone_pretrained_MTL.h5"
MODEL_DIR        = "/mnt/data/tst_MTL_cv_models"
OUTDIR           = "/mnt/data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOAD_PRETRAINED = True
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
PRE_HZ = 2
FT_HZ  = 5

WINDOW_PRE = DURATION_SEC * PRE_HZ
WINDOW_FT  = DURATION_SEC * FT_HZ
STRIDE_PRE = max(1, WINDOW_PRE // 4)
STRIDE_FT  = max(1, WINDOW_FT  // 4)

TST_D_MODEL   = 128
TST_NUM_HEADS = 8
TST_FF_DIM    = 256
TST_NUM_BLOCKS= 4
TST_DROPOUT   = 0.10

EMBED_DIM = 128

EPOCHS_PRE=40
EPOCHS_FT_HEADS=12
EPOCHS_FT_LAST=20
EPOCHS_FT_ALL=60
BATCH=64

K_FOLDS = 10

VAL_SIZE=0.15
TEST_SIZE=0.15
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
    iterator = dd.groupby(gkeys, sort=False) if gkeys else [("(none)", dd)]

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

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i   = np.arange(d_model)[None, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
    pe = pos * angle_rates
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    pe = tf.constant(pe, dtype=tf.float32)
    return tf.expand_dims(pe, axis=0)

def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout=0.1):
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(y, y)
    y = layers.Dropout(dropout)(y)
    x = layers.Add()([x, y])

    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(ff_dim, activation="relu")(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(d_model)(y)
    y = layers.Dropout(dropout)(y)
    return layers.Add()([x, y])

def build_backbone(input_shape,
                   d_model=TST_D_MODEL,
                   num_heads=TST_NUM_HEADS,
                   ff_dim=TST_FF_DIM,
                   num_blocks=TST_NUM_BLOCKS,
                   dropout_rate=TST_DROPOUT,
                   embed_dim=EMBED_DIM):
    inp = layers.Input(shape=input_shape, name="input")
    x = layers.Dense(d_model, name="proj_to_d_model")(inp)
    seq_len = input_shape[0]
    pe = positional_encoding(seq_len, d_model)
    x = layers.Add(name="add_positional_encoding")([x, pe])
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout=dropout_rate)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(embed_dim, activation="relu", name="embed")(x)
    x = layers.Dropout(0.2)(x)
    return keras.Model(inp, x, name="TimeSeriesTransformer")

def build_pre_model_regression(input_shape):
    bb = build_backbone(input_shape)
    y1 = layers.Dense(1, activation="linear", name="y1")(bb.output)
    return keras.Model(bb.input, y1, name="PretrainRegY1"), bb

def build_multitask_model(backbone):
    x  = backbone.output
    y1 = layers.Dense(1, activation="linear", name="y1")(x)
    y2 = layers.Dense(2, activation="softmax", name="y2")(x)
    y3 = layers.Dense(1, activation="linear", name="y3")(x)
    return keras.Model(backbone.input, [y1, y2, y3], name="MTL_y1reg_y2bin_y3reg")

def unfreeze_last_n(backbone, n):
    for L in backbone.layers:
        L.trainable = False
    elig_types = (layers.Dense, layers.BatchNormalization, layers.MultiHeadAttention, layers.LayerNormalization)
    elig = [L for L in backbone.layers if isinstance(L, elig_types)]
    for L in elig[-n:]:
        L.trainable = True

pre_df = pdf
ft_df  = fdf

FEATURES = drop_quasi_constant(ft_df, FEATURES, thr=1e-6)
FEATURES = drop_quasi_constant(pre_df, FEATURES, thr=1e-6)
print("Using FEATURES:", FEATURES)

if TIME_COL in ft_df.columns:
    ft_df[TIME_COL] = pd.to_datetime(ft_df[TIME_COL], errors="coerce")

X1_all, [y1_all], meta1 = windowize_with_meta(
    pre_df, FEATURES, [Y1_COL],
    window=WINDOW_PRE, stride=STRIDE_PRE,
    id_col=ID_COL, cond_col=COND_COL, time_col=None,
    block_len=WINDOW_PRE
)
X1_all, y1_all, meta1 = drop_nan_windows(X1_all, y1_all, meta=meta1)

gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
idx_all = np.arange(len(meta1))
trainval_idx, test_idx = next(gss1.split(idx_all, groups=meta1["block_id"].values))
gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE/(1.0 - TEST_SIZE), random_state=SEED)
train_idx, val_idx = next(gss2.split(trainval_idx, groups=meta1["block_id"].values[trainval_idx]))
train_idx = trainval_idx[train_idx]; val_idx = trainval_idx[val_idx]

X1_tr, X1_va, X1_te = X1_all[train_idx], X1_all[val_idx], X1_all[test_idx]
y1_tr, y1_va, y1_te = y1_all[train_idx], y1_all[val_idx], y1_all[test_idx]

z1 = zclip_fit(X1_tr); X1_tr = z1(X1_tr); X1_va = z1(X1_va); X1_te = z1(X1_te)
m1 = minmax_fit(X1_tr); X1_tr = m1(X1_tr); X1_va = m1(X1_va); X1_te = m1(X1_te)

input_shape_pre = (X1_tr.shape[1], X1_tr.shape[2])
model_pre, backbone = build_pre_model_regression(input_shape_pre)
model_pre.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.Huber(delta=1.0),
    metrics=[keras.metrics.MeanAbsoluteError(name="MAE")]
)
_ = model_pre.fit(
    X1_tr, y1_tr,
    validation_data=(X1_va, y1_va),
    epochs=EPOCHS_PRE, batch_size=BATCH, verbose=1
)
backbone.save_weights(BACKBONE_WEIGHTS)

y1_pred_te_pre = model_pre.predict(X1_te, batch_size=BATCH, verbose=0).ravel()
_ = eval_regression_with_plots(
    y1_te, y1_pred_te_pre,
    title_prefix="Pretrain_y1_fms",
    outdir=OUTDIR, scatter_kind=SCATTER_KIND
)

X2_all, [y1f_all, y2_all, y3_all], meta2 = windowize_with_meta(
    ft_df, FEATURES, [Y1_COL, Y2_COL, Y3_COL],
    window=WINDOW_FT, stride=STRIDE_FT,
    id_col=ID_COL, cond_col=COND_COL, time_col=(TIME_COL if TIME_COL in ft_df.columns else None),
    block_len=WINDOW_FT
)
X2_all, y1f_all, y2_all, y3_all, meta2 = drop_nan_windows(X2_all, y1f_all, y2_all, y3_all, meta=meta2)
y2_all = y2_all.astype(int)

y3_scale = 10.0
y3_all_raw = y3_all.copy()
y3_all = y3_all / y3_scale

gkf = GroupKFold(n_splits=K_FOLDS)
groups = meta2["block_id"].values
fold_metrics = {
    "y1_MAE": [], "y1_RMSE": [], "y1
