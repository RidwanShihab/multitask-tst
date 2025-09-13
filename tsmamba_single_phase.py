import os, warnings, gc, math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable, get_custom_objects
from dataclasses import dataclass, asdict
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.set_printoptions(edgeitems=3, linewidth=120)

MODEL_DIR  = "/mnt/data/tsmamba_MTL_cv_models"
OUTDIR     = "/mnt/data"
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
ID_COL   = "participant_id"
COND_COL  = "condition"
TIME_COL  = "Timestamp"

Y1_COL = "fms"
Y2_COL = "blur_effectiveness"
Y3_COL = "blur_label"

DURATION_SEC = 3
FT_HZ  = 5
WINDOW_FT  = DURATION_SEC * FT_HZ
STRIDE_FT  = max(1, WINDOW_FT // 4)

EMBED_DIM = 128
TSM_BASE = dict(
    model_states=32,
    projection_expand_factor=2,
    conv_kernel_size=4,
    num_layers=6,
    dropout_rate=0.2,
    conv_use_bias=True,
    dense_use_bias=False,
)

EPOCHS = 90
BATCH  = 64

K_FOLDS = 10

VAL_SIZE  = 0.15
TEST_SIZE = 0.15
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

def eval_regression_with_plots(y_true, y_pred, title_prefix, outdir=OUTDIR, scatter_kind="scatter"):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    pearson = (
        np.corrcoef(y_true, y_pred)[0,1]
        if (np.std(y_true)>0 and np.std(y_pred)>0) else np.nan
    )
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
    iterator = dd.groupby(gkeys, sort=False) if gkeys else [(("none","none"), dd)]

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

get_custom_objects().pop("tsmamba>MambaBlock", None)
get_custom_objects().pop("tsmamba>ResidualBlock", None)

@dataclass
class TSMArgs:
    model_input_dims: int = 64
    model_states: int = 64
    projection_expand_factor: int = 2
    conv_kernel_size: int = 4
    num_layers: int = 6
    dropout_rate: float = 0.2
    conv_use_bias: bool = True
    dense_use_bias: bool = False
    delta_t_rank: int = None

    def finalize(self):
        if self.delta_t_rank is None:
            self.delta_t_rank = math.ceil(self.model_input_dims/16)
        self.model_internal_dim = int(self.projection_expand_factor * self.model_input_dims)
        return self

def selective_scan(u, delta, A, B, C, D):
    dA   = tf.einsum('bld,dn->bldn', delta, A)
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    dA_cumsum = tf.pad(dA[:, 1:], [[0,0],[1,1],[0,0],[0,0]])[:, 1:,:,:]
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)
    dA_cumsum = tf.exp(dA_cumsum)
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])

    x = dB_u * dA_cumsum
    x = tf.math.cumsum(x, axis=1) / (dA_cumsum + 1e-12)

    y = tf.einsum('bldn,bln->bld', x, C)
    return y + u * D

@register_keras_serializable(package="tsmamba")
class MambaBlock(layers.Layer):
    def __init__(self, args: TSMArgs, *a, **k):
        super().__init__(*a, **k)
        self.args = args.finalize()

        self.in_projection = layers.Dense(
            self.args.model_internal_dim * 2,
            input_shape=(self.args.model_input_dims,), use_bias=False, name="in_proj"
        )

        self.groups_supported = 'groups' in layers.Conv1D.__init__.__code__.co_varnames
        if self.groups_supported:
            self.conv1d = layers.Conv1D(
                filters=self.args.model_internal_dim,
                kernel_size=self.args.conv_kernel_size,
                padding='causal',
                groups=self.args.model_internal_dim,
                use_bias=self.args.conv_use_bias,
                data_format='channels_last',
                name="dw_conv1d"
            )
        else:
            self.conv1d = layers.SeparableConv1D(
                filters=self.args.model_internal_dim,
                kernel_size=self.args.conv_kernel_size,
                padding='causal',
                use_bias=self.args.conv_use_bias,
                name="sep_conv1d_fallback"
            )

        self.x_projection = layers.Dense(
            self.args.delta_t_rank + self.args.model_states * 2,
            use_bias=False, name="x_proj"
        )
        self.delta_t_projection = layers.Dense(
            self.args.model_internal_dim, use_bias=True, name="delta_proj"
        )

        base = tf.range(1, self.args.model_states + 1, dtype=tf.float32)
        A_init = tf.tile(tf.expand_dims(base, axis=0), [self.args.model_internal_dim, 1])
        self.A_log = tf.Variable(tf.math.log(A_init), trainable=True, dtype=tf.float32, name="SSM_A_log")
        self.D     = tf.Variable(np.ones(self.args.model_internal_dim), trainable=True, dtype=tf.float32, name="SSM_D")

        self.out_projection = layers.Dense(
            self.args.model_input_dims,
            input_shape=(self.args.model_internal_dim,),
            use_bias=self.args.dense_use_bias, name="out_proj"
        )

    def call(self, x):
        x_and_res = self.in_projection(x)
        x_int, res = tf.split(x_and_res, 2, axis=-1)

        xc = self.conv1d(x_int)
        L = tf.shape(x_int)[1]
        xc = xc[:, :L, :]
        xc = tf.nn.swish(xc)

        y = self.ssm(xc)
        y = y * tf.nn.swish(res)
        return self.out_projection(y)

    def ssm(self, x):
        d_int, n = self.A_log.shape
        A = -tf.exp(tf.cast(self.A_log, tf.float32))
        D = tf.cast(self.D, tf.float32)

        x_dbl = self.x_projection(x)
        delta, B, C = tf.split(x_dbl, [self.args.delta_t_rank, n, n], axis=-1)
        delta = tf.nn.softplus(self.delta_t_projection(delta))

        return selective_scan(x, delta, A, B, C, D)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"args": asdict(self.args)})
        return cfg

    @classmethod
    def from_config(cls, config):
        args_dict = config.pop("args")
        args = TSMArgs(**args_dict).finalize()
        return cls(args=args, **config)

@register_keras_serializable(package="tsmamba")
class ResidualBlock(layers.Layer):
    def __init__(self, args: TSMArgs, *a, **k):
        super().__init__(*a, **k)
        self.args = args.finalize()
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="ln")
        self.mixer = MambaBlock(self.args)
        self.drop  = layers.Dropout(self.args.dropout_rate, name="drop")

    def call(self, x):
        return self.drop(self.mixer(self.norm(x)) + x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"args": asdict(self.args)})
        return cfg

    @classmethod
    def from_config(cls, config):
        args_dict = config.pop("args")
        args = TSMArgs(**args_dict).finalize()
        return cls(args=args, **config)

def TSMambaBackboneModel(input_shape, tsm_args: TSMArgs, embed_dim=EMBED_DIM):
    inp = layers.Input(shape=input_shape, name="input")
    args_local = TSMArgs(
        model_input_dims=input_shape[1],
        model_states=tsm_args.model_states,
        projection_expand_factor=tsm_args.projection_expand_factor,
        conv_kernel_size=tsm_args.conv_kernel_size,
        num_layers=tsm_args.num_layers,
        dropout_rate=tsm_args.dropout_rate,
        conv_use_bias=tsm_args.conv_use_bias,
        dense_use_bias=tsm_args.dense_use_bias,
    ).finalize()

    x = inp
    for i in range(args_local.num_layers):
        x = ResidualBlock(args_local, name=f"Residual_{i}")(x)
    x = layers.LayerNormalization(epsilon=1e-5, name="pre_head_norm")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(embed_dim, activation="relu", name="embed")(x)
    x = layers.Dropout(0.2)(x)
    return keras.Model(inp, x, name="TSMambaBackbone")

def build_backbone(input_shape, tsm_base=TSM_BASE, embed_dim=EMBED_DIM):
    args = TSMArgs(
        model_input_dims=input_shape[1],
        model_states=tsm_base["model_states"],
        projection_expand_factor=tsm_base["projection_expand_factor"],
        conv_kernel_size=tsm_base["conv_kernel_size"],
        num_layers=tsm_base["num_layers"],
        dropout_rate=tsm_base["dropout_rate"],
        conv_use_bias=tsm_base["conv_use_bias"],
        dense_use_bias=tsm_base["dense_use_bias"],
    )
    return TSMambaBackboneModel(input_shape, args, embed_dim=embed_dim)

def build_multitask_model(backbone):
    x  = backbone.output
    y1 = layers.Dense(1, activation="linear", name="y1")(x)
    y2 = layers.Dense(2, activation="softmax", name="y2")(x)
    y3 = layers.Dense(1, activation="linear", name="y3")(x)
    return keras.Model(backbone.input, [y1, y2, y3], name="MTL_y1reg_y2bin_y3reg")

ft_df = fdf

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
y2_all = y2_all.astype(int)

y3_scale = 10.0
y3_all_raw = y3_all.copy()
y3_all = y3_all / y3_scale

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
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED+fold_id)
    tr_sub_idx, va_sub_idx = next(gss_inner.split(tr_index, groups=tr_groups))
    tr_idx = tr_index[tr_sub_idx]
    va_idx = tr_index[va_sub_idx]
    te_idx = te_index

    X_tr, X_va, X_te = X_all[tr_idx], X_all[va_idx], X_all[te_idx]
    y1_tr, y1_va, y1_te = y1_all[tr_idx], y1_all[va_idx], y1_all[te_idx]
    y2_tr, y2_va, y2_te = y2_all[tr_idx], y2_all[va_idx], y2_all[te_idx]
    y3_tr, y3_va, y3_te = y3_all[tr_idx], y3_all[va_idx], y3_all[te_idx]
    y3_te_raw = y3_all_raw[te_idx]

    zf = zclip_fit(X_tr); X_tr = zf(X_tr); X_va = zf(X_va); X_te = zf(X_te)
    mf = minmax_fit(X_tr); X_tr = mf(X_tr); X_va = mf(X_va); X_te = mf(X_te)

    keras.backend.clear_session()
    gc.collect()
    input_shape = (X_tr.shape[1], X_tr.shape[2])
    backbone = build_backbone(input_shape)
    mt = build_multitask_model(backbone)

    cw_y2 = class_weights_binary(y2_tr)
    sw_y2_tr = np.array([cw_y2[int(v)] for v in y2_tr], dtype=np.float32)

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
    _ = mt.fit(
        X_tr, {'y1': y1_tr, 'y2': y2_tr, 'y3': y3_tr},
        sample_weight={'y2': sw_y2_tr},
        validation_data=(X_va, {'y1': y1_va, 'y2': y2_va, 'y3': y3_va}),
        epochs=EPOCHS, batch_size=BATCH, verbose=1
    )

    model_dir_fold = os.path.join(MODEL_DIR, f"tsmamba_MTL_fold{fold_id}")
    mt.save(model_dir_fold)
    print("Saved fold model:", model_dir_fold)

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
    r1 = (
        np.corrcoef(y1_te.ravel(), y1_pred.ravel())[0, 1]
        if (np.std(y1_te) > 0 and np.std(y1_pred) > 0) else np.nan
    )

    mae3 = mean_absolute_error(y3_te_raw, y3_pred)
    rmse3 = np.sqrt(mean_squared_error(y3_te_raw, y3_pred))
    r23   = r2_score(y3_te_raw, y3_pred)
    r3 = (
        np.corrcoef(y3_te_raw.ravel(), y3_pred.ravel())[0, 1]
        if (np.std(y3_te_raw) > 0 and np.std(y3_pred) > 0) else np.nan
    )

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

print(f"\n===================== {K_FOLDS}-fold CV Summary (finetune dataset) =====================")
print(summarize("y1 MAE",  fold_metrics["y1_MAE"]))
print(summarize("y1 RMSE", fold_metrics["y1_RMSE"]))
print(summarize("y1 R2",   fold_metrics["y1_R2"]))
print(summarize("y1 r",    fold_metrics["y1_r"]))
print(summarize("y3 MAE",  fold_metrics["y3_MAE"]))
print(summarize("y3 RMSE", fold_metrics["y3_RMSE"]))
print(summarize("y3 R2",   fold_metrics["y3_R2"]))
print(summarize("y3 r",    fold_metrics["y3_r"]))
print(summarize("y2 ACC",  fold_metrics["y2_ACC"]))