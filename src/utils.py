"""
公共工具模块：中文字体配置、无监督 V 折交叉验证选 K、子采样稳定性评估（PCS 框架）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold


def setup_chinese_font():
    """配置 Matplotlib 简体中文字体与负号显示。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300


def unsupervised_cv_kmeans(X, k_values, n_splits=5, random_state=42, n_init=10):
    """
    按照 VDS 第 7.7.1 节实现无监督 V 折交叉验证选择 K：
    1. 将样本划分为 V 折；
    2. 在 V-1 折训练集上运行 K-Means 得到簇中心；
    3. 将留出的伪验证折样本分配给最近的训练簇中心；
    4. 在伪验证折上计算平均轮廓系数与平均到中心平方距离。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for k in k_values:
        fold_sils = []
        fold_dists = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
            km.fit(X_train)
            val_labels = km.predict(X_val)

            n_unique = len(np.unique(val_labels))
            if 2 <= n_unique < len(X_val):
                sil = silhouette_score(X_val, val_labels)
            else:
                sil = np.nan
            fold_sils.append(sil)

            centers = km.cluster_centers_[val_labels]
            sq_dist = np.mean(np.sum((X_val - centers) ** 2, axis=1))
            fold_dists.append(sq_dist)

        results[k] = {
            "cv_sil_mean": float(np.nanmean(fold_sils)),
            "cv_sil_std": float(np.nanstd(fold_sils)),
            "cv_dist_mean": float(np.mean(fold_dists)),
            "cv_dist_std": float(np.std(fold_dists)),
        }
    return results


def subsampling_stability_ari(
    X,
    cluster_fn,
    n_runs=20,
    subsample_frac=0.8,
    random_state=42,
):
    """
    PCS 数据扰动稳定性检验（7.8 节）：
    重复进行多次随机子采样（默认 80%），对任意两次子采样结果的公共样本计算 ARI。
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    sub_n = int(n * subsample_frac)

    idx_list = []
    label_maps = []
    for run in range(n_runs):
        idx = np.sort(rng.choice(n, size=sub_n, replace=False))
        labels = cluster_fn(X[idx], seed=random_state + run)
        idx_list.append(set(idx.tolist()))
        label_maps.append(dict(zip(idx.tolist(), labels.tolist())))

    ari_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            common = sorted(idx_list[i].intersection(idx_list[j]))
            if len(common) >= 10:
                la = [label_maps[i][c] for c in common]
                lb = [label_maps[j][c] for c in common]
                ari_scores.append(adjusted_rand_score(la, lb))
    return np.array(ari_scores)
