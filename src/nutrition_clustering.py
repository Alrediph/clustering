"""
USDA FNDDS 食物营养数据集聚类分析与 PCS 稳健性审查（复现 VDS 第 7 章主案例）
数据来源：Yu-Group/vds-book-supplementary (USDA Food and Nutrient Database for Dietary Studies)
输出：
- data/nutrition/fndds_clean.csv（清洗后的 FNDDS 宽表缓存）
- figures/nutrition_01_log_vs_raw.png
- figures/nutrition_02_choose_k_cv.png
- figures/nutrition_03_pcs_stability.png
- figures/nutrition_04_cluster_profiles.png
- results_nutrition.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

from utils import setup_chinese_font, unsupervised_cv_kmeans, subsampling_stability_ari


KEY_NUTRIENTS_CN = {
    "protein": "蛋白质",
    "fat": "总脂肪",
    "carbohydrates": "碳水化合物",
    "fiber": "膳食纤维",
    "sugars": "总糖",
    "water": "水分",
    "sodium": "钠",
    "calcium": "钙",
    "iron": "铁",
    "potassium": "钾",
    "cholesterol": "胆固醇",
    "vitamin_c": "维生素C",
    "vitamin_a": "维生素A",
    "vitamin_b12": "维生素B12",
    "saturated_fat": "饱和脂肪",
}


def load_or_clean_fndds(data_dir="data/nutrition"):
    """按原书 clean_food_data.py 逻辑清洗 USDA FNDDS 数据并缓存。"""
    cache_path = os.path.join(data_dir, "fndds_clean.csv")
    if os.path.exists(cache_path):
        print(f"从缓存加载已清洗数据: {cache_path}")
        return pd.read_csv(cache_path, index_col=0)

    print("首次运行：正在按原书逻辑清洗 USDA FNDDS 原始表...")
    food = pd.read_csv(
        os.path.join(data_dir, "food.csv"),
        usecols=["fdc_id", "data_type", "description"],
    )
    fndds_food = food[food["data_type"] == "survey_fndds_food"].copy()
    fndds_ids = set(fndds_food["fdc_id"].tolist())
    nutrient_name = pd.read_csv(os.path.join(data_dir, "nutrient_name.csv"))

    # 分块读取大表，过滤出 FNDDS 食物记录（兼容尾部截断行）
    chunks = []
    reader = pd.read_csv(
        os.path.join(data_dir, "food_nutrient.csv"),
        usecols=["fdc_id", "nutrient_id", "amount"],
        chunksize=500_000,
        on_bad_lines="skip",
    )
    try:
        for ch in reader:
            sub = ch[ch["fdc_id"].isin(fndds_ids)]
            if not sub.empty:
                chunks.append(sub)
    except Exception as e:
        print(f"读取末尾遇到提示（已保留前面全部匹配行）: {e}")

    fn = pd.concat(chunks, ignore_index=True)
    merged = fn.merge(fndds_food, on="fdc_id", how="left").merge(
        nutrient_name, on="nutrient_id", how="left"
    )
    food_index = merged["description"].drop_duplicates()

    food_clean = (
        merged[["description", "amount", "nutrient_name"]]
        .drop_duplicates()
        .dropna(subset=["nutrient_name"])
        .groupby(["description", "nutrient_name"])["amount"]
        .mean()
        .unstack()
    )
    food_clean.columns.name = None
    food_clean = food_clean.reindex(index=food_index)

    # 去除方差为 0 或全缺失列
    food_clean = food_clean.dropna(axis=1, how="all")
    food_clean = food_clean.loc[:, food_clean.std(skipna=True) > 0]

    food_clean.to_csv(cache_path)
    print(f"清洗完成！共 {food_clean.shape[0]} 种食物 × {food_clean.shape[1]} 项营养素，已缓存至 {cache_path}")
    return food_clean


def preprocess_food_data(food_clean, log_transform=True, center=True, scale=True, remove_fat=False):
    """复现原书 preprocess_food_data.py。"""
    df = food_clean.copy()
    if log_transform:
        df = np.log10(df.clip(lower=0) + 1.0)
    if center:
        df = df - df.mean()
    if scale:
        df = df / df.std().replace(0, 1.0)
    if remove_fat and "fat" in df.columns:
        df = df.drop(columns=["fat"])
    df = df.fillna(df.mean())
    return df


def get_cluster_prototypes(X, labels, food_names, centers, n_proto=4):
    """按 VDS 7.4.1 节提取每个簇中离质心最近的代表样本（原型食物）。"""
    prototypes = {}
    for k in range(centers.shape[0]):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            prototypes[f"cluster_{k+1}"] = {"size": 0, "examples": []}
            continue
        dists = np.linalg.norm(X[idx] - centers[k], axis=1)
        nearest = idx[np.argsort(dists)[:n_proto]]
        prototypes[f"cluster_{k+1}"] = {
            "size": int(len(idx)),
            "examples": [str(food_names[i]) for i in nearest],
        }
    return prototypes


def run_nutrition_experiment():
    t0 = time.time()
    setup_chinese_font()
    os.makedirs("figures", exist_ok=True)

    food_clean = load_or_clean_fndds()
    food_names = food_clean.index.to_numpy()
    nutrient_cols = list(food_clean.columns)

    df_raw_std = preprocess_food_data(food_clean, log_transform=False, center=True, scale=True)
    df_log_std = preprocess_food_data(food_clean, log_transform=True, center=True, scale=True)
    df_log_nofat = preprocess_food_data(food_clean, log_transform=True, center=True, scale=True, remove_fat=True)

    X_raw_std = df_raw_std.values
    X_log_std = df_log_std.values
    X_log_nofat = df_log_nofat.values

    summary = {
        "n_foods": int(food_clean.shape[0]),
        "n_nutrients": int(food_clean.shape[1]),
        "sample_nutrients": nutrient_cols[:15],
        "skewness_raw_median": float(food_clean.skew().median()),
        "skewness_log_median": float(np.log10(food_clean.clip(lower=0) + 1.0).skew().median()),
    }

    # =========================================================================
    # 1. 预处理对比：未取对数 vs log10(x+1)（复现原书 7.3 & 7.8 核心发现）
    # =========================================================================
    km_raw_10 = KMeans(n_clusters=10, n_init=15, random_state=42).fit(X_raw_std)
    km_log_10 = KMeans(n_clusters=10, n_init=15, random_state=42).fit(X_log_std)
    km_raw_30 = KMeans(n_clusters=30, n_init=10, random_state=42).fit(X_raw_std)
    km_log_30 = KMeans(n_clusters=30, n_init=10, random_state=42).fit(X_log_std)

    sizes_raw_10 = np.sort(np.bincount(km_raw_10.labels_))[::-1]
    sizes_log_10 = np.sort(np.bincount(km_log_10.labels_))[::-1]
    sizes_raw_30 = np.sort(np.bincount(km_raw_30.labels_))[::-1]
    sizes_log_30 = np.sort(np.bincount(km_log_30.labels_))[::-1]

    summary["log_vs_raw"] = {
        "k10_raw_max_cluster_pct": float(sizes_raw_10[0] / len(X_raw_std) * 100),
        "k10_log_max_cluster_pct": float(sizes_log_10[0] / len(X_log_std) * 100),
        "k10_raw_sizes_sorted": [int(x) for x in sizes_raw_10],
        "k10_log_sizes_sorted": [int(x) for x in sizes_log_10],
        "k10_raw_vs_log_ari": float(adjusted_rand_score(km_raw_10.labels_, km_log_10.labels_)),
        "k30_raw_max_cluster_pct": float(sizes_raw_30[0] / len(X_raw_std) * 100),
        "k30_log_max_cluster_pct": float(sizes_log_30[0] / len(X_log_std) * 100),
        "k30_raw_top3_pct": float(sizes_raw_30[:3].sum() / len(X_raw_std) * 100),
        "k30_log_top3_pct": float(sizes_log_30[:3].sum() / len(X_log_std) * 100),
        "k30_raw_vs_log_ari": float(adjusted_rand_score(km_raw_30.labels_, km_log_30.labels_)),
    }

    # 图 1：未取对数 vs 对数变换 PCA 2D + 簇大小分布
    pca_raw = PCA(n_components=2, random_state=42).fit(X_raw_std)
    Z_raw = pca_raw.transform(X_raw_std)
    pca_log = PCA(n_components=2, random_state=42).fit(X_log_std)
    Z_log = pca_log.transform(X_log_std)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    cmap = plt.get_cmap("tab10")

    axes[0, 0].scatter(Z_raw[:, 0], Z_raw[:, 1], c=km_raw_10.labels_, cmap=cmap, s=6, alpha=0.55)
    axes[0, 0].set_title(
        f"未取对数（仅标准化）K-Means (K=10) PCA 投影\n最大单簇占比 {summary['log_vs_raw']['k10_raw_max_cluster_pct']:.1f}%（严重塌缩）",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel(f"PC1 ({pca_raw.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0, 0].set_ylabel(f"PC2 ({pca_raw.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(Z_log[:, 0], Z_log[:, 1], c=km_log_10.labels_, cmap=cmap, s=6, alpha=0.55)
    axes[0, 1].set_title(
        f"对数变换 log10(x+1) + 标准化 K-Means (K=10) PCA 投影\n最大单簇占比 {summary['log_vs_raw']['k10_log_max_cluster_pct']:.1f}%（结构均衡）",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel(f"PC1 ({pca_log.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0, 1].set_ylabel(f"PC2 ({pca_log.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0, 1].grid(alpha=0.25)

    bars1 = axes[1, 0].bar(range(1, 11), sizes_raw_10, color="#e6550d", alpha=0.85)
    axes[1, 0].set_title("未取对数：各簇样本量排序（K=10）", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("按规模排序的簇编号")
    axes[1, 0].set_ylabel("食物数量")
    axes[1, 0].set_xticks(range(1, 11))
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].text(1, sizes_raw_10[0] + 120, f"{sizes_raw_10[0]} ({summary['log_vs_raw']['k10_raw_max_cluster_pct']:.1f}%)", ha="center", fontsize=9, fontweight="bold", color="#a63603")

    bars2 = axes[1, 1].bar(range(1, 11), sizes_log_10, color="#3182bd", alpha=0.85)
    axes[1, 1].set_title("对数变换 + 标准化：各簇样本量排序（K=10）", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("按规模排序的簇编号")
    axes[1, 1].set_ylabel("食物数量")
    axes[1, 1].set_xticks(range(1, 11))
    axes[1, 1].set_ylim(0, max(sizes_raw_10[0] * 1.15, sizes_log_10[0] * 1.2))
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].text(1, sizes_log_10[0] + 120, f"{sizes_log_10[0]} ({summary['log_vs_raw']['k10_log_max_cluster_pct']:.1f}%)", ha="center", fontsize=9, fontweight="bold", color="#08519c")

    fig.suptitle("USDA FNDDS 食物营养数据：对数变换对高偏态数据聚类的决定性影响（复现 VDS 第 7 章）", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig("figures/nutrition_01_log_vs_raw.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 2. 选择 K：WCSS + 轮廓系数 + 5 折无监督交叉验证 (7.5 & 7.7)
    # =========================================================================
    k_values = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
    wcss_list = []
    sil_list = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_log_std)
        wcss_list.append(float(km.inertia_))
        sil_list.append(float(silhouette_score(X_log_std, km.labels_, sample_size=3000, random_state=42)))

    cv_res = unsupervised_cv_kmeans(X_log_std, k_values, n_splits=5, random_state=42, n_init=8)
    cv_sil_means = [cv_res[k]["cv_sil_mean"] for k in k_values]
    cv_sil_stds = [cv_res[k]["cv_sil_std"] for k in k_values]
    cv_dist_means = [cv_res[k]["cv_dist_mean"] for k in k_values]

    summary["choose_k"] = {
        "k_values": k_values,
        "wcss": wcss_list,
        "silhouette": sil_list,
        "cv_silhouette_mean": cv_sil_means,
        "cv_silhouette_std": cv_sil_stds,
        "cv_dist_mean": cv_dist_means,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    axes[0].plot(k_values, [w / 1e5 for w in wcss_list], "o-", color="#1f77b4", linewidth=2)
    axes[0].axvline(10, color="#ff7f0e", linestyle="--", alpha=0.8, label="中粒度候选 K=10")
    axes[0].axvline(30, color="#d62728", linestyle=":", alpha=0.85, label="原书细粒度 K=30")
    axes[0].set_title("簇内平方和（WCSS，单位 $10^5$）", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("簇数 K")
    axes[0].set_ylabel("WCSS ($\\times 10^5$)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_values, sil_list, "s-", color="#2ca02c", linewidth=2)
    axes[1].axvline(10, color="#ff7f0e", linestyle="--", alpha=0.8, label="K=10")
    axes[1].axvline(30, color="#d62728", linestyle=":", alpha=0.85, label="K=30")
    axes[1].set_title("平均轮廓系数（抽样 3000 样本估计）", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("簇数 K")
    axes[1].set_ylabel("轮廓系数")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    axes[2].errorbar(k_values, cv_sil_means, yerr=cv_sil_stds, fmt="D-", color="#9467bd", capsize=4, linewidth=2)
    axes[2].axvline(10, color="#ff7f0e", linestyle="--", alpha=0.8, label="K=10")
    axes[2].axvline(30, color="#d62728", linestyle=":", alpha=0.85, label="K=30")
    axes[2].set_title("5 折无监督交叉验证轮廓系数（±1σ）", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("簇数 K")
    axes[2].set_ylabel("伪验证折轮廓系数")
    axes[2].legend(fontsize=8.5)
    axes[2].grid(alpha=0.3)

    fig.suptitle("USDA FNDDS 食物营养数据：多指标与 5 折无监督交叉验证选择簇数 K", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/nutrition_02_choose_k_cv.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 3. PCS 稳健性审查：扰动 ARI 矩阵 + 子采样稳定性 (7.8)
    # =========================================================================
    # 为控制层次聚类内存与耗时，在 3000 个固定随机子样本上做全算法横向 ARI 对比
    rng = np.random.default_rng(42)
    eval_idx = np.sort(rng.choice(len(X_log_std), size=3000, replace=False))

    km_s42 = KMeans(n_clusters=10, n_init=15, random_state=42).fit_predict(X_log_std[eval_idx])
    km_s100 = KMeans(n_clusters=10, n_init=15, random_state=100).fit_predict(X_log_std[eval_idx])
    km_s2024 = KMeans(n_clusters=10, n_init=15, random_state=2024).fit_predict(X_log_std[eval_idx])
    km_nofat = KMeans(n_clusters=10, n_init=15, random_state=42).fit_predict(X_log_nofat[eval_idx])
    hc_ward = AgglomerativeClustering(n_clusters=10, linkage="ward").fit_predict(X_log_std[eval_idx])
    km_raw_sub = KMeans(n_clusters=10, n_init=15, random_state=42).fit_predict(X_raw_std[eval_idx])

    configs = [
        ("K-Means(对数,s42)", km_s42),
        ("K-Means(对数,s100)", km_s100),
        ("K-Means(对数,s2024)", km_s2024),
        ("K-Means(对数,去脂肪特征)", km_nofat),
        ("层次聚类 Ward(对数)", hc_ward),
        ("K-Means(未取对数)", km_raw_sub),
    ]
    m = len(configs)
    ari_mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            ari_mat[i, j] = adjusted_rand_score(configs[i][1], configs[j][1])

    # 80% 随机子采样稳定性（全量 8690 样本，12 次重复）
    sub_log_10 = subsampling_stability_ari(
        X_log_std, lambda X_sub, seed: KMeans(n_clusters=10, n_init=8, random_state=seed).fit_predict(X_sub), n_runs=12
    )
    sub_log_30 = subsampling_stability_ari(
        X_log_std, lambda X_sub, seed: KMeans(n_clusters=30, n_init=6, random_state=seed).fit_predict(X_sub), n_runs=12
    )
    sub_raw_10 = subsampling_stability_ari(
        X_raw_std, lambda X_sub, seed: KMeans(n_clusters=10, n_init=8, random_state=seed).fit_predict(X_sub), n_runs=12
    )

    summary["pcs_ari_matrix_key_pairs"] = {
        "log_s42_vs_s100": float(ari_mat[0, 1]),
        "log_s42_vs_s2024": float(ari_mat[0, 2]),
        "log_s42_vs_remove_fat": float(ari_mat[0, 3]),
        "log_s42_vs_ward": float(ari_mat[0, 4]),
        "log_s42_vs_raw": float(ari_mat[0, 5]),
    }
    summary["pcs_subsampling_ari"] = {
        "log_k10_mean": float(sub_log_10.mean()),
        "log_k10_std": float(sub_log_10.std()),
        "log_k30_mean": float(sub_log_30.mean()),
        "log_k30_std": float(sub_log_30.std()),
        "raw_k10_mean": float(sub_raw_10.mean()),
        "raw_k10_std": float(sub_raw_10.std()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
    im = axes[0].imshow(ari_mat, cmap="YlGnBu", vmin=0, vmax=1)
    names = [c[0] for c in configs]
    axes[0].set_xticks(range(m))
    axes[0].set_yticks(range(m))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8.8)
    axes[0].set_yticklabels(names, fontsize=8.8)
    for i in range(m):
        for j in range(m):
            val = ari_mat[i, j]
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.55 else "black", fontsize=8.5)
    axes[0].set_title("跨随机种子、特征删减、算法与预处理扰动 ARI 矩阵（K=10）", fontsize=10.5, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="调整兰德指数 (ARI)")

    bp = axes[1].boxplot(
        [sub_log_10, sub_log_30, sub_raw_10],
        tick_labels=["对数+标准化\n(K=10)", "对数+标准化\n(K=30)", "未取对数\n(K=10)"],
        patch_artist=True,
        medianprops={"color": "#d62728", "linewidth": 2},
    )
    for patch, col in zip(bp["boxes"], ["#6baed6", "#9e9ac8", "#fd8d3c"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.78)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("公共样本两两 ARI")
    axes[1].set_title("全量数据 80% 随机子采样稳定性（12 次重复）", fontsize=10.5, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("USDA FNDDS 食物营养数据：PCS 稳健性审查结果", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("figures/nutrition_03_pcs_stability.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 4. 高维特征画像与代表食物原型抽样 (7.4 & 7.9)
    # =========================================================================
    available_keys = [c for c in KEY_NUTRIENTS_CN.keys() if c in df_log_std.columns]
    # 如果某些列名略有不同，用子串补充常见营养素
    if len(available_keys) < 10:
        extra = [c for c in df_log_std.columns if c not in available_keys][: (12 - len(available_keys))]
        available_keys.extend(extra)

    df_prof = df_log_std[available_keys].copy()
    df_prof["cluster"] = km_log_10.labels_
    profile = df_prof.groupby("cluster").mean()
    sizes_10 = [int((km_log_10.labels_ == k).sum()) for k in range(10)]

    col_labels_cn = [KEY_NUTRIENTS_CN.get(c, c) for c in available_keys]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    im = ax.imshow(profile.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(col_labels_cn)))
    ax.set_xticklabels(col_labels_cn, rotation=30, ha="right", fontsize=9.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"簇 {k+1} (n={sizes_10[k]})" for k in range(10)], fontsize=9.5)
    for i in range(10):
        for j in range(len(col_labels_cn)):
            v = profile.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", color="white" if abs(v) > 0.85 else "black", fontsize=7.8)
    fig.colorbar(im, ax=ax, label="对数标准化后的簇均值 (Z-score)")
    ax.set_title("USDA FNDDS 食物营养聚类（K=10）核心营养素特征画像热力图", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/nutrition_04_cluster_profiles.png", bbox_inches="tight")
    plt.close()

    summary["prototypes_k10"] = get_cluster_prototypes(X_log_std, km_log_10.labels_, food_names, km_log_10.cluster_centers_, n_proto=4)
    summary["prototypes_k30_top8"] = dict(
        sorted(
            get_cluster_prototypes(X_log_std, km_log_30.labels_, food_names, km_log_30.cluster_centers_, n_proto=3).items(),
            key=lambda kv: kv[1]["size"],
            reverse=True,
        )[:8]
    )
    summary["runtime_seconds"] = round(time.time() - t0, 2)

    with open("results_nutrition.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"USDA FNDDS 实验完成！耗时 {summary['runtime_seconds']} 秒")
    print("Log vs Raw:", json.dumps(summary["log_vs_raw"], ensure_ascii=False, indent=2))
    print("PCS Key ARI:", json.dumps(summary["pcs_ari_matrix_key_pairs"], ensure_ascii=False, indent=2))
    print("PCS Subsampling:", json.dumps(summary["pcs_subsampling_ari"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_nutrition_experiment()
