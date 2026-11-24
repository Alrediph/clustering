"""
UCI Wine 数据集聚类分析与 PCS 稳健性审查（对应 VDS 第 7 章全流程）
输出：
- figures/wine_01_preprocessing_pca.png
- figures/wine_02_choose_k_cv.png
- figures/wine_03_dendrogram_linkage.png
- figures/wine_04_pcs_stability.png
- figures/wine_05_cluster_profiles.png
- results_wine.json
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, rand_score
from sklearn.preprocessing import StandardScaler

from utils import setup_chinese_font, unsupervised_cv_kmeans, subsampling_stability_ari


FEATURE_CN = {
    "alcohol": "酒精度",
    "malic_acid": "苹果酸",
    "ash": "灰分",
    "alcalinity_of_ash": "灰分碱度",
    "magnesium": "镁含量",
    "total_phenols": "总酚",
    "flavanoids": "类黄酮",
    "nonflavanoid_phenols": "非黄酮酚",
    "proanthocyanins": "原花青素",
    "color_intensity": "色度",
    "hue": "色调",
    "od280/od315_of_diluted_wines": "稀释酒吸光度",
    "proline": "脯氨酸",
}


def run_wine_experiment():
    setup_chinese_font()
    os.makedirs("figures", exist_ok=True)

    wine = load_wine(as_frame=True)
    df = wine.data.copy()
    y_true = wine.target.values
    feature_names = list(df.columns)
    feature_names_cn = [FEATURE_CN[f] for f in feature_names]

    X_raw = df.values
    X_std = StandardScaler().fit_transform(X_raw)
    X_log_std = StandardScaler().fit_transform(np.log1p(X_raw))

    summary = {
        "n_samples": int(X_raw.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "feature_scale_ratio": float(df.std().max() / df.std().min()),
        "max_std_feature": FEATURE_CN[df.std().idxmax()],
        "min_std_feature": FEATURE_CN[df.std().idxmin()],
    }

    # =========================================================================
    # 1. 预处理对比（Raw vs Z-score vs Log1p+Z-score）@ K=3
    # =========================================================================
    km_raw = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X_raw)
    km_std = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X_std)
    km_log = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X_log_std)

    summary["preprocessing_k3"] = {
        "raw": {
            "silhouette": float(silhouette_score(X_raw, km_raw.labels_)),
            "ri_vs_true": float(rand_score(y_true, km_raw.labels_)),
            "ari_vs_true": float(adjusted_rand_score(y_true, km_raw.labels_)),
            "cluster_sizes": [int(c) for c in np.bincount(km_raw.labels_)],
        },
        "std": {
            "silhouette": float(silhouette_score(X_std, km_std.labels_)),
            "ri_vs_true": float(rand_score(y_true, km_std.labels_)),
            "ari_vs_true": float(adjusted_rand_score(y_true, km_std.labels_)),
            "cluster_sizes": [int(c) for c in np.bincount(km_std.labels_)],
        },
        "log_std": {
            "silhouette": float(silhouette_score(X_log_std, km_log.labels_)),
            "ri_vs_true": float(rand_score(y_true, km_log.labels_)),
            "ari_vs_true": float(adjusted_rand_score(y_true, km_log.labels_)),
            "cluster_sizes": [int(c) for c in np.bincount(km_log.labels_)],
        },
    }

    # 图 1：PCA 2D 投影对比预处理影响
    pca = PCA(n_components=2, random_state=42)
    Z_std = pca.fit_transform(X_std)
    var_ratio = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [
        (km_raw.labels_, f"未标准化 K-Means (K=3)\n相对真实品种 ARI = {summary['preprocessing_k3']['raw']['ari_vs_true']:.3f}"),
        (km_std.labels_, f"Z-score 标准化 K-Means (K=3)\n相对真实品种 ARI = {summary['preprocessing_k3']['std']['ari_vs_true']:.3f}"),
        (y_true, "外部参考：真实葡萄酒品种\n（仅作对照，未参与聚类训练）"),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, (labels, title) in zip(axes, panels):
        for c in np.unique(labels):
            mask = labels == c
            ax.scatter(
                Z_std[mask, 0],
                Z_std[mask, 1],
                s=38,
                alpha=0.85,
                color=colors[c % len(colors)],
                edgecolors="white",
                linewidth=0.5,
                label=f"类别 {c+1} (n={mask.sum()})",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"主成分 1 ({var_ratio[0]:.1f}%)")
        ax.set_ylabel(f"主成分 2 ({var_ratio[1]:.1f}%)")
        ax.legend(fontsize=8.5, frameon=True)
        ax.grid(alpha=0.25)

    fig.suptitle("UCI Wine 数据集：特征标准化对 K-Means 聚类结果的决定性影响（PCA 二维投影）", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/wine_01_preprocessing_pca.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 2. 选择 K：WCSS 肘部法则 + 全样本轮廓系数 + 5 折无监督交叉验证 (7.5 & 7.7)
    # =========================================================================
    k_values = list(range(2, 9))
    wcss_list = []
    sil_list = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_std)
        wcss_list.append(float(km.inertia_))
        sil_list.append(float(silhouette_score(X_std, km.labels_)))

    cv_res = unsupervised_cv_kmeans(X_std, k_values, n_splits=5, random_state=42, n_init=20)
    cv_sil_means = [cv_res[k]["cv_sil_mean"] for k in k_values]
    cv_sil_stds = [cv_res[k]["cv_sil_std"] for k in k_values]

    summary["choose_k"] = {
        "k_values": k_values,
        "wcss": wcss_list,
        "silhouette": sil_list,
        "cv_silhouette_mean": cv_sil_means,
        "cv_silhouette_std": cv_sil_stds,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(k_values, wcss_list, "o-", color="#1f77b4", linewidth=2)
    axes[0].axvline(3, color="#d62728", linestyle="--", alpha=0.8, label="拐点 K=3")
    axes[0].set_title("簇内平方和（WCSS 肘部法则）", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("簇数 K")
    axes[0].set_ylabel("WCSS")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_values, sil_list, "s-", color="#2ca02c", linewidth=2)
    axes[1].axvline(3, color="#d62728", linestyle="--", alpha=0.8, label="峰值 K=3")
    axes[1].set_title("全样本平均轮廓系数", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("簇数 K")
    axes[1].set_ylabel("轮廓系数")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].errorbar(k_values, cv_sil_means, yerr=cv_sil_stds, fmt="D-", color="#9467bd", capsize=4, linewidth=2)
    axes[2].axvline(3, color="#d62728", linestyle="--", alpha=0.8, label="峰值 K=3")
    axes[2].set_title("5 折无监督交叉验证轮廓系数（±1σ）", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("簇数 K")
    axes[2].set_ylabel("伪验证折轮廓系数")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("UCI Wine 数据集：多准则与无监督交叉验证确定最优簇数 K", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/wine_02_choose_k_cv.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 3. 层次聚类树状图与 4 种 Linkage 对比 (7.2)
    # =========================================================================
    linkage_methods = [
        ("ward", "Ward 离差平方和"),
        ("complete", "Complete 全连接"),
        ("average", "Average 平均连接"),
        ("single", "Single 单连接"),
    ]
    linkage_metrics = {}
    linkage_labels = {}
    for method, cn_name in linkage_methods:
        hc = AgglomerativeClustering(n_clusters=3, linkage=method).fit(X_std)
        linkage_labels[method] = hc.labels_
        linkage_metrics[method] = {
            "cn_name": cn_name,
            "silhouette": float(silhouette_score(X_std, hc.labels_)),
            "ari_vs_kmeans": float(adjusted_rand_score(km_std.labels_, hc.labels_)),
            "ari_vs_true": float(adjusted_rand_score(y_true, hc.labels_)),
            "cluster_sizes": [int(c) for c in np.bincount(hc.labels_)],
        }
    summary["hierarchical_linkages"] = linkage_metrics

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    Z_link = linkage(X_std, method="ward")
    dendrogram(Z_link, truncate_mode="lastp", p=24, leaf_rotation=45, leaf_font_size=8, color_threshold=19, ax=axes[0])
    axes[0].axhline(19, color="#d62728", linestyle="--", linewidth=1.5, label="切分高度（对应 K=3）")
    axes[0].set_title("Ward 层次聚类树状图（截断展示顶部 24 节点）", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("合并簇编号（括号内为样本数）")
    axes[0].set_ylabel("合并距离（Ward）")
    axes[0].legend()

    x_pos = np.arange(len(linkage_methods))
    width = 0.26
    sils = [linkage_metrics[m]["silhouette"] for m, _ in linkage_methods]
    ari_km = [linkage_metrics[m]["ari_vs_kmeans"] for m, _ in linkage_methods]
    ari_tr = [linkage_metrics[m]["ari_vs_true"] for m, _ in linkage_methods]

    axes[1].bar(x_pos - width, sils, width, label="轮廓系数", color="#1f77b4")
    axes[1].bar(x_pos, ari_km, width, label="与 K-Means 的 ARI", color="#ff7f0e")
    axes[1].bar(x_pos + width, ari_tr, width, label="与真实品种的 ARI", color="#2ca02c")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([cn for _, cn in linkage_methods], rotation=12, fontsize=9)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("指标得分")
    axes[1].set_title("四种 Linkage 准则效果对比（K=3）", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("UCI Wine 数据集：层次聚类树状图与 Linkage 准则敏感性", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/wine_03_dendrogram_linkage.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 4. PCS 稳健性审查：扰动 ARI 矩阵 + 子采样稳定性 (7.8)
    # =========================================================================
    km_std_s2 = KMeans(n_clusters=3, n_init=20, random_state=100).fit(X_std)
    configs = [
        ("K-Means(标准化,s42)", km_std.labels_),
        ("K-Means(标准化,s100)", km_std_s2.labels_),
        ("K-Means(对数+标准化)", km_log.labels_),
        ("层次聚类(Ward)", linkage_labels["ward"]),
        ("层次聚类(Average)", linkage_labels["average"]),
        ("层次聚类(Complete)", linkage_labels["complete"]),
        ("层次聚类(Single)", linkage_labels["single"]),
        ("K-Means(未标准化)", km_raw.labels_),
    ]
    m = len(configs)
    ari_mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            ari_mat[i, j] = adjusted_rand_score(configs[i][1], configs[j][1])

    sub_km_std = subsampling_stability_ari(
        X_std, lambda X_sub, seed: KMeans(n_clusters=3, n_init=15, random_state=seed).fit_predict(X_sub), n_runs=25
    )
    sub_ward = subsampling_stability_ari(
        X_std, lambda X_sub, seed: AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_sub), n_runs=25
    )
    sub_km_raw = subsampling_stability_ari(
        X_raw, lambda X_sub, seed: KMeans(n_clusters=3, n_init=15, random_state=seed).fit_predict(X_sub), n_runs=25
    )

    summary["pcs_subsampling_ari"] = {
        "kmeans_std_mean": float(sub_km_std.mean()),
        "kmeans_std_std": float(sub_km_std.std()),
        "ward_std_mean": float(sub_ward.mean()),
        "ward_std_std": float(sub_ward.std()),
        "kmeans_raw_mean": float(sub_km_raw.mean()),
        "kmeans_raw_std": float(sub_km_raw.std()),
    }
    summary["pcs_ari_matrix_key_pairs"] = {
        "kmeans_seed42_vs_seed100": float(ari_mat[0, 1]),
        "kmeans_std_vs_log_std": float(ari_mat[0, 2]),
        "kmeans_std_vs_ward": float(ari_mat[0, 3]),
        "kmeans_std_vs_single": float(ari_mat[0, 6]),
        "kmeans_std_vs_raw": float(ari_mat[0, 7]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    im = axes[0].imshow(ari_mat, cmap="YlGnBu", vmin=0, vmax=1)
    names = [c[0] for c in configs]
    axes[0].set_xticks(range(m))
    axes[0].set_yticks(range(m))
    axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    axes[0].set_yticklabels(names, fontsize=8.5)
    for i in range(m):
        for j in range(m):
            val = ari_mat[i, j]
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.55 else "black", fontsize=8)
    axes[0].set_title("跨算法与预处理扰动的一致性 ARI 矩阵", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="调整兰德指数 (ARI)")

    bp = axes[1].boxplot(
        [sub_km_std, sub_ward, sub_km_raw],
        tick_labels=["K-Means\n(Z-score)", "层次聚类 Ward\n(Z-score)", "K-Means\n(未标准化)"],
        patch_artist=True,
        medianprops={"color": "#d62728", "linewidth": 2},
    )
    for patch, col in zip(bp["boxes"], ["#6baed6", "#74c476", "#fd8d3c"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("公共样本两两 ARI")
    axes[1].set_title("80% 随机子采样数据扰动稳定性（25 次重复）", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("UCI Wine 数据集：PCS 稳定性审查（算法/预处理扰动 + 数据子采样扰动）", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("figures/wine_04_pcs_stability.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # 5. 高维特征画像可视化 (7.4 & 7.9)
    # =========================================================================
    df_std = pd.DataFrame(X_std, columns=feature_names_cn)
    df_std["cluster"] = km_std.labels_
    profile = df_std.groupby("cluster").mean()

    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(profile.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(feature_names_cn)))
    ax.set_xticklabels(feature_names_cn, rotation=30, ha="right", fontsize=9.5)
    ax.set_yticks(range(3))
    sizes = [int((km_std.labels_ == k).sum()) for k in range(3)]
    ax.set_yticklabels([f"簇 {k+1} (n={sizes[k]})" for k in range(3)], fontsize=10)
    for i in range(3):
        for j in range(len(feature_names_cn)):
            v = profile.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color="white" if abs(v) > 0.85 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Z-score 标准化均值")
    ax.set_title("UCI Wine 最终三类簇的高维化学特征画像（各特征簇均值热力图）", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/wine_05_cluster_profiles.png", bbox_inches="tight")
    plt.close()

    summary["cluster_top_features"] = {}
    for k in range(3):
        s = profile.loc[k].sort_values(ascending=False)
        summary["cluster_top_features"][f"cluster_{k+1}"] = {
            "size": sizes[k],
            "highest_3": {idx: round(float(val), 2) for idx, val in s.head(3).items()},
            "lowest_3": {idx: round(float(val), 2) for idx, val in s.tail(3).items()},
        }

    with open("results_wine.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("UCI Wine 实验完成！")
    print(json.dumps(summary["preprocessing_k3"], ensure_ascii=False, indent=2))
    print("PCS 子采样 ARI:", summary["pcs_subsampling_ari"])


if __name__ == "__main__":
    run_wine_experiment()
