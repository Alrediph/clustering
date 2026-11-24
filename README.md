# Veridical Data Science (Ch. 7) — 聚类分析与 PCS 稳健性审查

本仓库为数学建模课程汇报配套项目，基于 Bin Yu & Rebecca L. Barter 所著 [*Veridical Data Science*](https://vdsbook.com/) 第 7 章（**Clustering**），在 **USDA FNDDS 食物营养数据集（8,690 × 57）** 与 **UCI Wine 葡萄酒数据集（178 × 13）** 上完整复现了聚类建模与 **PCS（Predictability, Computability, Stability）稳健性审查**流程。

## 仓库结构

```text
.
├── src/
│   ├── utils.py                    # 中文字体配置、无监督 V 折交叉验证选 K、子采样 ARI 稳定性
│   ├── wine_clustering.py          # UCI Wine 完整实验与 5 张中文图表生成
│   └── nutrition_clustering.py     # USDA FNDDS 食物营养完整实验与 4 张中文图表生成
├── figures/                        # 自动生成的 9 张简体中文高清实验图（300 DPI）
├── data/nutrition/
│   └── fndds_clean.csv             # 按原书逻辑清洗后的 FNDDS 宽表（8690 × 57）
├── results_wine.json               # UCI Wine 量化实验结果
├── results_nutrition.json          # USDA FNDDS 量化实验结果
├── VDS_Ch7_聚类分析学习笔记.md     # 第 7 章系统学习笔记
├── 实验报告_FoodNutrition与UCI_Wine.md  # 双数据集对比实验报告
├── 聚类分析汇报架构_VDS_Ch7.md     # 20-30 分钟数学建模课堂汇报架构（含图表嵌入位）
└── requirements.txt
```

## 快速复现

```bash
pip install -r requirements.txt

# 运行 UCI Wine 实验（约 3 秒）
python src/wine_clustering.py

# 运行 USDA FNDDS 食物营养实验（使用缓存宽表约 35 秒）
python src/nutrition_clustering.py
```

> 注：原始 USDA 大表 `food.csv` 与 `food_nutrient.csv` 体积较大未上传至本仓库，可从原书官方配套仓库 [Yu-Group/vds-book-supplementary](https://github.com/Yu-Group/vds-book-supplementary) 获取；仓库已保留清洗后的 `data/nutrition/fndds_clean.csv` 以便一键复现。
