# Цифровой генеалог — отчёт по 70 студенческим образцам

> **Главное:** откройте [`dashboard.html`](dashboard.html) в браузере — это интерактивный дашборд со всеми визуализациями (3D PCA, UMAP, ancestry, kinship-сеть, родословные деревья).

## Входные данные

- **VCF**: 14.3 ГБ, `families_plus_popref.vcf` — 278 сэмплов, 5 906 144 варианта (chr1–22, hg38)
- **70 студенческих** образцов (`s1..s95` с пропусками), известен только возраст
- **208 референсных** из 1000 Genomes (26 популяций × 8 сэмплов, 5 superpop: AFR, AMR, EAS, EUR, SAS)

## Пайплайн

| # | Шаг | Инструмент | Файл-результат |
|---|---|---|---|
| 1 | QC + конвертация VCF → pfile | `plink2`: `--max-alleles 2 --snps-only --autosome --maf 0.05 --geno 0.05 --hwe 1e-10` | `work/data_qc.*` (4 993 075 SNP) |
| 2 | LD-pruning (window 200, step 50, r²<0.2) | `plink2 --indep-pairwise` | `work/data_pruned.*` (181 101 SNP) |
| 3 | PCA (20 PC) | `plink2 --pca 20` | `work/pca_all.eigenvec/.eigenval` |
| 4 | Kinship KING-robust | `plink2 --make-king-table --king-table-filter 0.0442` | `work/king.kin0` (127 пар) |
| 5 | Каскадная классификация: superpop → population | sklearn (RF + kNN-5) | `results/population_predictions.tsv` |
| 6 | Глобальный ancestry (supervised NNLS на superpop частотах) | scipy NNLS | `results/ancestry_proportions.tsv` |
| 7 | UMAP-проекция PC1–10 | umap-learn | `results/umap.png` |
| 8 | Сборка семейных графов | networkx | `results/family_trees.png` |

Все скрипты воспроизводимы из `scripts/`.

## Результаты

### 1. Родство — 15 семей

Все 70 студентов разбились на **15 семей** через рёбра 1-й/2-й степени родства (kinship ≥ 0.0884):

| Семья | n | Состав | Возрасты |
|---|---|---|---|
| 1 | 12 | s6, s10, s42, s43, s46, s59, s61, s76, s84, s87, s92, s93 | 66, 47, 6, 6, 10, 18, 40, 75, 73, 76, 15, 15 |
| 2 | 7 | s1, s16, s30, s33, s34, s91, s94 | 76, 48, 18, 41, 8, 75, 48 |
| 3 | 6 | s26, s47, s48, s51, s57, s69 | 63, 9, 10, 38, 33, 53 |
| 4 | 6 | s28, s38, s49, s74, s75, s79 | 35, 42, 15, 13, 68, 61 |
| 5 | 6 | s5, s13, s17, s50, s83, s85 | 31, 5, 49, 35, 15, 53 |
| 6 | 5 | s14, s67, s68, s73, s95 | 36, 36, 13, 17, 36 |
| 7 | 4 | s9, s27, s32, s54 | 9, 39, 10, 32 |
| 8–15 | 3 каждая | (см. `student_summary.tsv`) | |

Распределение типов связей среди 127 родственных пар:
- **parent-child**: 68 (kinship ~0.25, IBS0 ≈ 0 — обязательно общий аллель в каждом локусе)
- **full-sibling**: 20 (kinship ~0.25, IBS0 > 0.005)
- **2nd-degree**: 36 (kinship 0.09–0.18 — полусибсы, бабушка/дедушка, дядя/тётя)
- **MZ/duplicate**: 3 (kinship 0.5, IBS0 = 0)

**3 пары однояйцевых близнецов** (одинаковый возраст подтверждает): **s92/s93 (15)**, **s16/s94 (48)**, **s67/s95 (36)**.

`results/family_trees.png` — все 15 семей с цветами по superpop, толщина рёбер по типу связи.
`results/kinship_network.png` — общий граф родства.
`results/kinship_pairs.tsv` — полная таблица всех 127 пар.

### 2. Популяционная принадлежность

**Каскадная классификация**: сначала RandomForest предсказывает superpopulation (5 классов) на PC1–10, затем kNN-5 внутри предсказанной superpop предсказывает population (8 классов).

**Точность на 5-fold CV (по референсной панели):**
- superpopulation: **99.0%**
- population: **65.4%** (близкие популяции в одной superpop разделить тяжело, например CEU vs GBR vs IBS — северо-/западно-европейские группы перетекают)

**Распределение студентов:**

| Superpop | n | Топ-population |
|---|---|---|
| EUR | 25 | CEU (19), GBR (4), IBS (2) |
| SAS | 16 | PJL (7), STU (5), ITU (4) |
| AFR | 13 | ASW (6), YRI (4), MSL (2), ACB (2) |
| EAS | 10 | CHS (7), KHV (1), плюс admixed |
| AMR | 7 | CLM (6), PUR (1) |

Полная таблица с топ-3 предсказаниями и вероятностями: `results/population_predictions.tsv`.

`results/pca_superpop.png` — PCA в координатах superpopulation.
`results/pca_population.png` — PCA в координатах 26 популяций.
`results/umap.png` — UMAP-проекция PC1–10, студенты отмечены крестиками с именами.

### 3. Глобальный ancestry (NNLS)

Для каждого студента доли AFR/AMR/EAS/EUR/SAS оценены через **supervised NNLS** на per-superpop allele frequencies (181 101 SNP × 5 superpop):

`results/ancestry_barplot.png` — stacked barplot всех 70 студентов, отсортированных по доминантной ancestry.
`results/ancestry_proportions.tsv` — точные доли.

**Заметки:**
- **57 / 70** студентов имеют максимальную долю < 0.9 — то есть формально «admixed»
- Это в значительной мере артефакт: 1000G популяции AMR (MXL, PEL, CLM, PUR) сами по себе admixed (EUR+NAT+AFR), поэтому при их использовании как «чистого» референса небольшая AMR-компонента появляется почти у всех. Для строгого ancestry-анализа лучше использовать HGDP/SGDP или провести unsupervised ADMIXTURE с K=5 (не сделано в этой итерации).
- **Уверенно admixed** (доминантная < 0.75): 11 человек. Среди них семья s5/s13/s17 (AMR с европейской/южноазиатской примесью).

### 4. Сводная таблица

`results/student_summary.tsv` — главный итоговый файл по 70 студентам, столбцы:

```
sample_id  age  family_id  n_close_relatives
pred_superpop  pred_superpop_prob
pred_pop1  pred_pop1_name  pred_pop1_prob
pred_pop2  pred_pop2_prob  pred_pop3  pred_pop3_prob
anc_AFR  anc_AMR  anc_EAS  anc_EUR  anc_SAS
```

## Ограничения и что улучшить

1. **Reference bias**: 1000G AMR — admixed популяции; supervised ADMIXTURE с ними даёт перекос. Решение: unsupervised ADMIXTURE K=5, либо использовать HGDP/SGDP.
2. **Локальный ancestry / chromosome painting** — не сделано. Требует фазированные данные (например, через `SHAPEIT` или `Beagle`), затем `RFMix` или `Gnomix`. Это +несколько часов работы.
3. **Связи 3-й степени** (кузены) собраны (см. `kinship_pairs.tsv`), но не использованы для группировки в семьи — пороги выбраны консервативно. При желании можно перестроить семьи с порогом 0.0442 — несколько семей объединятся.
4. **Реконструкция родителей**: KING даёт kinship и IBS0, по которым parent–child отличается от full-sibling. Можно дополнительно реконструировать направление (родитель/ребёнок) по возрасту — что отчасти сделано на визуализации семей через размер вершины пропорциональный возрасту.

## Визуализации

### Интерактивный дашборд

[`dashboard.html`](dashboard.html) — самодостаточный HTML с интерактивной графикой Plotly:
- **3D PCA** PC1/2/3, вращающийся, hover с возрастом / семьёй / предсказанной популяцией / ancestry-баром
- **2D PCA** PC1/2 + PC3/4 с тем же hover
- **UMAP** проекция PC1–10
- **Stacked ancestry barplot** 70 студентов
- **Sunburst** superpop → population → student
- **Kinship-network** с цветными рёбрами по типу связи

### Статические high-end визуализации

| Файл | Описание |
|---|---|
| [`radial_families.png`](radial_families.png) | Круговой chord-style: 70 студентов на окружности сгруппированы по семьям, наружные дуги = семьи (цвет = доминантная ancestry), внутри — Bezier-кривые родственных связей |
| [`family_pedigrees.png`](family_pedigrees.png) | Все 15 семей: ось Y = возраст, узлы = ancestry-пироги (5-way), рёбра разного типа разными цветами |
| [`families/family_*.png`](families/) | Hi-res родословные для 6 крупнейших семей |
| [`ancestry_clustered.png`](ancestry_clustered.png) | Stacked ancestry barplot, студенты отсортированы иерархической кластеризацией; цвет подписи = семья |
| [`sankey.html`](sankey.html) | Sankey: superpop → population → family (интерактив) |
| [`pca_superpop.png`](pca_superpop.png), [`pca_population.png`](pca_population.png), [`umap.png`](umap.png) | Статичные PCA/UMAP |
| [`ancestry_barplot.png`](ancestry_barplot.png) | Stacked ancestry, упорядочено по доминирующей ancestry |
| [`scree.png`](scree.png) | Variance explained по PC |

## Файлы

```
results/
  REPORT.md                       — этот отчёт
  dashboard.html                  — главный интерактивный дашборд (открыть в браузере)
  student_summary.tsv             — главная итоговая таблица (70 строк, 19 столбцов)
  population_predictions.tsv      — топ-3 популяции на студента
  superpop_predictions.tsv        — вероятности по 5 superpop
  ancestry_proportions.tsv        — доли AFR/AMR/EAS/EUR/SAS
  kinship_pairs.tsv               — 127 родственных пар с аннотацией
  pca_coords.tsv                  — PCA + UMAP координаты всех 278 сэмплов

  radial_families.png             — круговой "цифровой родовой герб"
  family_pedigrees.png            — 15 семейных деревьев с ancestry-пирогами
  families/family_NN.png          — hi-res родословные крупнейших семей
  ancestry_clustered.png          — иерархически кластеризованный ancestry
  sankey.html                     — superpop → pop → family Sankey
  pca_*.png  umap.png  scree.png  ancestry_barplot.png

scripts/
  01_pca_and_classify.py     — PCA + плоский RF (заменён каскадом)
  02_kinship_families.py     — родство + семьи + базовые family_trees.png
  03_ancestry_nnls.py        — supervised ancestry через NNLS
  04_umap_and_summary.py     — UMAP + итоговая таблица
  05_cascade_classify.py     — каскадная классификация (заменяет 01)
  06_dashboard.py            — сборка интерактивного dashboard.html
  07_pedigree_pies.py        — pedigree-стиль семейных деревьев
  08_radial_views.py         — radial + sankey + clustered ancestry

work/                              — промежуточные plink2 файлы (можно удалить)
```

Воспроизведение всего пайплайна:
```
plink2 → 01 → 02 → 03 → 05 (вызывает 04) → 06 → 07 → 08
```
