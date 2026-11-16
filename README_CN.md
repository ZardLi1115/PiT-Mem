# PiT-Mem：面向无前视偏差金融投研 Agent 的持久化双时态记忆存储与混合索引引擎（PVLDB Research Track 全套工程包）

本目录包含 **PiT-Mem** (*A Point-in-Time Bi-Temporal Storage and Indexing Engine for Look-Ahead-Free Agentic Memory*) 的完整 C++17 持久化存储与索引引擎、**431,217 条真实去重金融语料**驱动的双时态基准集 **FinPiT-Bench**、Google Colab Pro **NVIDIA A100-SXM4-80GB (`167 GiB` RAM + `12` vCPU + 本地 NVMe SSD)** 200 万级实测结果、PVLDB 矢量图表，以及编译完成的 **PVLDB Volume 20 官方非匿名双栏正文 12 页 + 第 13–14 页参考文献英文论文 PDF**（英文复现说明见 `README.md`）。

- **公开 GitHub 代码仓库**：https://github.com/ZardLi1115/PiT-Mem
- **公开全量数据与归档镜像（Google Drive）**：https://drive.google.com/drive/folders/14uNPJJ5iPavmf9UI4cL-SdeZZuzDWJmu

---

## 一、目录结构总览（Google Drive 归档布局）

```text
PiT-Mem_VLDB_Project/
├── README_CN.md                         # 项目总览、核心架构突破与 A100 200万级实测导读（本文档）
├── COLAB_GUIDE_CN.md                    # Google Colab Pro A100 一键复现、断点续跑与本地 NVMe 加速指南
├── cpp/                                 # C++17 + AVX2 + OpenMP + POSIX mmap + CRC32 WAL 核心存储引擎（~6,220 LOC + 1,169 LOC 基线）
│   ├── include/
│   │   ├── pit_types.hpp                # 64-bit 全局版本 ID 双时态七元组模型、PhaseFairSharedMutex 与 64B 对齐分配器
│   │   ├── simd_kernels.hpp             # AVX2 8路无符号四边刺穿过滤 + 非饱和 7x8-bit maddubs SQ8 核 + 4路 FMA FP32 核
│   │   ├── bit_hnsw.hpp                 # 流式因果宏段 BiT-HNSW（128K->416K Vitter 滞后卷积 + 1,032 B/rec 常驻 RAM + mmap 冷 FP32 + 解析代价路由 + CRC32 WAL）
│   │   ├── mvcc_provenance.hpp          # SQL:2011 EVOLVE/RESTATE 算子 + 全引擎 PVWL 事务 WAL & PITMANF1 原子清单 + 自适应 k_fetch 拆包重排 + 跨层检索
│   │   ├── rap_compact.hpp              # RAP-Compact："PIT3" 单位超球面对齐 SQ8 质心 + 56B 锚点槽位 + 28B 成员跨度 + 冷态撤回覆盖层
│   │   └── baselines_cpp.hpp            # 同硬件 C++17 对比基线（Grid-SQ8-Scan 消融、ACORN-BiTemp 2-Hop、HNSW-Post、SeRF、DSG）
│   ├── src/bindings.cpp                 # pybind11 零拷贝 NumPy 绑定（释放 GIL 并受 PhaseFairSharedMutex 与原子状态机保护）
│   └── setup.py / CMakeLists.txt        # 一键编译脚本（自动启用 -O3 -std=c++17 -mavx2 -mfma -fopenmp）
├── pit_mem/                             # Python 数据管线、外部数据库基线、评测编排与 PVLDB 绘图层
│   ├── covenant_parser.py               # 模式参数化契约与通用 SEC XBRL 财务科目确定性多跨度抽取器
│   ├── gdrive_sync.py                   # Google Drive v3 幂等文件夹递归同步工具
│   ├── data_pipeline/                   # ALFRED + SEC EDGAR XBRL/8-K Item 4.02 + 394,529 条去重金融披露/新闻流与 GPU 编码
│   ├── baselines/                       # 16GB shared_buffers 预热调优 PG-BiTemp (pgvector v0.8.0)、Zep/Graphiti、Mem0、GraphRAG
│   ├── benchmarks/                      # 六大系统实验（Exp 1~6D）+ 自动止损审计器 + test_mw1_mw4_fixes.py 全量回归测试
│   └── plotting/                        # PVLDB Best Paper 范式矢量 PDF 绘图（含原生 TikZ fig2）与 LaTeX 宏自动生成器
├── scripts/
│   └── build_notebooks.py               # Colab Pro 交互式 Notebook 自动生成脚本
├── notebooks/                           # Google Colab Pro 交互式实验 Notebooks
│   ├── 00_Colab_Pro_Master_Pipeline.ipynb
│   ├── 01_Data_and_GPU_Embedding.ipynb
│   ├── 02_System_Benchmarks_Exp1_to_3.ipynb
│   └── 03_Agent_Evaluation_Exp4_to_5.ipynb
├── data/
│   └── raw_seed/                        # 431,217 条真实去重公开金融语料（零合成克隆刷密度）
│       ├── alfred_macro_pit.parquet     # 美联储 ALFRED 7 大宏观序列多版本修正（3,464 条观测，2,659 次历史修正）
│       ├── sec_edgar_pit.parquet        # SEC EDGAR 111 家上市公司 33,224 条财报记录（含 2025–2026 重述与 75 份 8-K Item 4.02）
│       ├── hf_fin_news_texts.parquet    # 394,529 条去重 SEC 10-K QA 与金融新闻长文本（合计 431,217 条独立真实源语料）
│       ├── protocol_a_post2025.json     # 协议 A：2025.01–2026.11 大模型预训练截断后 180 条真实 SEC XBRL 重述轨迹
│       └── protocol_b_counterfactual.json # 协议 B：200 家脱敏实体乘性扰动 + 定量负面债务契约违约翻转评测集
├── results/
│   ├── local_pilot/                     # 本地快速回归测试结果
│   └── colab_pro/                       # Google Colab Pro NVIDIA A100-SXM4-80GB 200万级全量实测 CSV/JSON 与审稿归档
│       ├── exp1_scalability_qps_recall.csv
│       ├── exp2_selectivity_sweep.csv
│       ├── exp3_restate_ingestion.csv
│       ├── exp3_theta_ablation.csv
│       ├── exp3_clustered_burst.csv
│       ├── exp4_leakage_and_audit.csv
│       ├── exp5_rap_compact_ablation.csv
│       ├── exp5_tiered_hot_cold.csv
│       ├── exp6_streaming_and_persistence.csv
│       ├── exp6_nvme_offload.csv
│       ├── exp6_concurrent_mixed.csv
│       ├── hardware_env.json
│       ├── corpus_manifold_stats.json
│       ├── stop_loss_audit_report.json
│       └── colab_pro_execution.log
└── paper/                               # PVLDB Research Track 标准双栏正文 12 页 + 第 13–14 页参考文献源码与成品
    ├── main.tex                         # 正文严格落于 Pages 1–12、References 始于 Page 13 的完整 LaTeX 源码
    ├── generated_metrics.tex            # 从 results/colab_pro/ 自动生成的 100% 精确指标宏
    ├── generated_tables.tex             # 从 results/colab_pro/ 自动生成的等召回率对齐 LaTeX 表格宏
    ├── references.bib                   # 52 篇涵盖经典时态索引（MVBT/TSB-Tree/Timeline/HINT）、MVCC、评测模型/语料与 2024–2026 ANNS/Agent 文献库
    ├── figures/                         # 全部 7 张 PVLDB Best Paper 范式出版级矢量 PDF 图
    └── PiT-Mem_PVLDB.pdf                # 最终编译生成的 14 页（12 页正文 + 2 页参考文献）英文论文 PDF
```

---

## 二、核心架构突破与系统创新

1. **流式因果宏段 `BiT-HNSW`、解析多段代价路由与冷态 FP32 `mmap` 分层存储**
   - **有界因果宏段（$128\text{K} \to 256\text{K} \to 416\text{K}$）与在线 Vitter 滞后卷积校准**：在线维护 4,096 样本 **Vitter Algorithm R** 因果报告滞后蓄水池 $\Delta_j = \max(0, t_k^s - t_v^s)$，投影下一宏段有效时间分位数，并配备**全桶 $>20\%$ 偏斜早封存守卫**（消除 `(15,15)` 边界堆积与内部桶倾斜），保留段内紧凑 **32-bit 邻接边字**（`4b gk | 4b gv | 24b local_id`）与双预算因果前缀不变性（Theorem 1）。
   - **解析多段代价路由器与凹平方根 $\sqrt{s_m}$ 束分配**：基于 `StratifiedSIMDPilot6`（每格至多 6 个等步长 AVX2 块）在线估计跨出生重叠宏段的全局存活密度 $\hat{\rho} = \hat{N}_{\text{alive}}/\hat{N}_{\text{born}}$，直接对比多段图遍历代价 $\mathcal{C}_{\text{graph}}$ 与分格 SIMD 扫描代价 $\mathcal{C}_{\text{scan}}(N_{\text{slots}})$（零硬编码常数截断），彻底消除稀疏工况路由倒挂；图分支采用凹平方根束分配 $ef_m \propto \sqrt{s_m}$ 与 $R \le 1,536$ 精确 FP32 重排，将长生命周期多段快照召回率推入 $93.1\% \sim 95.2\%$ 高 SLA 区间。
   - **`1,032 B/rec` 核心索引数组常驻内存 + POSIX `mmap` 冷态 FP32 NVMe 卸载**：热路径仅常驻单份 64B 对齐分格 SQ8 编码、双预算邻接表与 SoA 死亡时间戳（**`1,032 B/rec`**，$2\text{M}$ 规模核心索引仅占 **`1.92 GiB` RAM**，比 `PG-BiTemp` 的 `4.87 GiB` 小 `2.5×`），将 `3,072 B/rec` 原始 FP32 向量按需映射至本地 NVMe SSD（All-RAM 总计 `7.64 GiB`），配合 `"PITMSNP3"` + `"PITPROV2"` + `"PITMANF1"` 清单级原子快照与 `"PWAL"`/`"PVWL"` IEEE 802.3 CRC-32 事务 WAL 实现崩溃一致性恢复。

2. **无饥饿 `PhaseFairSharedMutex`、自适应 $k_{\text{fetch}}$ 倍增 `Lazy-Branch MVCC` 与凸热度分支**
   - 采用阶段公平读写锁 `PhaseFairSharedMutex` 协调高并发快照读与流式 `INSERT`/`EVOLVE`/`RESTATE`/宏段滚动，彻底杜绝写饥饿。
   - 读取时基于搜索前沿距离 $d_{\text{front}}$ 校验洁净完备性条件 $n_{\le \text{front}} \ge k$：常规查询零额外轮次返回，密集跨季重述突发自动几何倍增 $k_{\text{fetch}} \leftarrow \min(2k_{\text{fetch}}, k_{\max})$ 递归拆包多跳谱系并执行精确 FP32 Top-$k$ 重排（Theorem 3 证明 $|\mathcal{O}| \le k, \text{LCR}=0, \text{STR}=0$ 及前沿洁净完备性）；配合无锁 `CAS(0 -> 1)` 仅对高频热点脏反思触发异步重固化。

3. **单位超球面对齐 `RAP-Compact`（`"PIT3"`）与冷热跨层统一时点检索（`query_tiered_pit`）**
   - 二进制 `"PIT3"` 槽位段内共置字典编码 `56B PackedAnchorTuple`（IEEE-754 双精度阈值）、`28B PackedMemberSpan`（含 64-bit `source_mem_id` 与成员级双时态边界）及单位超球面 $\ell_2$ 归一化子空间 SQ8 质心，使冷层弦距离 $\|\tilde{\mathbf{q}} - \bar{\mathbf{c}}\|_2^2 \in [0, 4]$ 与热层 `BiT-HNSW` 精确对齐，支持 `query_tiered_pit` 单次调用完成跨热/冷层去重归并与确定性数值锚点联接。

---

## 三、Google Colab Pro (`NVIDIA A100-SXM4-80GB` + 12 vCPU + 167 GiB RAM + NVMe SSD) 实测结果摘要

- **Exp 1（跨量级双工况评测：$100\text{K} \to 500\text{K} \to 2,000,000$ 条 768 维 BGE 稠密向量）**：
  - **Workload 1A（中等快照 $\sigma \approx 12.0\%$，100% 触发双预算因果图）**：在 $N = 2,000,000$ 下，`PiT-Mem` 在 `ef=192` 达到 **`93.1%` Recall@10 @ `283` 1T QPS（`4.92 ms` P99）**，峰值达到 **`95.2%` Recall@10**（`ef=96` 达 `493` 1T / `3,160` 12T QPS，`90.5%` R@10，`2.97 ms` P99）——相比 `Grid-SQ8-Scan`（`31.4` QPS，`32.66 ms` P99）实现 **`9.0× ~ 15.7×` 吞吐加速与 `6.6×` P99 降低**，全面超越 `ACORN-BiTemp`（`46.1` QPS -> **`6.1×` QPS**）与 16GB 预热内存 `PG-BiTemp`（`132.8` QPS -> **`2.1×` QPS**）。
  - **Workload 1B（稀疏历史审计 $\sigma \approx 1.0\%$，零倒挂解析路由）**：解析代价路由器在 `100K`、`500K`、`2M` 全量级精准选择分格 AVX2 SQ8 扫描（`0%` 图触发），在 $N = 2,000,000$ 下实现 **`100.0%` Recall@10 @ `565` 1T QPS（`2,935` 12T QPS，`2.15 ms` P99，核心常驻 RAM `1.92 GiB` + `5.72 GiB` mmap，建表 `183.9 s`）**，与纯扫描上限 `Grid-SQ8-Scan`（`536` QPS，`2.28 ms` P99）完全持平且零倒挂，相比预热 `PG-BiTemp`（`72.5` QPS，`31.76 ms` P99）实现 **`7.8× ~ 186×` QPS 与 `14.8× ~ 266×` P99 加速**。
- **Exp 2（跨 3 个数量级选择度扫描 $\sigma \in [0.05\%, 50\%]$，$N = 1,000,000$）**：
  - 宽窗口 $\sigma = 50\%$ 下 `PiT-Mem`（`422` QPS，`3.38 ms` P99）比 `Grid-SQ8-Scan`（`20` QPS，`51.05 ms` P99）快 **`15.1×` P99 / `21.4×` QPS**；窄窗口 $\sigma = 0.05\%$ 平滑切换扫描分支，取得 **`100.0%` Recall@10、`8,974` QPS、`0.17 ms` P99**（而 `ACORN-BiTemp` 跌至 `32.5%` R@10，`HNSW-Post` 跌至 `54.0%` R@10）。
- **Exp 3 & Exp 3B（重述写入吞吐、端到端 `Write + Read` 总 Token 与 3 跳聚集重述突发）**：
  - 10% 重述流下维持 **`1,012` ops/s**（`904.5 μs` P50），**`PiT-Mem`（$\theta_{\text{hot}}=3$）** 平均节省 **`75.7%` 端到端总 LLM Token**；在 3 跳谱系 `8~32` 条顶层聚集重述突发（Exp 3B）下，静态 $k_{\text{fetch}}$ 在突发 `24`/`32` 时跌至 `10.0%` 洁净 R@10，而自适应 $k_{\text{fetch}}$ 倍增在全突发强度下保持 **`100.0%` 洁净 Recall@10**（平均 `1.05` 轮，`1.20 ms` P99，`0.0%` STR）。
- **Exp 4（双协议前视泄漏、审计 F1 与回测夏普虚高，A100 `Qwen2.5-7B-Instruct`）**：
  - `PiT-Mem (Full)` 在协议 A（2025–2026 SEC XBRL）与协议 B 上取得 **`0.00%` LCR**、**`0.0%` Stale Taint** 与最高审计 F1（**`71.2%` / `66.7%`**），完全消除 `Mem0`（`+4.35`）与 `Zep/Graphiti`（`+0.25` / `+1.53`）的夏普比率虚高。
- **Exp 5 & Exp 5C（`RAP-Compact` 对决 `Parquet-Zstd` / `SQLite-BTree` 及跨热冷层统一检索）**：
  - $\beta = 25\%$ 下实现 **`13.2×` 压缩比**（$\beta = 15\%$ 达 `15.7×`）、**`8.4 μs` 延迟**与 **`100.0%` 契约问答准确率**（比 `Parquet-Zstd` 的 `1,929 μs` **快 `230×`**，比 `SQLite-BTree` **快 `5.2×`**），在 2025 后真实 SEC EDGAR XBRL 上达 **`100.0%` 准确率、`18.5×` 压缩比、`9.0 μs` 延迟**；跨层 `query_tiered_pit`（Exp 5C）实现 **`90.0%` 统一问答准确率**（显著超越 `Hot-Only` 的 `42.0%` 与 `Cold-Only` 的 `50.0%`，`17.0 μs` 均值，`0.0%` STR）。
- **Exp 6A–6D（在线偏斜消除、NVMe 快照/WAL、冷 FP32 卸载与 12 线程混合并发压测）**：
  - **Exp 6A（$N = 1,000,000$ 流式到达）**：`(15,15)` 堆积从冻结网格的 `95.1%` 降至 **`0.0%`**，封存段最大单格占比 **`6.7%`**、活跃尾段最大单格占比 **`7.8%`**，吞吐达 **`3,153` QPS（`0.36 ms` P99，提速 `3.7×`）**。
  - **Exp 6B（NVMe 快照与 CRC-32 WAL）**：`7.46 GiB` 快照写入 `26.2 s`（`292 MiB/s`）、预热恢复 `6.40 s`（`1.17 GiB/s`）、`1,000` 帧 CRC-32 WAL 重放 **`68.8 ms`** 位级一致。
  - **Exp 6C（$N = 2,000,000$ POSIX `mmap` 冷 FP32 卸载，严格对齐 Table 3 `ef=192` / `ef=96`）**：常驻核心内存压缩至 **`1.92 GiB`（`1,032 B/rec`）**；热页缓存吞吐（W1A `281 / 1,882` QPS，W1B `628 / 3,339` QPS）与全内存持平；即使每次微批前通过 `DONTNEED` 强制清空页缓存触发真实冷 NVMe 缺页，仍维持 W1A **`86 / 250` QPS（`14.38 ms` P99，`93.3%` R@10）** 与 W1B **`281 / 944` QPS（`4.94 ms` P99，`100.0%` R@10）**。
  - **Exp 6D（$N = 500,000$ 12 线程并发读 + 持续写/重述/宏段滚动）**：`PhaseFairSharedMutex` 在 4 次在线宏段滚动与持续 `INSERT`/`EVOLVE`/`RESTATE` 并发下维持 **`1,613` 混合读 QPS（`13.04 ms` P99）+ `134` 写 ops/s**，**`0` 次快照隔离或基数违例**。
