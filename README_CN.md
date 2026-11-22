# PiT-Mem：面向无前视偏差金融投研 Agent 的持久化双时态记忆存储与混合索引引擎（PVLDB Research Track 工程归档）

本目录包含 **PiT-Mem** (*A Point-in-Time Bi-Temporal Storage and Indexing Engine for Look-Ahead-Free Agentic Memory*) 的完整 C++17 持久化存储与索引引擎、**431,217 条公开金融种子语料**构建的 **FinPiT-Bench** 基准、Google Colab Pro **NVIDIA A100-SXM4-80GB（`167 GiB` RAM + `12` vCPU + 本地 NVMe SSD）** 200 万级归档实验 CSV、PVLDB 矢量图表，以及严格遵循官方 [`vldbproceedings/VLDB-Template`](https://github.com/vldbproceedings/VLDB-Template)（`acmart.cls` v2.19 + `pvldb.sty` v1.0，零排版压缩 hack）编译而成的 **12 页正文 + 第 13–14 页参考文献** 英文论文 PDF（英文复现说明见 `README.md`）。

- **公开 GitHub 代码仓库**：https://github.com/ZardLi1115/PiT-Mem
- **公开全量数据与归档镜像（Google Drive）**：https://drive.google.com/drive/folders/14uNPJJ5iPavmf9UI4cL-SdeZZuzDWJmu

---

## 一、目录结构总览

```text
PiT-Mem_VLDB_Project/
├── README.md                            # 英文 PVLDB Vol. 20 复现与构建说明
├── README_CN.md                         # 中文架构、实测摘要与评测边界说明（本文档）
├── COLAB_GUIDE_CN.md                    # Google Colab Pro A100 复现与 NVMe 防踩坑说明
├── cpp/                                 # C++17 + AVX2 + OpenMP + POSIX mmap + CRC-32 WAL 引擎
│   ├── include/
│   │   ├── pit_types.hpp                # 64-bit 全局版本 ID 双时态七元组模型、PhaseFairSharedMutex 与 64B 对齐分配器
│   │   ├── simd_kernels.hpp             # AVX2 8路无符号刺穿过滤 + 非饱和 7x8-bit maddubs SQ8 核 + 4路 FMA FP32 核
│   │   ├── bit_hnsw.hpp                 # 流式因果宏段 BiT-HNSW（128K->416K Vitter 滞后卷积 + 1,032 B/rec 核心索引数组 + mmap 冷 FP32 + 解析代价路由 + CRC32 WAL）
│   │   ├── mvcc_provenance.hpp          # SQL:2011 EVOLVE/RESTATE + PVWL 事务 WAL & PITMANF1 原子清单 + 自适应 k_fetch 拆包重排 + 跨层检索
│   │   ├── rap_compact.hpp              # RAP-Compact（"PIT3" 槽位段）：单位超球面 SQ8 质心 + 56B 锚点槽位 + 28B 成员跨度 + 冷态撤回覆盖层
│   │   └── baselines_cpp.hpp            # 同硬件 C++17 双时态适配基线（Grid-SQ8-Scan, ACORN-Style, HNSW-Flat, SeRF-Style, DSG-Style）及 vendored nmslib/hnswlib v0.8.0 绑定
│   ├── third_party/hnswlib/             # 上游 nmslib/hnswlib v0.8.0 头文件（供补充验证与回归测试使用）
│   ├── src/bindings.cpp                 # pybind11 零拷贝 NumPy 绑定（释放 GIL 并受 PhaseFairSharedMutex 保护）
│   └── setup.py / CMakeLists.txt        # C++17 编译配置（-O3 -std=c++17 -mavx2 -mfma -fopenmp）
├── pit_mem/                             # Python 数据管线、外部 DBMS 基线、实验编排与 PVLDB 绘图层
│   ├── covenant_parser.py               # 模式参数化契约与 SEC XBRL 财务科目多跨度抽取器
│   ├── gdrive_sync.py                   # Google Drive v3 幂等文件夹同步工具
│   ├── data_pipeline/                   # ALFRED + SEC EDGAR XBRL/8-K Item 4.02 + 394,529 条去重金融披露/新闻与 GPU 编码
│   ├── baselines/                       # PostgreSQL 16 + pgvector v0.8.0 (PG-BiTemp) 及 Exp 4 策略代理基线（Latest-Summary / Overwrite / Append-Only）
│   ├── benchmarks/                      # Exp 1–6D 实验脚本与 test_mw1_mw4_fixes.py 全量单元/并发回归测试
│   └── plotting/                        # 出版级矢量 PDF 图表生成器与 CSV-to-LaTeX 宏/表格自动生成器
├── notebooks/                           # Google Colab Pro 交互式复现 Notebooks（00–03）
├── data/raw_seed/                       # 431,217 条公开去重金融种子语料 + Protocol A/B 审计用例
├── results/
│   ├── local_pilot/                     # 本地快速 CPU 回归测试结果
│   └── colab_pro/                       # Google Colab Pro A100 200万级归档实验 CSV/JSON 与日志
└── paper/                               # 官方 VLDB-Template (acmart v2.19 + pvldb v1.0) LaTeX 源码、自动生成宏/表格、矢量图与 PiT-Mem_PVLDB.pdf
```

---

## 二、核心架构与设计要点

1. **流式因果宏段 `BiT-HNSW`、解析多段代价路由与冷态 FP32 `mmap` 分层**
   - **有界因果宏段（$128\text{K} \to 256\text{K} \to 416\text{K}$）与在线 Vitter 滞后卷积校准**：在线维护 4,096 样本 **Vitter Algorithm R** 报告滞后蓄水池 $\Delta_j = \max(0, t_k^s - t_v^s)$，投影下一宏段有效时间分位数，并配备全桶 $>20\%$ 偏斜早封存守卫，保留段内 32-bit 邻接边字（`4b gk | 4b gv | 24b local_id`）与双预算因果前缀不变性（Theorem 1）。
   - **解析多段代价路由器与凹平方根 $\sqrt{s_m}$ 束分配**：基于 `StratifiedSIMDPilot6` 在线估计存活密度，动态对比多段图遍历代价 $\mathcal{C}_{\text{graph}}$ 与分格 SIMD 扫描代价 $\mathcal{C}_{\text{scan}}(N_{\text{slots}})$，在宽/中等选择度下走因果图、在窄窗口下平滑切换分格 AVX2 SQ8 扫描。
   - **单份 SQ8 核心索引数组 + POSIX `mmap` 冷态 FP32 卸载**：计数的核心索引数组仅需 **`1,032 B/rec`**（$2\text{M}$ 规模 **`1.92 GiB`**，不含辅助 ID 哈希映射元数据及 OS 页缓存驻留），将 `3,072 B/rec` 原始 FP32 向量按需映射至本地 NVMe SSD（All-RAM 计数布局 `7.64 GiB`），配合清单级原子快照与 IEEE 802.3 CRC-32 事务 WAL。

2. **无饥饿 `PhaseFairSharedMutex` 与自适应 $k_{\text{fetch}}$ 倍增 `Lazy-Branch MVCC`**
   - 以阶段公平读写锁 `PhaseFairSharedMutex` 协调高并发快照读与流式写入/宏段滚动，结合搜索前沿距离 $d_{\text{front}}$ 校验洁净完备性并在多跳重述突发时几何倍增 $k_{\text{fetch}}$ 执行精确 FP32 重排（保证 $|\mathcal{O}| \le k, \text{LCR}=0, \text{STR}=0$），通过无锁 `CAS(0 -> 1)` 仅对高频热点脏反思触发异步重固化。

3. **`RAP-Compact`（`"PIT3"`）与跨热/冷层统一时点检索（`query_tiered_pit`）**
   - 在紧凑二进制槽位段中共置 56B `PackedAnchorTuple`（IEEE-754 双精度阈值）、28B `PackedMemberSpan` 与归一化子空间 SQ8 质心（使冷层分数与热层共享 $[0, 4]$ 数值区间），支持 `query_tiered_pit` 单次调用完成热/冷层去重归并与结构化锚点返回。

---

## 三、Google Colab Pro (`A100-SXM4-80GB` + 12 vCPU + 167 GiB RAM) 归档实测摘要

- **Exp 1（跨量级评测：$100\text{K} \to 500\text{K} \to 2,000,000$ 条 768 维 BGE 向量）**：
  - **Workload 1A（中等快照 $\sigma \approx 12.0\%$，100% 触发双预算因果图）**：$N = 2,000,000$ 下 `PiT-Mem` 在 `ef=192` 达到 **`93.1%` Recall@10 @ `283` 1T QPS（`4.92 ms` P99）**，峰值 `95.2%` R@10（`ef=96` 为 `493` 1T / `3,160` 12T QPS，`90.5%` R@10）；相比 `Grid-SQ8-Scan`（`31` QPS，`32.66 ms` P99）实现 **`9.0×` QPS 与 `6.6×` P99 优势**，优于 C++17 `ACORN-Style` 适配基线（`46` QPS -> **`6.1×` QPS**）与 `PG-BiTemp`（`132.8` QPS -> **`2.1×` QPS**）。
  - **Workload 1B（稀疏历史审计 $\sigma \approx 1.0\%$）**：路由器在全量级切换至分格 AVX2 SQ8 扫描（`0%` 图触发），$N = 2,000,000$ 下取得 **`100.0%` 实测采样 Recall@10 @ `565` 1T QPS（`2,935` 12T QPS，`2.15 ms` P99，构建 `183.9 s`）**，与 `Grid-SQ8-Scan`（`536` QPS）持平，相比 `PG-BiTemp`（`72.5` QPS，`31.76 ms` P99，磁盘关系+索引 `4.87 GiB`）实现 **`7.8×` QPS 与 `14.8×` P99 优势**。
- **Exp 2（选择度扫描 $\sigma \in [0.05\%, 50\%]$，$N = 1,000,000$）**：
  - 宽窗口 $\sigma = 50\%$ 下 `PiT-Mem`（`422` QPS，`3.38 ms` P99）比 `Grid-SQ8-Scan`（`20` QPS，`51.05 ms` P99）快 **`15.1×` P99**；窄窗口 $\sigma = 0.05\%$ 切换扫描分支，维持 `100.0%` 实测召回、`8,974` QPS 与 `0.17 ms` P99，而 C++17 `HNSW-Flat` / `ACORN-Style` 适配基线分别跌至 `54.0%` / `32.5%`。
- **Exp 3 & Exp 3B（重述元数据吞吐、校准 Token 代价模型与 3 跳聚集突发）**：
  - 10% 重述流下 C++ 元数据循环尝试速率为 **`1,012` ops/s**（`904.5 μs` P50）；在校准的 `Write + Read` Token 记账模型下，`PiT-Mem`（$\theta_{\text{hot}}=3$）平均估算节省 **`75.7%` LLM Token**（$\theta_{\text{hot}}=5$ 为模型最优点 `-76.0%`）。在实测 3 跳 `24`/`32` 条顶层聚集重述突发（Exp 3B）下，静态 $k_{\text{fetch}}$ 跌至 `10.0%` 洁净 R@10，而自适应 $k_{\text{fetch}}$ 倍增保持 **`100.0%` 洁净 Recall@10**（均值 `1.05` 轮，`1.20 ms` P99，`0.0%` STR）。
- **Exp 4（双协议前视泄漏与审计 F1，`Qwen2.5-7B-Instruct`）**：
  - 在每协议前 120 条案例（合计 480 次查询/策略）上对比三种自实现记忆策略代理，`PiT-Mem (Full)` 取得 **`0.00%` LCR**、**`0.0%` Stale Taint** 与最高审计 F1（**`71.2%` / `66.7%`**），显著优于 `Overwrite (Mem0-style)`（`99.2%` LCR）、`Append-Only (GraphRAG-style)`（`68.3%` LCR / `100.0%` Stale Taint）与 `Latest-Summary (Graphiti-style)`（`72.5%` / `50.0%` LCR）。
- **Exp 5 & Exp 5C（`RAP-Compact` 槽位保真、延迟与跨层统一检索）**：
  - 在 200 条生成 Protocol B 契约模板（给定已知财务实际值）及 XBRL 模板披露上，三类抽取器驱动方案均实现 `100.0%` 模板槽位保真；$\beta = 25\%$ 下常驻 C++ `RAP-Compact`（SQ8）达 **`13.2×` 压缩比**与 **`8.4 μs` 查询延迟**（比内存 `SQLite-BTree` 的 `43.4 μs` **快 `5.2×`**；按请求读取文件的 `Parquet-Zstd` 为 `1,929 μs`）。跨热/冷层 `query_tiered_pit`（Exp 5C）实现 **`90.0%` 目标槽位覆盖率**（对比 `Hot-Only` `42.0%`、`Cold-Only` `50.0%`，`17.0 μs` 均值，`0.0%` STR）。
- **Exp 6A–6D（流式偏斜、NVMe 快照/WAL、冷 FP32 卸载与 12 线程并发压测）**：
  - **Exp 6A（$N = 1,000,000$）**：`(15,15)` 堆积从冻结网格 `95.1%` 降至 **`0.0%`**，封存/尾段最大单格占比 `6.7%` / `7.8%`，吞吐达 **`3,153` QPS（`0.36 ms` P99，提速 `3.7×`）**。
  - **Exp 6B**：`7.46 GiB` 快照落盘 `26.2 s`（`292 MiB/s`）、顺序 `mmap` 加载 `6.40 s`（`1.17 GiB/s`）、`1,000` 帧 CRC-32 WAL 重放 `68.8 ms`（2M 规模验证记录数一致性，撕裂事务回滚与全状态一致由单元回归测试验证）。
  - **Exp 6C（$N = 2,000,000$ POSIX `mmap` 冷 FP32 卸载）**：计数核心索引数组降至 **`1.92 GiB`（`1,032 B/rec`）**；热缓存吞吐与 All-RAM 持平（W1A `281` vs `277` QPS，W1B `628` vs `599` QPS）；在每次微批前发出 `POSIX_FADV_DONTNEED` + `MADV_DONTNEED` 建议驱逐后，仍维持 W1A **`86 / 250` QPS（`14.38 ms` P99，`93.3%` R@10）** 与 W1B **`281 / 944` QPS（`4.94 ms` P99，`100.0%` R@10）**（注：归档运行未埋点物理 NVMe 主缺页计数）。
  - **Exp 6D（$N = 500,000$ 12 线程混合并发）**：`PhaseFairSharedMutex` 在 4 次在线宏段滚动与持续写入/重述并发下维持 **`1,613` 混合读 QPS（`13.04 ms` P99）+ `134` 尝试写 ops/s**，**`0` 次隔离或基数违例**。

---

## 四、评测范围与补充工程说明

- 论文 Section 7.7（*Threats to Validity & Scope*）对归档实验的基线适配性质（Exp 1–2 自实现 C++17 双时态图适配、Exp 4 策略代理、Exp 3 校准 Token 模型、Exp 5 模板槽位检查与不同执行路径、Exp 6C 计数数组内存与建议驱逐边界）做了透明披露。
- 仓库 `cpp/` 目录同时包含审计后加固的 C++17 实现（显式 `committed` 事务返回标志、无冲突 `HIST_SLICE_BASE` 内部 ID 空间、撕裂 WAL 位级状态一致性校验以及 vendored `nmslib/hnswlib` v0.8.0 绑定），可通过 `PYTHONPATH=. python3 -m pit_mem.benchmarks.test_mw1_mw4_fixes` 一键验证。
