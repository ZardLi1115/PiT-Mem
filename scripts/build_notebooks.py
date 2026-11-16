"""Generate clean, turnkey Colab Pro Jupyter Notebooks (.ipynb) using nbformat v4."""

from __future__ import annotations

import os
import nbformat as nbf


def make_master_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# PiT-Mem (PVLDB): 双时态金融 Agent 记忆存储与索引引擎 — Colab Pro 一键复现主控台\n\n"
            "本 Notebook 对应论文 **PiT-Mem: A Point-in-Time Bi-Temporal Storage and Indexing Engine for Look-Ahead-Free Agentic Memory** 的完整单卡 A100 High-RAM 实验流水线（`Exp 1 ~ Exp 6`）。\n\n"
            "### 核心架构与实验覆盖：\n"
            "1. **高速本地 NVMe 冷热分层与持久化评测 (`Exp 6B/6C/6D`)**：`1,032 B/rec` 核心索引常驻内存 + `3,072 B/rec` 原始 FP32 向量 POSIX `mmap` 按需分页卸载（`DONTNEED` 冷页缺失压测）、原子 `mmap` CSR 快照与 IEEE 802.3 CRC32 WAL 恢复、`PhaseFairSharedMutex` 12 线程并发读 + 持续流式写入/重述/宏段滚动压测，以及 16GB `shared_buffers` 热驻留 PostgreSQL (`pgvector v0.8.0`) 均运行在 `/content/local_ssd/` 本地 NVMe 上。\n"
            "2. **431,217 条真实去重金融语料 + 分片断点续跑**：自动拉取 ALFRED 宏观多版本流、SEC EDGAR `10-K/10-Q/8-K Item 4.02` 重述流与 39.4 万条去重金融披露文本，按 20 万条/分片完成 200 万条 `BAAI/bge-base-en-v1.5` (768d FP16) 向量编码，并审计跨轨迹多样性与同谱系演化余弦相似度。\n"
            "3. **全场景对比基线 (`Exp 1 ~ Exp 6`)**：覆盖 Workload 1A (`12%` 凹函数多段束分配高召回因果图区间) 与 Workload 1B (`1%` 零倒挂解析代价路由区间)、`Grid-SQ8-Scan`、`ACORN-BiTemp (SIGMOD'24)`、`PG-BiTemp (2M Warm RAM)`、`SeRF-Adapt`、`DSG-1D`、`Zep/Graphiti`、端到端 Write+Read Token U 型曲线与 3 跳密集重述风暴自适应 `k_fetch` 倍增 (`Exp 3/3B`)、双协议去偏审计 (`Exp 4`)、`Parquet-Zstd` / `SQLite-BTree` / 2025 后真实 SEC XBRL 披露及 `query_tiered_pit` 冷热跨层统一检索 (`Exp 5/5C`)，以及 Vitter 滞后卷积水库 + 全桶 `>20%` 倾斜守卫在线流式滚动 (`Exp 6A`)。"
        ),
        nbf.v4.new_code_cell(
            "# Step 1: 挂载 Google Drive 并将项目高速同步至本地 NVMe (/content/local_ssd)\n"
            "from google.colab import drive\n"
            "import os, shutil\n\n"
            "drive.mount('/content/drive')\n"
            "DRIVE_ROOT = '/content/drive/MyDrive/PiT-Mem_VLDB_Project'\n"
            "LOCAL_ROOT = '/content/local_ssd/pit-mem'\n"
            "os.makedirs('/content/local_ssd', exist_ok=True)\n"
            "if os.path.exists(LOCAL_ROOT):\n"
            "    shutil.rmtree(LOCAL_ROOT)\n"
            "shutil.copytree(DRIVE_ROOT, LOCAL_ROOT, ignore=shutil.ignore_patterns('*.so', 'build'))\n"
            "%cd /content/local_ssd/pit-mem\n"
            "!nvidia-smi\n"
            "!lscpu | head -n 18 && free -h"
        ),
        nbf.v4.new_code_cell(
            "# Step 2: 安装 PostgreSQL + pgvector v0.8.0、C++17/OpenMP 依赖并编译 pit_mem_core (AVX2 + FMA)\n"
            "!sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
            "build-essential cmake pybind11-dev postgresql postgresql-contrib postgresql-server-dev-all zstd texlive-latex-extra texlive-fonts-extra\n"
            "!sudo apt-get install -y -qq postgresql-16-pgvector || sudo apt-get install -y -qq postgresql-14-pgvector || true\n"
            "!pip install -q --upgrade pybind11 psycopg[binary] sentence-transformers transformers accelerate pyarrow pandas matplotlib scikit-learn\n"
            "!pip install --no-build-isolation ./cpp\n"
            "!PYTHONPATH=. python3 -m pit_mem.benchmarks.test_mw1_mw4_fixes\n"
            "import pit_mem_core as pm\n"
            "print('Compiled & verified pit_mem_core successfully! INF_TICK =', pm.INF_TICK)"
        ),
        nbf.v4.new_code_cell(
            "# Step 3: 运行 200 万规模全套基准实验（Exp 1 ~ Exp 6）\n"
            "!PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m pit_mem.benchmarks.run_all_experiments \\\n"
            "    --mode colab_pro \\\n"
            "    --work-dir /content/local_ssd/work_2m \\\n"
            "    --raw-seed-dir data/raw_seed \\\n"
            "    --out-dir results/colab_pro \\\n"
            "    --dim 768 \\\n"
            "    --llm-model Qwen/Qwen2.5-7B-Instruct"
        ),
        nbf.v4.new_code_cell(
            "# Step 4: 生成 PVLDB 出版级矢量 PDF 图表与 LaTeX 实测指标宏，并同步回 Google Drive\n"
            "!PYTHONPATH=. python3 -m pit_mem.plotting.generate_figures results/colab_pro paper/figures\n"
            "!PYTHONPATH=. python3 -m pit_mem.plotting.generate_latex_macros --results-dir results/colab_pro --out-tex-dir paper\n\n"
            "import shutil\n"
            "shutil.copytree('results/colab_pro', f'{DRIVE_ROOT}/results/colab_pro', dirs_exist_ok=True)\n"
            "shutil.copytree('paper/figures', f'{DRIVE_ROOT}/paper/figures', dirs_exist_ok=True)\n"
            "shutil.copy2('paper/generated_metrics.tex', f'{DRIVE_ROOT}/paper/generated_metrics.tex')\n"
            "shutil.copy2('paper/generated_tables.tex', f'{DRIVE_ROOT}/paper/generated_tables.tex')\n"
            "print('All Colab Pro results & vector figures synced back to Google Drive:', DRIVE_ROOT)"
        ),
    ]
    return nb


def make_sub_notebook(title: str, desc: str, cmd: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{desc}"),
        nbf.v4.new_code_cell(cmd),
    ]
    return nb


def build_all(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "00_Colab_Pro_Master_Pipeline.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(make_master_notebook(), f)

    nb1 = make_sub_notebook(
        "Module 01: 43.1 万条真实金融语料构建与 200 万向量 GPU 分片编码 (ALFRED + SEC EDGAR + HF)",
        "独立复现宏观多版本修正流、SEC `10-K/10-Q/8-K Item 4.02` 重述流、394,529 条去重金融披露抓取与 `BAAI/bge-base-en-v1.5` 768d GPU 分片编码。",
        "%cd /content/local_ssd/pit-mem\n"
        "!PYTHONPATH=. python3 -m pit_mem.data_pipeline.fetch_macro_pit\n"
        "!PYTHONPATH=. python3 -m pit_mem.data_pipeline.fetch_sec_edgar\n"
        "!PYTHONPATH=. python3 -m pit_mem.data_pipeline.fetch_fin_news\n"
        "!PYTHONPATH=. python3 -m pit_mem.data_pipeline.build_finpit_bench",
    )
    with open(os.path.join(out_dir, "01_Data_and_GPU_Embedding.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb1, f)

    nb2 = make_sub_notebook(
        "Module 02: 百万级系统索引、流式宏分段与 NVMe 持久化压测 (Exp 1, Exp 2, Exp 3, Exp 6)",
        "对比 `PiT-Mem`、`Grid-SQ8-Scan`、`ACORN-BiTemp (SIGMOD'24)`、`PG-BiTemp (Warm 16GB RAM)`、`Zep/Graphiti`、`HNSW-Post`、`SeRF-Adapt` 与 `DSG-1D`，以及流式因果宏分段滚动和 `mmap`+CRC32 WAL 恢复。",
        "%cd /content/local_ssd/pit-mem\n"
        "!PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m pit_mem.benchmarks.run_all_experiments --mode colab_pro --work-dir /content/local_ssd/work_2m --raw-seed-dir data/raw_seed --out-dir results/colab_pro",
    )
    with open(os.path.join(out_dir, "02_System_Benchmarks_Exp1_to_3.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb2, f)

    nb3 = make_sub_notebook(
        "Module 03: 端到端回测前视去偏审计与 RAP-Compact 混合查询/野生财报消融 (Exp 4 & Exp 5)",
        "在 Protocol A (2025–2026 真实 SEC XBRL 重述)、Protocol B (反事实信贷契约) 与真实野生 SEC 10-K/10-Q 财报上评测 LCR、Audit F1、Sharpe Inflation，以及相对 `Parquet-Zstd` 和 `SQLite-BTree` 的微秒级混合查询加速。",
        "%cd /content/local_ssd/pit-mem\n"
        "!PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -c \"\n"
        "from pit_mem.benchmarks.run_all_experiments import run_exp4_leakage_and_audit, run_exp5_rap_compact_ablation\n"
        "run_exp4_leakage_and_audit('data/raw_seed/protocol_a_post2025.json', 'data/raw_seed/protocol_b_counterfactual.json', 'results/colab_pro', use_gpu_embed=True, use_gpu_llm=True)\n"
        "run_exp5_rap_compact_ablation('data/raw_seed/protocol_b_counterfactual.json', 'data/raw_seed/protocol_a_post2025.json', 'results/colab_pro', use_gpu_embed=True, use_gpu_llm=True)\n\"",
    )
    with open(os.path.join(out_dir, "03_Agent_Evaluation_Exp4_to_5.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb3, f)

    print(f"[Notebooks] Generated 4 Colab Pro notebooks -> {out_dir}")


if __name__ == "__main__":
    build_all("notebooks")
