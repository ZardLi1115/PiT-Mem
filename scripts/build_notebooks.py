"""Generate clean, turnkey Colab Pro Jupyter Notebooks (.ipynb) using nbformat v4."""

from __future__ import annotations

import os
import nbformat as nbf


def make_master_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# PiT-Mem (PVLDB): 双时态金融 Agent 记忆存储与索引引擎 — Colab Pro 一键复现主控台\n\n"
            "本 Notebook 对应论文 **PiT-Mem: A Point-in-Time Bi-Temporal Storage and Indexing Engine for Look-Ahead-Free Agentic Memory** 的完整单卡 A100 实验流水线（`Exp 1 ~ Exp 6D`）。\n\n"
            "### 核心架构与实验覆盖：\n"
            "1. **本地 NVMe 冷热分层与持久化评测 (`Exp 6A–6D`)**：`1,032 B/rec` 计数核心索引数组（`1.92 GiB @ 2M`）+ `3,072 B/rec` 原始 FP32 向量 POSIX `mmap` 按需分页卸载、清单级原子 CSR 快照与 IEEE 802.3 CRC-32 WAL、`PhaseFairSharedMutex` 12 线程混合并发压测，以及 `PostgreSQL 16 + pgvector v0.8.0` (`shared_buffers = 12GB`) 均运行在 `/content/local_ssd/` 本地 NVMe 上。\n"
            "2. **431,217 条公开去重金融种子语料 + GPU 编码**：基于 ALFRED 宏观多版本流、SEC EDGAR `10-K/10-Q/8-K Item 4.02` 重述流与 394,529 条去重金融披露文本，扩展至 200 万条 `BAAI/bge-base-en-v1.5` (768d) 双时态向量。\n"
            "3. **端到端系统与智能体实验 (`Exp 1 ~ Exp 6D`)**：涵盖多量级双工况吞吐/召回（`Exp 1`）、跨选择度路由切换（`Exp 2`）、重述元数据吞吐与校准 Token 代价模型及 3 跳突发实测（`Exp 3/3B`）、`Qwen/Qwen2.5-7B-Instruct` 双协议前视泄漏与审计 F1（`Exp 4`）、`RAP-Compact` 槽位保真与跨热冷层统一检索（`Exp 5/5C`）以及流式宏段偏斜消除与并发持久化（`Exp 6A–6D`）。"
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
            "# Step 2: 安装 PostgreSQL + pgvector v0.8.0 与依赖，并编译验证 pit_mem_core\n"
            "!sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
            "build-essential cmake pybind11-dev postgresql postgresql-contrib postgresql-server-dev-all zstd texlive-latex-extra texlive-fonts-extra\n"
            "!sudo apt-get install -y -qq postgresql-16-pgvector || sudo apt-get install -y -qq postgresql-14-pgvector || true\n"
            "!pip install -q --upgrade pybind11 \"psycopg[binary]\" sentence-transformers transformers accelerate pyarrow pandas matplotlib scipy\n"
            "!pip install --no-build-isolation ./cpp\n"
            "!PYTHONPATH=. python3 -m pit_mem.benchmarks.test_mw1_mw4_fixes\n"
            "import pit_mem_core as pm\n"
            "print('Compiled & verified pit_mem_core successfully! INF_TICK =', pm.INF_TICK)"
        ),
        nbf.v4.new_code_cell(
            "# Step 3: 运行 200 万规模归档基准实验（Exp 1 ~ Exp 6D）\n"
            "!PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m pit_mem.benchmarks.run_all_experiments \\\n"
            "    --mode colab_pro \\\n"
            "    --work-dir /content/local_ssd/work_2m \\\n"
            "    --raw-seed-dir data/raw_seed \\\n"
            "    --out-dir results/colab_pro \\\n"
            "    --dim 768 \\\n"
            "    --llm-model Qwen/Qwen2.5-7B-Instruct"
        ),
        nbf.v4.new_code_cell(
            "# Step 4: 生成 PVLDB 出版级矢量 PDF 图表、LaTeX 指标宏/表格与论文 PDF，并同步回 Google Drive\n"
            "!PYTHONPATH=. python3 -m pit_mem.plotting.generate_figures results/colab_pro paper/figures\n"
            "!PYTHONPATH=. python3 -m pit_mem.plotting.generate_latex_macros --results-dir results/colab_pro --out-tex-dir paper\n"
            "!cd paper && make\n\n"
            "import shutil\n"
            "shutil.copytree('results/colab_pro', f'{DRIVE_ROOT}/results/colab_pro', dirs_exist_ok=True)\n"
            "shutil.copytree('paper/figures', f'{DRIVE_ROOT}/paper/figures', dirs_exist_ok=True)\n"
            "shutil.copy2('paper/generated_metrics.tex', f'{DRIVE_ROOT}/paper/generated_metrics.tex')\n"
            "shutil.copy2('paper/generated_tables.tex', f'{DRIVE_ROOT}/paper/generated_tables.tex')\n"
            "shutil.copy2('paper/PiT-Mem_PVLDB.pdf', f'{DRIVE_ROOT}/paper/PiT-Mem_PVLDB.pdf')\n"
            "print('All Colab Pro results, vector figures & PDF synced back to Google Drive:', DRIVE_ROOT)"
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
        "Module 01: 43.1 万条公开金融种子语料构建与 200 万向量 GPU 编码 (ALFRED + SEC EDGAR + HF)",
        "独立复现宏观多版本修正流、SEC `10-K/10-Q/8-K Item 4.02` 重述流、去重金融披露抓取与 `BAAI/bge-base-en-v1.5` 768d GPU 分片编码。",
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
        "运行 `PiT-Mem`、`Grid-SQ8-Scan`、`PG-BiTemp` (`pgvector v0.8.0`) 及 C++17 双时态图适配基线（`ACORN-Style`, `HNSW-Flat`, `SeRF-Style`, `DSG-Style`），以及流式宏段滚动、NVMe 快照/WAL、冷 FP32 `mmap` 卸载与 12 线程并发压测。",
        "%cd /content/local_ssd/pit-mem\n"
        "!PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m pit_mem.benchmarks.run_all_experiments --mode colab_pro --work-dir /content/local_ssd/work_2m --raw-seed-dir data/raw_seed --out-dir results/colab_pro --only-exps 1,2,3,6",
    )
    with open(os.path.join(out_dir, "02_System_Benchmarks_Exp1_to_3.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb2, f)

    nb3 = make_sub_notebook(
        "Module 03: 端到端时点前视泄漏审计与 RAP-Compact 槽位保真评测 (Exp 4 & Exp 5)",
        "在 Protocol A (2025–2026 SEC XBRL 重述) 与 Protocol B (脱敏信贷契约模板) 上使用 `Qwen/Qwen2.5-7B-Instruct` 评测 LCR、Stale Taint 与 Grounded Audit F1，并评测 `RAP-Compact` 与 `SQLite-BTree` / `Parquet-Zstd` 及跨层 `query_tiered_pit`。",
        "%cd /content/local_ssd/pit-mem\n"
        "!PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m pit_mem.benchmarks.run_all_experiments --mode colab_pro --work-dir /content/local_ssd/work_2m --raw-seed-dir data/raw_seed --out-dir results/colab_pro --only-exps 4,5",
    )
    with open(os.path.join(out_dir, "03_Agent_Evaluation_Exp4_to_5.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb3, f)

    print(f"[Notebooks] Generated 4 Colab Pro notebooks -> {out_dir}")


if __name__ == "__main__":
    build_all("notebooks")
