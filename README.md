# PiT-Mem: A Point-in-Time Bi-Temporal Storage and Indexing Engine for Look-Ahead-Free Agentic Memory

Official artifact repository (source code, **FinPiT-Bench** datasets, benchmark scripts, Colab Pro notebooks, verified experimental CSVs, and PVLDB paper sources) for **PiT-Mem** (*Proceedings of the VLDB Endowment, Vol. 20*).

- **GitHub Repository**: https://github.com/ZardLi1115/PiT-Mem
- **Complete Data & Artifact Mirror (Google Drive, Public Read)**: https://drive.google.com/drive/folders/14uNPJJ5iPavmf9UI4cL-SdeZZuzDWJmu
- **Chinese Documentation**: See `README_CN.md` and `COLAB_GUIDE_CN.md`.

---

## 1. Repository Structure

```text
pit-mem/
├── README.md                            # English PVLDB Vol. 20 Artifact & Reproduction Guide
├── README_CN.md                         # Chinese architecture & experimental summary
├── COLAB_GUIDE_CN.md                    # Google Colab Pro A100 reproduction guide
├── cpp/                                 # C++17 + AVX2 + OpenMP + POSIX mmap + CRC-32 WAL engine (~6,220 LOC + 1,169 LOC baselines)
│   ├── include/
│   │   ├── pit_types.hpp                # 64-bit bi-temporal 7-tuple model, PhaseFairSharedMutex, 64B-aligned allocators
│   │   ├── simd_kernels.hpp             # Unsigned 8-wide AVX2 stabbing filters, biased 7x8b maddubs SQ8, 4-way FMA FP32 kernels
│   │   ├── bit_hnsw.hpp                 # Streaming Causal Macro-Segment BiT-HNSW, Vitter lag-convolution reservoir, analytical cost router, mmap offload & WAL
│   │   ├── mvcc_provenance.hpp          # SQL:2011 EVOLVE/RESTATE, PVWL transactional WAL, PITMANF1 atomic snapshots, adaptive k_fetch Lazy-Branch MVCC
│   │   ├── rap_compact.hpp              # RAP-Compact ("PIT3"): unit-hypersphere SQ8 centroids, 56B IEEE-754 anchor slots, unified hot+cold tiered search
│   │   └── baselines_cpp.hpp            # In-house C++17 bi-temporal adaptations (Grid-SQ8-Scan, ACORN-Style, HNSW-Flat, SeRF-Style, DSG-Style) + vendored nmslib/hnswlib v0.8.0 bindings
│   ├── src/bindings.cpp                 # Zero-copy pybind11 NumPy bindings with GIL release
│   └── setup.py / CMakeLists.txt        # C++17 build configuration (-O3 -std=c++17 -mavx2 -mfma -fopenmp)
├── pit_mem/                             # Python data pipeline, external DBMS baselines, benchmark runner, and plotting
│   ├── covenant_parser.py               # Schema-parameterized multi-span covenant & SEC XBRL slot extractor
│   ├── data_pipeline/                   # FinPiT-Bench builders (St. Louis Fed ALFRED, post-2025 SEC EDGAR XBRL/8-K Item 4.02, financial news) & BGE embedding
│   ├── baselines/                       # Tuned PostgreSQL + pgvector v0.8 (PG-BiTemp) and Exp 4 surrogate memory policies (Latest-Summary, Overwrite, Append-Only)
│   ├── benchmarks/                      # End-to-end benchmark suite (Exp 1–Exp 6D) and unit/concurrency regression tests
│   └── plotting/                        # Publication vector PDF figure generator & automated CSV-to-LaTeX macro/table generator
├── data/
│   └── raw_seed/                        # 431,217 deduplicated public financial seed passages + Protocols A & B audit sets
├── results/
│   ├── colab_pro/                       # Archived N = 2,000,000 NVIDIA A100-SXM4-80GB (12 vCPU, 167 GiB RAM, NVMe SSD) CSVs & logs
│   └── local_pilot/                     # Fast local CPU regression CSVs
├── notebooks/                           # Interactive Google Colab Pro reproduction notebooks (00–03)
└── paper/                               # Official PVLDB Vol. 20 (VLDB-Template) LaTeX sources, generated macros/tables, figures, and PiT-Mem_PVLDB.pdf
```

---

## 2. Hardware & Software Prerequisites

- **CPU & OS**: Linux x86\_64 with **AVX2** and **FMA** instruction sets (`g++ >= 11` with C++17 and OpenMP).
  - *Paper benchmark hardware*: Google Colab Pro instance with 12 Intel Xeon vCPUs (AVX2, FMA), 167 GiB RAM, local NVMe SSD (`/content/local_ssd`), and 1x NVIDIA A100-SXM4-80GB GPU.
- **Python**: Python `>= 3.10` with `numpy`, `pandas`, `pyarrow`, `pybind11`, `matplotlib`, `scipy`, `requests`.
  - Optional for dense corpus encoding (`BAAI/bge-base-en-v1.5`) & LLM audit (`Qwen/Qwen2.5-7B-Instruct`): `torch`, `sentence-transformers`, `transformers`, `accelerate`.
  - Optional for `PG-BiTemp` relational+vector baseline: PostgreSQL `>= 16` + `pgvector >= 0.8.0` + `psycopg[binary]`.
- **LaTeX** (optional, to recompile `paper/PiT-Mem_PVLDB.pdf`): TeX Live (`pdflatex`, `bibtex`).

---

## 3. Quick Start: Build C++17 Engine & Run Verification Tests (~30 Seconds)

```bash
# 1. Clone the repository and create a virtual environment
git clone https://github.com/ZardLi1115/PiT-Mem.git
cd PiT-Mem
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pybind11 numpy pandas pyarrow scipy matplotlib requests

# 2. Compile and install the C++17 AVX2/OpenMP engine (pit_mem_core)
pip install --no-build-isolation ./cpp

# 3. Run unit, crash-recovery WAL, cold-tier retraction, router auto-calibration, and 12-thread concurrency tests
PYTHONPATH=. python3 -m pit_mem.benchmarks.test_mw1_mw4_fixes
```

---

## 4. Dataset Preparation (`FinPiT-Bench`)

All **431,217** authentic unique financial seed passages (`alfred_macro_pit.parquet`, `sec_edgar_pit.parquet`, `hf_fin_news_texts.parquet`, `protocol_a_post2025.json`, `protocol_b_counterfactual.json`) are included under `data/raw_seed/` in the public Google Drive artifact folder:

- Download `data/raw_seed/` from https://drive.google.com/drive/folders/14uNPJJ5iPavmf9UI4cL-SdeZZuzDWJmu (or extract `data/finpit_bench_seed.tar.zst` if present locally) into `data/raw_seed/`.

---

## 5. Reproducing Experiments (`Exp 1` – `Exp 6D`)

### Option A: Fast Local CPU Pilot Verification (~3 Minutes, No GPU Required)
Runs all six experimental pipelines (`Exp 1`–`Exp 6D`) at pilot scale to verify functional correctness and output CSV schemas in `results/local_pilot/`:

```bash
PYTHONPATH=. python3 -m pit_mem.benchmarks.run_all_experiments \
  --mode local_pilot \
  --work-dir ./work_pilot \
  --raw-seed-dir data/raw_seed \
  --out-dir results/local_pilot \
  --no-postgres
```

### Option B: Full Million-Scale Reproduction ($N = 2{,}000{,}000$, Paper Setting)
On an A100 + local NVMe SSD machine (or via `notebooks/00_Colab_Pro_Master_Pipeline.ipynb` on Google Colab Pro High-RAM A100):

```bash
pip install torch sentence-transformers transformers accelerate "psycopg[binary]"

PYTHONPATH=. python3 -m pit_mem.benchmarks.run_all_experiments \
  --mode colab_pro \
  --work-dir /content/local_ssd/work_2m \
  --raw-seed-dir data/raw_seed \
  --out-dir results/colab_pro
```

You can also run any individual experiment slice via `--only-exps` (e.g., `--only-exps 1,2`, `--only-exps 6c`).

---

## 6. Regenerating All Paper Figures, LaTeX Macros/Tables & PDF

Every metric macro (`paper/generated_metrics.tex`), table (`paper/generated_tables.tex`), and figure (`paper/figures/*.pdf`) in the paper is generated directly from `results/colab_pro/*.csv`:

```bash
# Regenerate vector figures (Figures 1–7) and LaTeX metric/table macros (Tables 3–7)
PYTHONPATH=. python3 -m pit_mem.plotting.generate_figures results/colab_pro paper/figures
PYTHONPATH=. python3 -m pit_mem.plotting.generate_latex_macros --results-dir results/colab_pro --out-tex-dir paper

# Compile official single-blind PVLDB Vol. 20 PDF (paper/PiT-Mem_PVLDB.pdf)
cd paper && make
```
