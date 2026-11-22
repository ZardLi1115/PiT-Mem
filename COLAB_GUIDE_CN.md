# Google Colab Pro 实验复现与工程操作手册（COLAB_GUIDE_CN）

## 1. 快速复现步骤（在浏览器打开 Colab Pro）

1. 打开 Google Drive，进入文件夹 `MyDrive/PiT-Mem_VLDB_Project/notebooks/`。
2. 双击打开 `00_Colab_Pro_Master_Pipeline.ipynb`（或按阶段分步打开 `01` ~ `03`）。
3. 在 Colab 顶部菜单选择 **Runtime -> Change runtime type**：
   - **Hardware accelerator**：推荐 `A100 GPU`（40 GB / 80 GB VRAM，提供 12 vCPU 与 `83 GiB+` 系统内存，支持 PostgreSQL `shared_buffers = 12GB` 缓冲对比至 200 万向量）。
4. 点击 **Runtime -> Run all** 即可完成：
   - 将云盘代码与真实种子数据同步至容器本地高速 NVMe 盘 `/content/local_ssd/pit-mem`；
   - 编译 C++17 `pit_mem_core`（`-O3 -std=c++17 -mavx2 -mfma -fopenmp`）、运行 `test_mw1_mw4_fixes` 单元与并发回归测试，并启动本地 `PostgreSQL 16 + pgvector v0.8.0`；
   - 基于 **431,217 条公开去重金融种子语料**在 GPU 上完成 200 万条双时态版本 `BAAI/bge-base-en-v1.5` (768d) 编码与 `Qwen/Qwen2.5-7B-Instruct` 审计推理；
   - 运行 **Exp 1 ~ Exp 6D** 全套归档基准实验，重新生成全部 7 张矢量 PDF 图表及 `paper/generated_metrics.tex` / `paper/generated_tables.tex`，并编译符合官方 `vldbproceedings/VLDB-Template` 的 `paper/PiT-Mem_PVLDB.pdf`。

---

## 2. 四大 Colab 工程防踩坑说明

1. **避免 Google Drive FUSE 随机 I/O 干扰微秒/亚毫秒级延迟评测**
   - 严禁直接在 `/content/drive/MyDrive` 挂载盘内构建索引或测量延迟。主控脚本自动将全部工作目录、PostgreSQL 数据簇与 POSIX `mmap`/WAL 持久化隔离至本地 `/content/local_ssd/`，仅在实验完成后同步 CSV 与 PDF。
2. **PostgreSQL 内存与索引配置**
   - `pit_mem/baselines/pg_bitemp.py` 自动启用 `shared_buffers = '12GB'`、`work_mem = '256MB'`、`halfvec(768)` HNSW（`m=16`）与复合覆盖 B-Tree `(tk_s, tv_s) INCLUDE (tk_e, tv_e)`，建表后执行 `VACUUM (ANALYZE)` 与 `CHECKPOINT`。
3. **分片幂等编码**
   - 200 万条 GPU 向量编码采用分片落盘机制，中断后可断点续跑。
4. **基线范围与单元测试验证**
   - 归档结果中 Exp 1–2 的图基线为同硬件 C++17 双时态适配实现，Exp 4 为三种架构策略代理；仓库同时包含审计后加固的 C++17 校验与 vendored `nmslib/hnswlib` v0.8.0 绑定，可通过 `python3 -m pit_mem.benchmarks.test_mw1_mw4_fixes` 验证。
