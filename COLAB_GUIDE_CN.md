# Google Colab Pro 实验复现与防踩坑实操手册（COLAB_GUIDE_CN）

## 1. 快速复现步骤（在浏览器打开 Colab Pro）

1. 打开 Google Drive，进入文件夹 `MyDrive/PiT-Mem_VLDB_Project/notebooks/`。
2. 双击打开 `00_Colab_Pro_Master_Pipeline.ipynb`（或按阶段分步打开 `01` ~ `03`）。
3. 在 Colab 顶部菜单选择 **Runtime -> Change runtime type**：
   - **Hardware accelerator**：推荐 `A100 GPU`（80 GB / 40 GB VRAM）
   - **Runtime shape**：开启 `High-RAM`（A100 High-RAM 提供 12 vCPU 与最高 167 GiB 系统内存，支持 PostgreSQL `shared_buffers = 16GB` 全内存热缓冲对比至 200 万向量）
4. 点击 **Runtime -> Run all** 即可一键完成：
   - 将云盘代码与真实种子数据同步至容器本地高速 NVMe 盘 `/content/local_ssd/pit-mem`；
   - 编译 C++17 `pit_mem_core`（`-O3 -std=c++17 -mavx2 -mfma -fopenmp`）并启动本地 `PostgreSQL + pgvector v0.8.0`；
   - 自动校验 394,529 条去重金融披露语料（合计 **431,217 条真实唯一金融源文档**）与同血缘版本余弦流形，在 GPU 上完成 200 万条双时态版本 `BAAI/bge-base-en-v1.5` (768d FP16) 分片编码与 `Qwen/Qwen2.5-7B-Instruct` 审计推理；
   - 自动跑完 **Exp 1（Workload 1A 12% 凹函数多段束宽分配因果图 + Workload 1B 1% 零倒挂解析代价路由，最高 2M 向量）~ Exp 6（Exp 6A Vitter 流式滚动消除倾斜、Exp 6B NVMe `mmap`/CRC32 WAL 恢复、Exp 6C `1,032 B/rec` 冷态 FP32 NVMe `mmap` 卸载与冷页缺失压测、Exp 6D `PhaseFairSharedMutex` 12 线程并发读 + 持续流式写入/重述/宏段滚动混合压测）**、重新生成全部 7 张 PVLDB Best Paper 范式矢量 PDF 图表及 `paper/generated_metrics.tex` / `paper/generated_tables.tex`，并将结果同步回 Google Drive。

---

## 2. 四大 Colab 工程防踩坑机制说明

1. **杜绝 Google Drive FUSE 随机 I/O 干扰微秒/亚毫秒级延迟评测**
   - 严禁直接在 `/content/drive/MyDrive` 挂载盘内构建索引或测量微秒级延迟。主控脚本自动将全部工作目录、PostgreSQL 数据簇与 POSIX `mmap`/WAL 持久化及冷态 FP32 卸载压测隔离至本地 `/content/local_ssd/`，仅在实验完成后归档同步 CSV 与 PDF。
2. **公平全内存热缓存对比配置（PostgreSQL `pgvector v0.8.0`）**
   - `pit_mem/baselines/pg_bitemp.py` 在 High-RAM 实例上自动启用 `shared_buffers = '16GB'`、`work_mem = '256MB'`、`maintenance_work_mem = '4GB'`，建索引后强制执行 `VACUUM (ANALYZE)`、`CHECKPOINT` 与预热查询，并使用服务端二进制 Prepared Statements 与选择度自适应执行计划，公平对比至 200 万向量规模。
3. **分片幂等断点续跑（Sharded Idempotent Checkpointing）**
   - 200 万条流式分片生成与 GPU 向量编码均采用临时写入 + 原子落盘机制。若会话中断，重新运行会自动跳过已完成的 `vecs_shard_*.npy` 分片，并支持 `--only-exps` 与 `--reuse-baselines-dir` 精准增量重跑。
4. **止损门禁自动审计（Stop-Loss Gate Auditor）**
   - 每次实验结束后自动输出 `results/colab_pro/stop_loss_audit_report.json`，严格核验 Workload 1A $\ge 90\%$ 高召回与因果图加速、Workload 1B 相对 `Grid-SQ8-Scan` 零路由倒挂、单线程真实墙钟 QPS 与延迟一致性、双协议零前视污染（`LCR = 0.00%`）、端到端 Write+Read Token 节省、RAP-Compact 与跨层 `query_tiered_pit` 保真度，以及流式活跃尾段倾斜 $\le 15\%$ 门禁。
