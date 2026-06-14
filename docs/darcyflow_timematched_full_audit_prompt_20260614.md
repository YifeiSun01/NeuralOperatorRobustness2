# DarcyFlow Time-Matched Full Audit Prompt - 2026-06-14

Use this prompt to port the completed Burgers time-matched audit workflow to
DarcyFlow/data flow. This is intentionally written as an executable instruction
prompt for a future agent. It is not a Burgers rerun request.

```text
在 /workspace/NeuralOperatorRobustness2 里处理 DarcyFlow / data flow 的
time-matched 训练、结果审计、可视化、统计总表、organized release 和 R2/GitHub 同步。
使用虚拟环境 /workspace/adv_robust 或 repo 内 adv_robust。先读 AGENTS.md 和
/etc/vast_agents/*.md。

关键判断：
1. 这是 DarcyFlow/data flow，不是 Burgers。DarcyFlow 比 Burgers 多 physics loss；
   所有训练、日志、统计表、结论里都要记录 data loss / prediction metrics 以及
   physics loss / PDE residual 相关指标。
2. 不要默认重跑。先审计本地和 R2 上已有 DarcyFlow/data flow 结果，包括模型、
   52 数据集 evaluation、robustness、SVD/Jacobian、dense 图、统计表、markdown、
   organized release、R2 远端目录。
3. 如果已有模型、52 数据集 evaluation、robustness、SVD/Jacobian、图和统计量
   已够用，就不要重跑 self-training 或昂贵 SVD，只做整理、补图、补 markdown、
   补统一 CSV/JSON/MD 总表。
4. 如果某些昂贵项在本地/R2 都没找到，先写 coverage note 说明“未找到/部分覆盖/
   不补算”，不要为了补齐 top50/top100 SVD 或完整 affine/local-gain sweep 而
   默认开重跑，除非用户明确要求。

模型设置：
- 模型组：
  - baseline
  - loss1
  - loss2
  - loss3
  - random clean
  - random solver
- DarcyFlow 使用 physics loss。训练和 evaluation 里要保存：
  - RMSE
  - Relative L2
  - MSE 或 data loss
  - physics loss / PDE residual
  - 如果代码已有其他 DarcyFlow 可用指标，也一起记录，但不要把不必要的图塞进主图。
- 以 loss3 的目标 work-clock 为基准。DarcyFlow 大约按 loss3 3000-3500 epoch
  作为目标范围，先用已有 logs 或新跑 10-30 epoch timing calibration 估计：
  - loss3 目标 work-clock 对应 epoch 数
  - loss1/loss2/random clean/random solver 在相同 work-clock 下对应 epoch 数
- work-clock 只包含真正训练工作：
  attack/delta generation、random generation、solver target generation、physics
  residual/loss computation、forward/backward、optimizer step。
  不包含 evaluation、plot、postprocess、upload。
- 图上的 wall-clock/work-clock 横轴要画到目标时间附近，例如 8 小时左右；不要因为
  random clean 只有两三个小时就把所有图截到最短曲线。短曲线可以提前结束，右端
  仍保留目标时间。

如果需要补跑：
1. 先 smoke test 每个方法，小 epoch 跑通训练、52 数据集 evaluation、checkpoint、
   final analysis、plot。
2. checkpoint 必须保存并恢复 model + optimizer state，避免续训 loss 曲线突变。
3. 正式跑后跟踪日志约 10 分钟，确认无错后停止主动跟踪，让任务继续。
4. 所有正式训练/矩阵计算必须走 GPU；如果 CUDA/PyTorch/JAX GPU 路径不正常，先修
   环境，不要静默 CPU fallback。

记录与分析：
1. 每个 epoch 对 train/test/50 generalization 共 52 数据集记录：
   RMSE、Relative L2、MSE/data loss、physics loss/PDE residual、以及已有可用指标。
2. 训练完成后保存最终模型，并完整记录 52 数据集最终 RMSE、Relative L2、
   MSE/data loss、physics loss/PDE residual。
3. 对最终模型做 robustness：
   - 每个 52 数据集固定 50 样本做 attack。
   - 记录 clean loss、adv loss、loss increase、relative increase、delta RMS/L2。
   - DarcyFlow 还要记录 attack 前后 physics loss/PDE residual 的变化。
4. SVD/Jacobian 只对固定 25 样本：
   train 2、test 2、generalization 21 个数据集各 1 个。
   50-sample attack manifest 必须包含这 25 个样本。
5. 计算并保存：
   - singular values / top-k singular spectrum
   - J^T error norm / RMS
   - bias-gradient norm / RMS
   - clean residual norm / MSE
   - attack loss increase
   - physics residual 与 attack/clean/error/SVD 指标的关系
   - 同样本 Pearson/Spearman correlation
   - singular vector、J^T error vector、attack delta 的 cosine similarity 和 angle
   - model-solver subspace similarity：top1/top5/top10/top20；如果已有 top50/top100
     就记录，没找到就写 coverage note，不默认重跑
6. 统一生成很多 CSV 和 Markdown 表，而不是只写结论：
   - 每个 metric/model/scope 的 mean、std、median、n、rank、is_best、best_model、
     runner_up_model、runner-up gap
   - best-vs-other paired t-test、Wilcoxon、one-sided better p-value、two-sided p-value、
     BH-FDR q-value
   - loss3-vs-other 的同样统计检验
   - per-dataset/per-sample best 表
   - correlation ranking 表
   - coverage note 表：哪些完整、哪些部分完整、哪些没找到但不补
7. Markdown 总报告要把每个指标的最优模型加粗，并用自然语言说明：
   - generalization 结论
   - robustness 结论
   - physics loss/PDE residual 结论
   - SVD/Jacobian 结论
   - correlation/similarity 结论
   - loss3 是否显著优于其他模型，或者哪个模型实际最优
   - 哪些指标 coverage 不完整，以及为什么不补算

可视化要求：
1. 主训练曲线只画 RMSE 和 Relative L2；不要画 grad norm、optimizer internal loss、
   额外柱状图。DarcyFlow 的 physics loss 要记录进数据表/报告；如需画 physics loss，
   单独放 diagnostic 子目录，不混进主曲线 bundle。
2. 每个主指标都画：
   - 横轴 epoch
   - 横轴 work-clock / wall-clock time，右端保留目标时间附近，例如 8 小时左右
   - train/test/generalization mean 合图
   - 50 个 generalization 分两张图，每张 25 个
3. 所有 loss/metric 曲线都加 baseline model 的水平线。
4. 不做 moving average、rolling mean、插值或 smoothing；画原始 evaluation points。
5. 数据集标题不要写 DD15/DD16/DD20 这种编号，要用 descriptive name：
   例如 kernel/type/range/parameter/seed/family 这类可读名字。DarcyFlow 就用它的
   permeability/coefficient/source/BC/generalization 参数描述。
6. legend 要清楚、字体足够大、线条加透明度，避免曲线互相遮住。
7. 至少生成这些版本：
   - all six models
   - no random clean
   - loss123 only，即去掉 baseline、random clean、random solver
   - linear-y
   - log-y
8. dense image-only 大图也要做：
   - group00 到 group04 用固定/代表样本
   - group05 专门挑 loss3 优势最明显的样本，即 loss3 attack 后 loss/loss increase
     明显低于其他模型
   - 每个 group 生成 all-model / no-random-clean / loss123-only，以及 linear/log attack
     progression 版本
   - DarcyFlow 图面板不要照搬 Burgers 的 initial condition 命名，要用 DarcyFlow 语义：
     attack delta、input coefficient/permeability、solver target/output、model output、
     model-solver error、physics residual/error map（如果已有）
   - 底部画 attack loss progression；如果有 physics loss progression，可单独附加或
     放 diagnostic 图，不遮挡主图

输出目录：
- 单独醒目的 final audit 文件夹，例如：
  outputs/darcyflow_timematched_full_or_audit_<date>/
- 子目录必须包括：
  figures/
  data/
  reports/
  logs/
  manifests/
- 再生成一个 organized release 文件夹，例如：
  outputs/darcyflow_timematched_organized_release_<date>/
- 所有东西都要装进这些文件夹：
  - data 里放 clean/attack/robustness/SVD/correlation/ranked/coverage CSV/JSON
  - figures 里放训练曲线、log-y/no-random-clean/loss123-only、dense group00..group05 图
  - dense trace/csv/json/npz 也要在 release 里分层整理
  - reports 里放完整 Markdown 结论
  - manifests 里放文件清单、样本清单、R2 sync/coverage manifest

统计总表要求：
1. 生成机器可读 CSV：
   - metric_model_summary_ranked.csv
   - metric_best_summary_ranked.csv
   - metric_best_vs_other_significance_tests.csv
   - metric_loss3_vs_other_significance_tests.csv
   - clean_52dataset_metric_long_ranked.csv
   - attack_52dataset_metric_long_ranked.csv
   - robustness_25sample_metric_long_ranked.csv
   - svd_error_topk/top20 ranked tables
   - model-level scalar ranked tables
   - correlations sorted tables
   - random/coverage partial metric notes
2. 生成完整 Markdown appendix：
   - 很多个表，不是一个表
   - 每个指标标明最优模型，加粗显示
   - 写 mean、std、n、runner-up、优势量
   - 写 paired t-test / Wilcoxon / FDR 显著性
   - 写 loss3 vs 其他模型是否真的显著
3. CSV 里用字段记录：
   rank、is_best、best_model、runner_up_model、advantage_vs_runner_up、
   p-value、q-value、n_pairs、coverage_status。

上传与提交：
- 代码、sh、Markdown 提交 GitHub branch: vast-ai。
- 大结果和图上传 Cloudflare R2 bucket prefix:
  neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected
- 不要把 token/secret 写入代码、Markdown、shell 或 commit。
- 使用环境变量 R2_ACCESS_KEY_ID、R2_SECRET_ACCESS_KEY、R2_ENDPOINT、GITHUB_TOKEN。
- R2 sync 后用 rclone size --json 验证 object count/bytes，并把 verification 写进
  report/ledger。

完成前检查：
1. final audit 里 data/ranked tables 是否完整。
2. organized release 里是否也复制了 ranked tables。
3. figures 里 group00..group05 是否每组都有所有 required variants。
4. dense trace/csv/json/npz 是否也在 release 里，不只是 PNG。
5. Markdown 报告是否明确写了 generalization、robustness、physics loss、
   SVD/Jacobian、correlation/similarity、coverage gaps。
6. EXPERIMENT_LEDGER.md 是否更新。
7. git status 只 staged/committed 代码、sh、Markdown；大结果留在 R2。
8. push 到 vast-ai。
```

