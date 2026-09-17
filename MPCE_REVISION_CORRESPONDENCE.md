# MPCE 投稿前审查意见 → 改稿一一对应表

审查报告：《MPCE 投稿前全面审查与拒稿风险评估》(2026-09-16)，固定审计提交 `92ed627`。
改稿日期：2026-09-16。本文档逐条记录每项审查意见对应的修改位置与改后内容。

数值来源：`recompute_all.py` / `recompute_extras.py` / `recompute_cct2.py` / `variants_acc.py` /
`decidable_vs_alpha.py` / `n12_rmse.py`（全部按修正协议重算；论文数值宏与之一一对应）。

---

## 一、核心实现错误（P0，已修复并重算）

| 编号 | 问题 | 修复位置 | 结果（旧值 → 新值） |
|---|---|---|---|
| **A1** 成对分数丢失误差符号 | 代码用 `err=abs(pred-y)` 再求差 | `conformal_tighter.py`、`recompute_all.py`：保留 `signed_err`，$R_{\mathrm{pw}}=\max_{i,j,t}\|e_i-e_j\|$ | 论文 Sec.4.4 公式注明符号；q_pw **102°→148.1°**（α=0.1），coverage 0.893 (159/178)。摘要/结论/表2 同步改 |
| **A2** 末时刻判据 | 分类与 CCT 用 `tr[:,-1]` | `evaluate.py`、`train.py`、`compute_round8.py`、`make_figures.py` 等全部改为 max-over-time spread（与参考标签一致） | 准确率 **96.7%→95.4%**（TP/FP/FN/TN=318/6/38/598）；CCT MAE **12.3→25.2 ms**（max 204.9 ms，bus 12）；新增逐母线 CCT 表（Table 6）。one-axis 97.1→94.0、damped 94.8→93.4、118 95.0→92.9、N-1 各分拆也重算 |
| **A3** 门控覆盖率混入校准样本 | 门控评估用全集合 | `compute_round8.py`、`recompute_all.py`：只在不相交评估半上报告；分列全门控覆盖、oracle 子集覆盖、错误放行率、样本数 | 论文 Sec.5.6：gate purity 96.6%、cov(pred-stable) 0.914 (n=58)、oracle 0.946 (n=56)、wrong-release 1.8% (2/109)，并标注 oracle 诊断为"非部署保证" |
| **A4** 近似分位数 | `np.quantile((1-α)(1+1/n))` 线性插值 | 全部 conformal 脚本改为次序统计量 $k=\lceil(n_{cal}+1)(1-α)\rceil$（`order_quantile`） | 论文 Sec.4.4 明确写出次序统计量规则；q̂ **111→110.1°**，coverage **0.893→0.882** (157/178, CI 0.83-0.93) |
| **A5** 分组/首摆窗口依赖真值 | 分组与窗口用真实轨迹定义 | 论文 Sec.5.6 明确标注为 **offline diagnostic**；峰值定义固定为故障后 pairwise spread 峰值 $\max_{t\ge t_c}(\max_i\tilde\delta_i-\min_i\tilde\delta_i)$（`conformal_analysis.py` 重写） | 分组改为 3 组（<45°/45-90°/>90°），带宽 76.1/94.2/124.6°，逐组覆盖率 0.863/0.923/0.800，图上标注样本量 |

## 二、覆盖率与校准协议

| 编号 | 修改 |
|---|---|
| **B1** 真实稳定条件下的覆盖 ≠ 安全放行保证 | Sec.4.4 概率式改写为：可交换性 + 真稳子群条件 + 有限网格；"distribution-free" 不再暗示无条件/逐场景；摘要与结论同步降级（"calibrated trajectory band, not a stability certificate"）；Sec.5.6 报告错误放行率与回退成本（2/109 + TDS fallback） |
| **B2** "不稳定轨迹不适定"概念错误 | 删除 "only well-posed in the stable region" 因果表述，改为"不稳定轨迹误差量级超出任何有用界，band 限定于稳定子群是研究范围选择"；补充训练 ±3π 截断与评估用未截断真值的区别（Sec.4.4） |
| **B3** 离散网格表述为连续全时域 | 概率式与分数均写为 $\max_{t\in\mathcal{T}_{\mathrm{grid}}}$；Sec.5 补 2 ms 密网格抽查（n=200，0 个标签翻转） |
| **B4** q<180° 不充分 | Sec.4.4/5.6：报告可判定占比 = predicted spread + q̂_pw < π：**α=0.1 时 0/58=0%**，α=0.3 时 31%、α=0.5 时 67%（`decidable_vs_alpha.py`）；错误放行率 1.8% (2/109)；Šidák 附加结构条件说明 |
| **B5** 启发式 CCT 区间表述为认证区间 | Related Work "conformal CCT estimate" 删除；Sec.4.4 给出端点算法、无解/截断规则与"无联合保证"原因（覆盖对象是固定 tc 的轨迹而非泛函）；Sec.5 报告端点：2q̂=296.2° 时保守端点在全部 12 母线都落到搜索下界 10 ms——**带宽空泛**，如实报告 |
| **B6** 二分分辨率≠物理精度 | `dataset.py:simulate_batch` 重写为事件对齐 RK4（在 tc 处分裂积分步+线性插值到输出网格）；`evaluate.py` 参考 CCT 显式 dt=0.5 ms；Sec.4.5 区分搜索容差 (12 μs)/积分步长 (0.5 ms)/输出采样 (10 ms)；0.25 ms 步长收敛检查：12 母线 CCT 在 0.1 ms 分辨率下不变；事件对齐使测试标签翻转 1/960、训练 1/2160（Sec.5.1 报告） |
| **B7** 二分单调性未验证 | Sec.3.2 CCT 定义改为"从零开始的首个失稳穿越"；Sec.5 报告稠密扫描：参考 1-3 次局部翻转/母线，算子 1-5 次（集中于最大 CCT 母线 12/37）；新增逐母线带符号误差表（Table 6） |
| **B8** π-4s 判据≠一般暂态稳定 | Sec.3.1 加操作性标签限定；Discussion 明确"未验证富模型全部场景的首摆/多摆"；删除"first swing determines the CCT"绝对化，改为"in the scenarios studied" |

## 三、基准系统与复现可信度

| 编号 | 修改 |
|---|---|
| **A6** 118 节点参数非逐机对应 | Sec.5.9 明确定位为"基于 IEEE 118 静态网络的自定义动态算例"（未做母线映射与基准换算、非 Demetriou 系统复现）；低 CCT 归因改为"assigned parameter set"；报告类别比例（25.4% 稳定 → 多数类基线 74.6%）；CCT 相对误差尺度说明（13.3 ms 相对 8-135 ms 尺度） |
| **A7** 速度混用吞吐/延迟 | `measure_speedup.py` 重写：B=1 延迟、B=64 吞吐、真实 16 步 CCT 全流程分开；线程固定 1、warm-up、30 次重复、median/P95 | 吞吐 **700×→214×**（0.131 vs 28.0 ms）；端到端 CCT 含编码 **318×**（30.6 ms vs 9735 ms）；100 点扫描 12.3 ms；Table 1 与 Sec.5.2 全部改写 |
| **A8** 复现包不完整 | 新增 `requirements.txt`（实测版本锁定）；README 更新为修正后协议与数值表；修复文档中 "Table 1" 旧引用与 jackknife+/cross-conformal 命名残留（`evaluate.py` 旧函数标注弃用） |

## 四、创新性与公平对比

| 编号 | 修改 |
|---|---|
| **C1** MLP 能力描述错误 | Fig.4 图注与 Sec.5.3/6.6 删除"MLP 不提供连续时间轨迹或保形界"；改为同一协议下评估：MLP q_pw=86.9°(cov 0.837)、CCT 20.7 ms、GRU q_pw=104.3°(0.831)、CCT 33.5 ms（Table 3）；Discussion 诚实承认"MLP 在认证紧度与 CCT 上不弱于算子"，算子优势限于轨迹 RMSE/低数据/连续时间求值 |
| **C2** 创新点拼接风险 | Related Work 新增 Table 2 功能矩阵（ES-PINN/Conformalized-DeepONet/Voltage conf./DeepONet-grid-UQ/Ours × 编码/轨迹变量/拓扑泛化/校准单位/覆盖对象/CCT）；"first" 限定为"转子角轨迹带+CCT 提取"并指路矩阵 |
| **C3** 基线预算不对齐 | Sec.5.2 修正"共享同一损失"的错误表述（GCN 用 BCE/300 epochs，明确说明）；新增 severity-MLP 分类器 (93.0%) 与 CCT 回归器 (47.0 ms) 同输入对照；MLP/GRU 配同一保形+CCT 协议 |
| **C4** 小样本结论偏强 | Sec.5.4 样本效率结论降级（"indicative rather than statistically established"，3 种子 14.0° 波动）；Sec.5.7 物理残差负结果限定范围（"this setting…does not rule out…"）；Sec.5.8 边界采样上升不再单一归因 |
| **C5** 编码唯一性 | Sec.4.1 明确"不声称单射"（初始加速度只描述初态向量场），并说明共形带负责标记不可靠场景 |
| **C6** 固定运行点部署 | Discussion 补充：20% 增负荷场景的功率平衡/重调度未重优化说明；部署定位限定为"校准域内筛查工具"；OOD 检测/拒绝列为必要 future work |
| **C7** transfer 表述 | 全文 "transfers to" 改为 "retrained on"（one-axis/exciter/damped/118 均为重训，编码不变）；高阶模型残差仍为简化摆方程残差，不声称完整 DAE 残差（Sec.5.5） |
| **C8** 鲁棒性验证范围 | Sec.4.1 说明 a_i 为标幺速度加速度（角加速度=ωs·a_i），除以 M 非完整无量纲化；Sec.5.10 噪声实验改为"feature noise"限定，并报告新数字（acc 89.1/82.9/73.0%），声明相关/偏置参数误差与共形退化待测 |

## 五、数值与统计一致性

| 编号 | 修改 |
|---|---|
| **D1** 96.7/97.1 口径冲突 | 全部主结果由 `recompute_all.py` 单脚本导出；噪声节的 clean 基线引用同一宏 \pctACC；表 2 附 TP/FP/FN/TN；四种子 94.0%±1.2%（宏 \accSeed） |
| **D2** RMSE 定义混用 | Sec.5.2 明确定义 pooled RMSE/每场景 MAE 并声明不可比；区域 RMSE 按一次性平方和计算（fault-on 2.3°/post 36.9°/full 36.3°） |
| **D3** 104° 推导不成立 | 删除 180°/√3 论证，改为实测均匀随机预测器 RMSE **110.1°**（`recompute_all.py` D3） |
| **D4** 覆盖率样本量 | Sec.5.6 全部覆盖率附 k/n 与二项 95% CI；近边界 n=13 的 46.2% 附 19-75% CI 并标注描述性 |
| **D5** 118 结论收窄 | 118 报告类别比例、多数类基线、相对 CCT 误差；"classification scalability does not imply trajectory-level scalability" 明示 |

## 六、图表（全部按新协议重生成）

| 编号 | 修改 |
|---|---|
| **E1** Fig.1 | `make_fig_arch.py` 重画：训练/校准/推断三阶段分离，校准数据只进入 band；输出改为 "trajectory band ±q̂ (stable scenarios, grid)"；图注说明不是稳定证书 |
| **E2** Fig.2 | `make_figures.py` 补画首摆共形阴影带（70.8°@α=0.1，与正文/图注同源）；图注标注窗口由真值定义（offline diagnostic）；fault bus 11, tc=0.170 s 与图一致 |
| **E3** Fig.3 | 补画 ±25.2 ms MAE 参考带（图注明确"MAE 参考线，非预测区间"）+ 母线编号标注；最大误差点（bus 12）正文解释 |
| **E4** Fig.4 | 图注补 λ 值与单种子声明；删 MLP 能力错误描述；柱值按新协议（141.4/39.2/43.1/36.3°，88.0/95.6/96.9/95.4%） |
| **E5** Fig.5 | `conformal_analysis.py` 重写：左图仅 4 个报告水平+二项误差条；右图 3 组与正文一致，柱上标注 n_cal/n_eva 与覆盖率 |

## 七、参考文献（27 条逐条）

| 编号 | 处理 |
|---|---|
| [1] Sauer&Pai | 保留（合理基础来源）；正文补"参数表在代码仓库" |
| [2] Kundur | 书名大小写统一 ✓ |
| [3] Stott 1979 | 保留；DOI 待投稿前按 IEEE 元数据终核（已列入投稿前核查清单） |
| [4] Alimi 综述 | 正文 "all/most" 措辞已限定（abstract 改 "many classification-based surrogates"） |
| [5] DeepONet | 补 DOI 10.1038/s42256-021-00302-5 |
| [6] Lei 2018 | 正文落实次序统计量与可交换性（A4） |
| [7] Karampinis | 保留（EPSR 263:113755, 2027，审查已核实 ScienceDirect）；补 "in press" 状态说明 |
| [8] Mollaali | 升级为"最接近工作"重点比较（Related Work + Table 2 矩阵） |
| [9] Lu PES GM | 保留；删除未核实的页码；会议全称待投稿前终核（列入清单） |
| [10] FNO | 补 arXiv:2010.08895 |
| [11] PINN | 补 DOI 10.1016/j.jcp.2018.10.045 |
| [12] Moya Physica D | 审查已核实；作为创新边界依据（C2 矩阵） |
| [13] Huang AEES | 保留；删除未核实的页码 264-270（列入终核清单） |
| [14] Chen 博士论文 | 补永久链接 eScholarship uc/item/8sh6v93r |
| [15] Chen IJCNN | 补 arXiv:2205.06576；页码 4933-4940 列入终核清单 |
| [16] Hao ES-PINN | 正文标明 preprint 状态与日期（arXiv:2607.27681, 2026-07） |
| [17] Stiasny | **描述修正**："通过残差训练网络、直接评估"（不再写"推断时逐场景最小化残差"） |
| [18] Wang 2025 | 描述收敛到题名事实（同步/异步发电 CCT 快速计算），不再概括为小扰动线性法 |
| [19] Moya grid-UQ | 作者顺序修正为 Moya, Zhang, Lin, Yue；补 pp.166-182（审查经 Crossref 核实） |
| [20] Ma UQNO | 补 arXiv:2402.01960 |
| [21] Athay | 正文注明"含电导路径积分项的实现（直线近似），非无损简化" |
| [22] Wang IMEAC | 正文注明 "IMEAC-style，非标准方法完全复现" |
| [23] Vovk | 保留（Mondrian 基础书目） |
| [24] Gibbs&Candès | 保留；页码列入终核清单 |
| [25] Lei&Wasserman | 保留；正文不声称任意局部条件覆盖 |
| [26] Romano CQR | 明确列为 future work（不再暗示已评估） |
| [27] Demetriou | 保留；正文 A6 已如实定位参数使用方式 |

## 八、写作与版式

| 编号 | 处理 |
|---|---|
| **F2** 公式引用残留/无编号 | LaTeX 源无 [eq:] 残留（Word 版问题源于导出管线）；重建 PDF 中公式自动编号、Section 交叉引用全部有效（\ref 无 undefined） |
| **F3** 表格图注跨页 | 终稿 PDF 重排；DOCX 用 pandoc 重导出时保持表格整体 |
| **F4** 作者信息 | 补通讯作者标记（Zhenyu Liu, 邮箱 3293857014@qq.com）；Funding/Competing interests 改 "The authors"；CRediT 保留并提示与真实贡献核对 |
| **F5** 图像质量 | 5 幅图全部矢量 PDF 重生成 |
| **G1** 摘要过密 | 摘要重写（约 200 词），"many classification-based surrogates" 限定，条件与主要可信结果明示 |
| **G2** 标题措辞 | 标题改为 **"A Physics-Encoded Neural Operator with Conformal Trajectory Bands for Fast Transient Stability Screening"**（审查建议方向）；正文明确 physics-informed 指编码、band ≠ certificate |
| **G3** 绝对化措辞 | "far more efficiently/order-of-magnitude/confirms/inherently/clearly" 全部降级为在本文条件下的观察（Sec.2/5 多处） |
| **G4** 自我评价表述 | 删除全部 "openly report/honestly/we report this openly"；开发历史压缩为"scaling check"一句 |
| **G5** 术语单位 | q̂ 半宽与 2q̂ 成对上界在 Sec.4.4 区分；N-0/N−0、Fig. 统一；时间单位 ms 统一标注；a_i 与 ωs·a_i 区分（C8） |

## 投稿前仍需作者核对的事项（审查报告"待核验"项，无法由代码审计完成）

1. [3] Stott DOI、[9] PES GM 会议全称、[13] AEES 页码、[15] IJCNN 页码、[24] NeurIPS 页码——按正式 proceedings 终核；
2. CRediT/基金/利益冲突声明与真实贡献核对（不虚构）；
3. MPCE 官方模板迁移与 10 页限制（当前为 elsarticle 单栏 34 页，需按官方模板压缩/转移细节到补充材料）；
4. 按期刊要求披露 LLM 使用情况；
5. 论文与仓库绑定同一 tag 提交（修复后提交见 git log）。

## 修复前后核心数字对照

| 指标 | 修复前（92ed627） | 修复后 |
|---|---|---|
| 分类准确率（max-over-time） | 96.7%（末时刻判据） | **95.4%** |
| 成对认证界 q_pw (α=0.1) | 102°（abs 分数） | **148.1°** |
| 可判定占比 (α=0.1 / 0.3 / 0.5) | 未报告 | **0% / 31% / 67%** |
| 单机带半宽 q̂ / 覆盖率 | 111° / 0.893（近似分位数） | **110.1° / 0.882 (157/178)** |
| CCT MAE / 最大误差 | 12.3 / 48.8 ms | **25.2 / 204.9 ms**（bus 12） |
| 吞吐加速比 | ≈700× | **214×**（线程固定+重复测量）；端到端 CCT 318× |
| 118 分类 / RMSE | 95.0% / 118.8° | **92.9% / 121.9°**（自定义算例定位） |
| 门控覆盖率 | 0.946（混入校准样本） | 全门控 0.914 / oracle 0.946，纯度 96.6%，错误放行 1.8% |
