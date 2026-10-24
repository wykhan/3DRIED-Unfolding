# 3DRIED imaging baseline suite

授权：2026-10-24 用户要求创建独立基线分支，包含 BP、RMA、ISTA、CV-ISTA、RMIST，点仿真与真实回波，100%、70%、50%、30%。本分支为 `baseline/3dried-imaging-suite`，从 `main@76ad526960e4f9d7e6aed9dee866ca9083dc196b` 建立。M9 尚未合并到 main；本套件复用 M8 以前已接受的算法，不依赖 M9 归一化，也不修改 main。

## 实验矩阵

5 算法 × 2 数据来源 × 4 采样率 × 2 策略 = **80 行**。100% 在两个策略下相同，保留两行便于完整比较。每行输出一个 `[13,512,512]` 复数体，按 `[range_y,z,x]` 排列。

- `point_sim`：相同 3DRIED 参数下，点反射率 → 已接受 RM 前向 → 截取 200×407 真实孔径。提供原始点 GT 和传播谱投影后的 GT。明确是模型匹配的回归实验（inverse crime），不能证明真实泛化。
- `real`：只读 `output_rawdata.mat`，Rx1，已有轨迹整理与蛇形扫描还原，256 点距离 FFT，MATLAB bins 12–24。无体素 GT。
- `random_2d`：在真实孔径上随机排列位置后取前 K 个。
- `height_lines`：随机排列 200 个高度后保留整行，水平孔径完整。
- 同一策略共享排序，因此 `M30 ⊆ M50 ⊆ M70 ⊆ M100`。每个 mask 对所有 fast-time 采样、所有距离层和所有算法共用。
- 采样数量采用 round-half-up。当前 81400 个位置在四档恰好是 81400/56980/40700/24420；高度为 200/140/100/60。
- padding 不属于观测；所有误差一致性计算只统计真实保留位置。

## 算法定义

| 名称 | 明确定义 | 关键限制 |
|---|---|---|
| BP | 每个距离层的球面波 Green 匹配滤波反投影；用线性 FFT 卷积加速严格离散空间求和 | 窄带逐层 BP，尚非完整 FMCW 距离徙动 BP；与 RM 存在模型差异 |
| RMA | 原 M2 `ky exp(-j ky y)`，保留 linspace、fftshift、居中 padding、一基距离约定 | 官方示例逐距离层相位补偿，不含宽带 Stolt |
| ISTA | 原 M4 Ini-ISTA：9 步，tau=1e-4，theta=1e-3，RM_dagger 更新和复数径向阈值 | 沿用原文复现的补零全域残差；不是掩模数据项的严格近端梯度 |
| CV-ISTA | 本分支暂定义的经典复值 L1 ISTA：真实 masked adjoint，安全谱界步长，100 步 | 未声称复现任何未指定的同名论文；若用户所指为特定网络，应另行指定 |
| RMIST | 原 M6 epoch99 的 9 阶段、18 个标量，按 float32 精确恢复，固定推理 | 不训练、不归一化；所有新 mask 均为跨 mask，非 50% 和整行策略增加 OOD |

BP 使用

\[
g_y(\Delta z,\Delta x)=\frac{\Delta x_a\Delta z_a}{2\pi j}\frac{e^{j2kR}}R,
\quad R=\sqrt{y^2+\Delta z^2+\Delta x^2}.
\]

\[
\hat x_{BP}(p)=\frac{\sum_{a\in\mathrm{observed}}S(a)g_y(p-a)^*}
{\sum_{a\in\mathrm{full\ physical\ aperture}}|g_y(p-a)|^2}.
\]

分母固定为完整孔径灵敏度，避免逐 SR 增益补偿掩盖缺失幅度。线性卷积与直接逐像素求和的相对误差测试必须通过；这并不意味着它与截断、周期 RM 谱的传播模型完全相同。

令 `H=RM`，`D=RM_dagger`，则原 ISTA/RMIST 保持

\[
x_0=D y_s,\qquad x_{t+1}=\operatorname{shrink}_{\theta_t}\{x_t+\tau_tD(y_s-Hx_t)\}.
\]

CV-ISTA 独立实现，不改原算法：

\[
A=MH,\quad A^H=H^HM,\quad
\min_x\frac12\|Ax-y_s\|_2^2+\lambda\sum_i|x_i|,
\]

\[
x_{t+1}=\operatorname{shrink}_{\eta\lambda}\{x_t-\eta A^H(Ax_t-y_s)\},
\quad \eta=\frac1{\max|\Phi|^2},\quad
\lambda=0.01\|A^Hy_s\|_\infty.
\]

`H^H` 的谱因子是 `conj(Phi)`，不是 `ky exp(-j ky y)`。100 次只是固定预算，并不宣称收敛。接近传播截止频率的谱值使全局安全步长保守，必须结合目标函数日志解读结果。

## 参数来源与约束

唯一运行配置：[`configs/baseline_3dried.yaml`](configs/baseline_3dried.yaml)。独立明文说明：[`docs/BASELINE_PARAMETERS.md`](docs/BASELINE_PARAMETERS.md)。两种数据来源共用同一对象和指纹，不另外复制一套仿真物理参数。

不改坐标、不融合其他 Rx、不把实测 RMA 当标签、不训练新模型、不改变已接受 RMIST。点仿真当前未包含完整 chirp 去斜、距离迁移或多散射效应。需要上述物理仿真时，应增加独立模型并保留现有回归实验。

## 验收与报告

1. 保留原有回归测试；外部 Hammer/GPU 缺失时明确 skipped，不能据此判定实测/GPU PASS。
2. masked forward/adjoint 内积、CV 目标函数单调下降、复数相位等变、BP 直接和/FFT 等价、RMA 与 M2、NumPy 与原 PyTorch RMIST 等价。
3. 仿真验证物理孔径裁剪、参数一致性；真实 loader 使用 HDF5 fixture 测试全流程及单 Rx 隔离，但 fixture 不当作实测验收。
4. 全尺寸点仿真 40 行，以及实际 Hammer 40 行分开记录。数据缺失时退出码 2、总体 INCOMPLETE。
5. 输出复数体、固定/逐图两类 dB MIP/切片、mask、3D 点云、距离层能量、参数/环境/命令/模型来源及指标 CSV。
6. 仿真可报 GT NRMSE；实测只报观测残差和 RMA reference proxy，无虚构 PSNR/GT。几何 BP 的 RM 残差是跨模型诊断。
7. CPU 运行时间只包括重建，不与未运行的 GPU 比较；不使用这批结果做最终算法优劣排名。
