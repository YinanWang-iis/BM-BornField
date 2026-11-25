# BM-BornField：EGCNN（Born 有效电荷网络）GPU 版 LAMMPS 接口 `fix becfield/kk`

（英文说明见 README.md。本仓库是课题组集群上 `/homenfsxx3/ynwang/symmetrix_test/egcnn/` 的源码，第 1 节是集群上的安装位置；
仓库里对应的目录：`lammps/`（fix 源码与共用 CUDA 核心）、`python/`（Python 模块 bornfield）、`tools/`（导出与内核生成）、`cluster/`（集群用的构建与环境脚本）、`tests/`。
导出的模型 json 和测试结构不在仓库里。）

给 EFMLP 方法 2 用：零场 MACE（`pair_style symmetrix/mace/foundation/float32`）＋ EGCNN 预测的 Born 有效电荷 Z\*，
在 GPU 上算场致力和累计极化，取代 `efmlp_lammps.BECFieldForce`（`fix external pf/callback` + Python）。

- 2560 原子 BaTiO3：每步 EGCNN 部分 **147 ms → 4.6 ms（约 32×）**
- 8960 原子 hBN：**445 ms → 15.5 ms（约 29×）**

上面两组数字都是在共享的 RTX 5090 上测的（同卡另有作业，GPU 利用率约 88%）。
Z\* 与 Python 参考逐原子一致：float32 下最大差 ≤ 7e-6 e。

另外提供 **Python 模块 `bornfield`**（ASE calculator，见第 7 节）：同一份 CUDA 网络，加上自带的 GPU 邻居表，
可以在 ASE 里和 `symmetrix.experimental.MACEFoundation` 叠加使用。2560 原子每次调用 1.9 ms，8960 原子 8.4 ms。

---

## 1. 安装位置与激活

| 内容 | 路径 |
|---|---|
| 根目录 | `/homenfsxx3/ynwang/symmetrix_test/egcnn/` |
| 激活（在有 GPU 的节点上） | `source /homenfsxx3/ynwang/symmetrix_test/egcnn/env.sh` |
| LAMMPS 可执行（按显卡自动选） | `/homenfsxx3/ynwang/symmetrix_test/egcnn/bin/lmp` |
| Blackwell 版（5090、RTX PRO 6000/5000） | `egcnn/lammps_sm120/` |
| 非 Blackwell 版（3090、4090、6000 Ada、A100） | `egcnn/lammps_sm80/` |
| 已转换好的模型 | `egcnn/models/egcnn_bto.json`、`egcnn/models/egcnn_hbn.json` |
| 源码 | `egcnn/lammps/`（fix、共用 CUDA 核心 `becfield_core.cuh`、生成的内核头文件、install.sh） |
| Python 模块 | `egcnn/python/`（源码），`egcnn/python_sm120/`、`egcnn/python_sm80/`（编好的 `bornfield` 包） |
| 构建脚本 | `egcnn/build_lammps.sh`、`egcnn/build_python.sh`（`ARCH=sm120\|sm80`） |
| 工具 | `egcnn/tools/`（导出、内核生成、纯 torch 参考实现） |
| 测试 | `egcnn/tests/` |

**`env.sh` 做的事**

- 先 `source field/env.sh`：按 GPU 选 Python venv（sm120 / sm80），这一步和原来一样。
- 再把 LAMMPS 换成 `egcnn/lammps_<arch>`：设置 `SYMMETRIX_LMP`、`PATH`、`LD_LIBRARY_PATH`。
- 导出 LAMMPS 的 Python 模块和 `bornfield` 模块：`PYTHONPATH=egcnn/python_<arch>:egcnn/lammps-src/python`。

用 Python 驱动时，可以直接用它代替 `MD/common/symx_env.sh` 里的 `field/env.sh` 那一层。没有 GPU 的节点上，先 `export SYMX_ARCH=sm120` 或 `sm80`。

**这套 LAMMPS 和 field 版的关系**

- 它是 field 版（`field/lammps*`）的超集：同一份 LAMMPS 源码（stable 30Sep2026），同一个 symmetrix（含 MACE-Field 的 `efield`），只多了 `fix becfield/kk`。
- 方法 1/3/4 也可以用它跑。
- `field/env.sh`、`field/bin/lmp`、`field/lammps*` 都没有改动。

## 2. 模型转换

在 EGCNN 训练环境里运行（需要 e3nn + torch_geometric，CPU 即可）：

```bash
/homenfsxx3/ynwang/EGCNN/bin/python /homenfsxx3/ynwang/symmetrix_test/egcnn/tools/export_egcnn.py \
    /homenfsxx3/ynwang/EFMLP_method_compare/BEC_PRED/BaTiO3 -o egcnn_bto.json
```

**导出的内容**

- 把 e3nn 的部分全部展开成普通数组：嵌入、径向 MLP、17 条张量积路径（含 e3nn 的归一化）、每层 Linear、输出 Linear 加 CartesianTensor。
- 超参数写死为 `num_layers=6, feature_size=64, num_radial=32, edge_sh_lmax=2, cutoff=4.5, embedding_dim=32, equivariant_cart=True`，和 `efmlp_lammps.load_egcnn` 一致。
- 导出时用 float64，和 e3nn 原模型的差为 5e-15 e。

**现成的文件**（2026-11-25 由 `trained_model.pth` 导出）

- `models/egcnn_bto.json`：BEC_PRED/BaTiO3
- `models/egcnn_hbn.json`：BEC_PRED/hBN_v2

## 3. LAMMPS 输入写法

```
atom_modify  map yes                      # 必须
pair_style   symmetrix/mace/foundation/float32
pair_coeff   * * NON_FIELD__mace_bto.json Ba Ti O
variable     EFX equal 0.0
variable     EFY equal 0.0
variable     EFZ equal "0.03*sin(2*PI*step/20000)"       # V/Å，可含 step，每步重新求值
fix          ef all becfield /homenfsxx3/ynwang/symmetrix_test/egcnn/models/egcnn_bto.json Ba Ti O efield v_EFX v_EFY v_EFZ
thermo_style custom step temp pe f_ef[1] f_ef[2] f_ef[3] f_ef[4] f_ef[5] f_ef[6]
```

运行：`lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.xxx`。

用 `-sf kk` 时写 `becfield` 即可，也可以直接写 `becfield/kk`。

**语法**

```
fix ID all becfield <model.json> <elem_type1> ... <elem_typeN> efield Ex Ey Ez [可选关键字]
```

- 元素符号按 LAMMPS 原子类型顺序写，个数等于 ntypes。BaTiO3 是 `Ba Ti O`，hBN 是 `B N`。
- `Ex Ey Ez`：数值或 `v_名字`（equal 变量），单位 V/Å。

| 可选关键字 | 默认 | 含义 |
|---|---|---|
| `p0 Px Py Pz` | 0 0 0 | 累计极化的初值（e·Å），续算用 |
| `precision float32\|float64` | float32 | 网络计算精度（float32 与训练和原 Python 版一致） |
| `radial spline\|exact` | spline | 径向权重 w(r) 的算法：`spline` 为三次 Hermite 表（误差约 1e-8 e）；`exact` 为每步用 cuBLAS 直接算 MLP，较慢，用于核对 |
| `grid N` | 1024 | 样条表的区间数（[0, 4.5] Å） |
| `chunk N` | 262144 | 只对 `radial exact` 有效：每批最多处理的边数，用来限制显存 |
| `zfile 文件 every` | 无 | 每 `every` 步把全部原子的 Z\*（按 atom ID，9 分量）追加写入文件，用来核对 |
| `timing yes\|no` | no | 每次 run 结束时打印平均每步耗时 |

### 物理量

**Z\***

- 约定为 `Z[a,b]`：a 是电场或极化方向，b 是位移方向，按行展开成 9 个分量，与 VASP 和训练标签一致，没有转置。
- 先减去全体原子的平均值（ASR，与 `apply_acoustic_sum_rule` 相同），所以 ΣF_field = 0。

**场致力**

- `F_i[b] += Σ_a ε_a Z_i[a,b]`。
- 这部分力没有对应的能量，不进 pe，也不进维里或压强，与原 Python 版相同。

**累计极化 ΔP**（全局向量 `f_ef[1..3]`，单位 e·Å，整个模拟盒，不除以体积）

- 计算式：`ΔP_a += Σ_i ½ (Z_i(t) + Z_i(t−dt))[a,b] · Δr_i[b]`。
- Δr 取分数坐标差再做最小镜像，然后乘以两步的平均晶胞，与 `BECFieldForce._callback` 完全相同。
- 每次力计算都累加一次，包括每个 `run` 开始时的 setup 调用：位置没变时增量为 0。

**`f_ef[4..6]`**：当前步的电场 (Ex, Ey, Ez)，单位 V/Å。

### 续算（两种方式都测过）

1. **write_restart / read_restart**：ΔP 写在 restart 文件里。`read_restart` 之后用**同一个 fix ID 和同样的参数**重新定义这个 fix，ΔP 会自动恢复。
2. **自己的断点（experiments.py 现在的做法）**：`write_data` 加上自己保存的 P，续算时写 `fix ef all becfield ... p0 Px Py Pz`。

续算后的第一次调用只记录参考位置，不累加，与 Python 版 `prev=None` 的行为相同。

### cg 弛豫（minimize）

minimize 时**不需要 unfix**。这个 fix 在 minimize 中什么也不做：

- 不加场致力，所以 cg 收敛到纯 MACE 的结构；
- 不更新 ΔP。

minimize 之后的下一个 `run`，开头的 setup 调用会把"弛豫前位置 → 弛豫后位置"这一步按梯形公式计入 ΔP。这与现在 experiments.py 的做法一致：unfix 之后用同一个 `BECFieldForce` 对象重新 fix。

如果不想计入这一步，可以在 minimize 之后再定义这个 fix，或者 unfix 后重新 fix 并给 `p0`。

### 与 Python 驱动一起用时

`lmp.close()` 之后请再调用 `lmp.finalize()`。

否则 Python 退出时 Kokkos 会报 `cudaErrorCudartUnloading` 并 abort（退出码 134）。这是原有问题：field 环境里不用这个 fix 也会出现。

## 4. 验证结果

全部数值来自 `tests/`。参考实现有两个：

- **Zpy32**：`efmlp_lammps.BECFieldForce.predict_becs`，即 e3nn 的 model.py，float32，ASR。
- **Z64**：`tools/egcnn_ref.py`，对导出的 JSON 用纯 torch 做 float64 计算。它和 e3nn 原模型的差为 5e-15 e。

### 4.1 Z\* 与场致力（`tests/sp_bec.py` + `compare_sp.py`）

场致力 ΔF 用 `pair_style zero 6.0` 单独取出，避免 float32 MACE 每次运行的不可重复噪声（约 1e-3 meV/Å）混进来。
ε1 = (0, 0, 0.03) V/Å，ε2 = (0.02, −0.01, 0.03) V/Å。

| 结构 | N | float32：max\|Z−Zpy32\| (e) | float32：max\|Z−Z64\| (e) | float64：max\|Z−Z64\| (e) | float32 场致力误差 ε1 / ε2 (meV/Å) | 场致力最大值 ε1 / ε2 (meV/Å) | \|ΣF_field\| (meV/Å) |
|---|---|---|---|---|---|---|---|
| BaTiO3 测试集 3 帧 | 40 | 2.7e-6 | 2.4e-6 | 8e-9 | 4.8e-5 / 5.2e-5 | 211 / 212 | < 1e-12 |
| hBN 测试集 3 帧 | 36 | 1.1e-6 | 6.4e-7 | 1.4e-9 | 1.6e-5 / 1.3e-5 | 9.2 / 56 | < 1e-12 |
| BaTiO3 4×4×4 超胞（扰动 0.03 Å） | 2560 | 4.3e-6 | 3.4e-6 | 1.0e-8 | 9.3e-5 / 8.7e-5 | 221 / 223 | < 1e-11 |
| hBN 双层畴壁（H3 的 relaxed.data） | 8960 | 6.9e-6 | 1.2e-6 | 1.6e-9 | 2.9e-5 / 3.3e-5 | 9.5 / 54 | < 1e-9 |

说明：

- **float32 的误差来源**：Python 版自己的 float32（Zpy32 对 Z64）误差就有 0.5–6e-6 e。本接口 float32 的误差与它同一量级。
- **场致力误差**：等于 ε 乘以 Z\* 的 float32 噪声。
- **`radial exact` + float64**：与 Z64 的差为 5e-10 e，这是 zfile 打印精度的极限，力的误差为 1e-12 meV/Å。由此确认内核本身是精确的，`spline` 只带来约 1e-8 e 的误差。
- **sm80 版（RTX 3090）**：同样的测试结果为 float32 max\|Z−Z64\| ≤ 2.0e-6 e，float64 ≤ 1.0e-8 e。
- **与 MACE 一起跑**：ΔF = F(ε) − F(0) 与 ε·Z64 之差约 1.5e-3 meV/Å，全部来自 float32 MACE 力本身的运行间抖动。

### 4.2 累计极化（`tests/md_P.py` + `replay_P.py`）

测试设置：

- NVE，600 K，200 步，dt = 1 fs，sm120。
- 随步数变化的电场：`ex = 0.01 sin(0.05 step)`，`ey = −0.005`，`ez = 0.0006 step − 0.03`。

对照方法：把 LAMMPS 每步的轨迹交给 Python，用 `predict_becs` 加上 `_callback` 里完全相同的累计公式重放。

| 体系 | \|ΔP\| 终值 (e·Å) | 200 步内 max\|P_lammps − P_python\| (e·Å) |
|---|---|---|
| BaTiO3 320 原子 | 90.3 | 1.2e-6 |
| hBN 324 原子 | 0.46 | 8.9e-7 |

### 4.3 续算

做法：先连续跑 200 步，再从第 100 步分两种方式续算（read_restart；write_data + p0），比较第 200 步的 ΔP。

| 体系 | read_restart | write_data + p0 |
|---|---|---|
| BaTiO3 320（sm120） | 3.2e-6 | 7.8e-6 |
| hBN 324（sm120） | 3.3e-6 | 1.4e-6 |
| BaTiO3 320（sm80，3090） | 9.2e-6 | 4.2e-6 |

单位为 e·Å。差异来自 float32 MACE 轨迹本身的不可重复性，ΔP 的量级约为 90。

### 4.4 minimize（`tests/min_check.py`）

- 带着这个 fix（ε_z = 0.03）做 cg：能正常跑完，不报 DualView 错误，fnorm 收敛到 0.007 eV/Å（纯 MACE）。
- minimize 期间 ΔP 不变；之后接着跑 MD 也正常。

### 4.5 原有 pair_style 结果不变（`tests/pair_same.sh` + `compare_pair.py`）

用这套 LAMMPS 重跑之前 MACE-Field 的单点对照（BaTiO3 40/320 原子，4 个场，3 种精度），与原 sm_120 构建（field/tests/lammps/sp）比较：

- **fp64**：能量差 0，力差 ≤ 1e-10 meV/Å（打印精度），P 差 0。
- **fp32 / fp32-mixed**：dE ≤ 3e-5 meV/atom，dF ≤ 1.9e-3 meV/Å，在 float32 噪声范围内，与之前 sm80 构建的核对结果相同。

### 4.6 速度（`tests/bench.sh`）

测试条件：NVE 200 步，ε_z = 0.03。显卡都与其他作业共享，数字偏保守。

| 体系 | 卡 | 原 Python 回调 | fix becfield/kk | 加速 | 整步：MACE 单独 → MACE + 本 fix |
|---|---|---|---|---|---|
| BaTiO3 2560（67,780 条边） | RTX 5090（共享，利用率约 88%） | 147 ms | **4.6 ms** | 32× | 4.9 → 9.4 ms（约 106 步/s；原来约 6–7 步/s） |
| hBN 8960（311,804 条边） | RTX 5090（同上） | 445 ms | **15.5 ms** | 29× | 12.5 → 28.1 ms（约 36 步/s） |
| BaTiO3 2560 | RTX 3090（与一个 BaTiO3 MD 共享，利用率 93%） | — | 12.5 ms | — | 15.9 → 27.2 ms |
| hBN 8960 | RTX 3090（同上） | — | 54.5 ms | — | 46.7 → 99.7 ms |

5090 上的耗时分布（`BECFIELD_PROFILE=1` 环境变量可以打印）：

- BaTiO3：建边 1.5 ms，张量积 1.3 ms，线性层 0.24 ms。
- hBN：建边 1.5 ms，张量积 9.6 ms，线性层 2.4 ms。

## 5. 实现概要

- **邻居表**：直接用 LAMMPS Kokkos 的 full 邻居表（GPU 上）。ghost 原子通过 atom map 映射回本地原子，与 symmetrix foundation pair 的做法相同。整个过程没有 CPU 邻居表，没有 host↔device 拷贝，也没有 Python 回调。
- **每层的计算**：
  - 径向权重 w(r) = cut(r)·MLP(rbf(r)) 只依赖距离，在初始化时由 float64 算成三次 Hermite 表（值和导数，1024 个区间）。
  - 张量积按"每个接收原子一个 block、64 个通道"做 gather，没有原子操作，结果确定。
  - 17 条路径的 Clebsch-Gordan 系数由 `tools/gen_kernels.py` 生成为直线代码（`egcnn_generated.hpp`）。
  - 每层的 e3nn Linear 是分块对角的，用专门的 kernel 计算。
- **ASR、力和 ΔP**：在一个 kernel 里完成。ΔP 按 atom ID 保存上一步的分数坐标和 Z\*，不受原子排序或迁移影响，用 double 累加。
- **安全检查**：加载模型时会核对 JSON 里的张量积系数是否和编译进去的内核一致，不一致就报错。

## 6. 已知限制

- **只支持单进程单卡**：多个 MPI rank 时直接报错。fix 必须作用于 group all，atom ID 必须是连续的 1..N，必须 `atom_modify map yes`。
- **pair_style 截断**：需要 ≥ 4.5 Å（EGCNN 的截断）。MACE 的模型都满足，不满足时 init 会报错。
- **架构固定**：只支持这一种 EGCNN 架构（6 层、64 通道、lmax=2、equivariant_cart=True），BaTiO3 和 hBN_v2 两个模型都是。改超参数需要重新生成内核头文件并重新编译。
- **没有能量和维里**：场致力不进入压强，所以 NPT 的压强里没有场的贡献。这和原 Python 版一样，是方法 2 本身的定义。
- **NPT 下的 ΔP**：晶胞取两步平均，公式与 Python 版相同。但 ΔP 只在固定晶胞（NVE）下测过，没有在变晶胞的 MD 里做过逐步对照。
- **非 Blackwell 卡**：sm80 版只在 RTX 3090 上测过。4090、6000 Ada、A100 这次没有空闲的卡。sm_80 的二进制在这些卡上可以直接运行，之前的 field sm80 环境在 4090 和 A100 上测过。
- **速度数字**：都是在共享显卡上测的，独占时应更快，但没有测过。
- **CPU 的 `fix becfield`（不加 kk）**：没有实现，用了会报错并提示加 `-sf kk`。
- **Python 模块**：只有 GPU 版。邻居表的分格在 CPU 上做（每次调用把坐标拷到 GPU），对上万原子仍然很快，但比 LAMMPS 版多一点开销。

## 7. Python 模块 `bornfield`（ASE calculator）

`source egcnn/env.sh` 之后可以直接 `import bornfield`（按显卡自动选 sm120 / sm80 版）。
网络和 LAMMPS fix 用的是同一份 CUDA 代码（`lammps/becfield_core.cuh`），邻居表是模块自带的 GPU cell list。

```python
import numpy as np
from ase.calculators.mixing import SumCalculator
from ase.md.verlet import VelocityVerlet
from ase import units
from symmetrix.experimental import MACEFoundation
from bornfield import BornField, PolarizationTracker

E = "/homenfsxx3/ynwang/symmetrix_test/egcnn/models/egcnn_bto.json"
M = "/homenfsxx3/ynwang/EFMLP_method_compare/MD/models_lammps/NON_FIELD__mace_bto.json"

bf = BornField(E, electric_field=(0, 0, 0.03))          # V/Å；也可以传 f(atoms) -> (3,)，或设 atoms.info["electric_field"]
Z = bf.becs(atoms)                                       # (N, 3, 3)，Z[i, a, b]，已做 ASR
atoms.calc = SumCalculator([MACEFoundation(M, dtype="float32"), bf])   # 零场 MACE + 场致力

tr = PolarizationTracker()                               # 可以给初值 PolarizationTracker(P0=(Px, Py, Pz))
atoms.get_forces(); tr.update(atoms)                     # 先记录起点
dyn = VelocityVerlet(atoms, 1.0 * units.fs)
dyn.attach(lambda: tr.update(atoms), interval=1)         # 每步累加 ΔP（e·Å），公式与 LAMMPS fix 相同
dyn.run(1000)
print(tr.P)
```

**BornField 的输出**

- `forces`：场致力 F_i[b] = Σ_a ε_a Z_i[a,b]。
- `becs`：Z\*，形状 (N, 3, 3)。
- `energy`、`stress`：返回 0。场致力不是保守力，和 LAMMPS 版一样没有能量和维里。

**参数**

| 参数 | 默认 | 含义 |
|---|---|---|
| `electric_field` | (0, 0, 0) | 电场（V/Å）。可以是 3 个数，也可以是函数 f(atoms)；`atoms.info["electric_field"]` 优先 |
| `precision` | "float32" | "float32" 或 "float64" |
| `radial` | "spline" | "spline" 或 "exact"（与 LAMMPS fix 的同名关键字相同） |
| `grid` | 1024 | 样条区间数 |

**低层接口**：`bornfield.BornFieldGPU(model, precision, radial, grid).compute(positions, numbers, cell, pbc)`
返回 (N, 9) 的 numpy 数组。

**支持的体系**：三维周期、部分周期（例如 pbc = [True, True, False] 的板）、非周期（没有晶胞的团簇）都可以。
坐标不需要先折回晶胞。

**PolarizationTracker**

- 每次力计算之后调用一次 `update(atoms)`，Z\* 从 `atoms.calc` 里的 BornField 结果读取，支持 SumCalculator。
- 也可以自己传 `update(atoms, Z)`。
- 公式与 `BECFieldForce._callback` 和 LAMMPS fix 完全相同：梯形、最小镜像、两步平均晶胞。

**Python 模块的验证**（`tests/py_bornfield.py` + `compare_py.py`，sm120 版在 5090 上）

| 体系 | N | float32：max\|Z−Z64\| | float32：与 LAMMPS fix 之差 | float64：max\|Z−Z64\| | float64：与 LAMMPS fix 之差 |
|---|---|---|---|---|---|
| BaTiO3 测试帧 | 40 | 2.1e-6 | 8.9e-7 | 6.8e-9 | 5e-10 |
| hBN 测试帧 | 36 | 6.9e-7 | 7.6e-7 | 8.9e-10 | 5e-10 |
| BaTiO3 超胞 | 2560 | 3.4e-6 | 1.9e-6 | 9.7e-9 | 5e-10 |
| hBN 畴壁 | 8960 | 1.1e-6 | 9.6e-7 | 1.1e-9 | 5e-10 |
| BaTiO3 原始三斜帧（未旋转、坐标平移到晶胞外） | 40 | 1.8e-6 | – | 8.2e-9 | – |
| BaTiO3 非周期团簇（无晶胞） | 65 | 1.1e-6 | – | 8.8e-9 | – |
| hBN 板（pbc = T T F） | 36 | 5.4e-7 | – | 1.1e-9 | – |

单位均为 e。补充说明：

- 边数和 ASE `neighbor_list` 完全一致：2560 原子 68,290 条，8960 原子 324,464 条。
- **旋转等变性**：Z(R r) 与 R Z Rᵀ 之差，float32 为 1.3e-6 e，float64 为 3e-8 e。
- **ASE MD**：MACE（symmetrix）+ BornField + PolarizationTracker，320 原子跑 50 步正常，|ΣF_field| = 5e-16 eV/Å。
- **速度**：每次调用（含 CPU 分格和拷贝）2560 原子 1.9 ms，8960 原子 8.4 ms（5090，与其他作业共享）。
- **sm80 版（RTX 3090，与一个 BaTiO3 MD 共享，利用率 100%）**：同样 7 个体系 float32 max\|Z−Z64\| ≤ 1.9e-6 e，
  float64 ≤ 1e-8 e，旋转等变 1.6e-6 / 3e-8 e，ASE MD 正常；速度 15 ms / 61 ms（卡被占满，仅供参考）。

**拆出共用核心后 LAMMPS fix 的回归检查**：

- float64 的 Z\* 与拆分前逐位相同（sm120 和 sm80 都是）。
- float32 的差在 1e-6 e 量级，属于 float32 运行间的噪声。
- 200 步 MD 的 ΔP 与拆分前的差为 1.4e-5 e·Å（float32 MACE 轨迹的抖动）。

## 8. 测试文件

`tests/` 下的文件：

- **结构**：`make_struct.py` 生成 `struct/`。BaTiO3 是三斜晶胞，所以参考坐标取自写出后的 LAMMPS 数据（已经过旋转）。
- **单点**：`sp_bec.py`（LAMMPS）、`compare_sp.py`（EGCNN 环境，CPU）。
- **MD 和续算**：`md_P.py`、`replay_P.py`。
- **minimize**：`min_check.py`。
- **速度**：`bench.sh` + `in.bench`。
- **原有 pair_style 是否改变**：`pair_same.sh` + `compare_pair.py`。
- **Python 模块**：`py_bornfield.py`（GPU）+ `compare_py.py`（EGCNN 环境，CPU）。
- **日志**：`final_sm120_gpu.log`、`final_sm120_cpu.log`、`final_sm80_win3090.log`、`final_sm80_cpu.log`、`compare_pair_sm120.log`、`py_sm120_gpu.log`、`py_sm120_cpu.log`、`py_sm80_win3090.log`、`py_sm80_cpu.log`。

## 9. 引用

使用本代码请引用提出该 Born 有效电荷网络（EGCNN，BM1 架构）的论文：

> A. Kutana, K. Shimizu, S. Watanabe, R. Asahi, "Representing Born effective charges with equivariant graph
> convolutional neural networks", *Scientific Reports* **15**, 16719 (2025). https://doi.org/10.1038/s41598-025-01250-5

同时请引用所依赖的软件：

- LAMMPS：A. P. Thompson *et al.*, *Comput. Phys. Commun.* **271**, 108171 (2022)，https://doi.org/10.1016/j.cpc.2021.108171
- LAMMPS 的 KOKKOS 包：A. Johansson *et al.*, *Proc. SC '25 Workshops*, 1217–1232 (2025)，https://doi.org/10.1145/3731599.3767498
- Kokkos：C. R. Trott *et al.*, *IEEE Trans. Parallel Distrib. Syst.* **33**, 805–817 (2022)，https://doi.org/10.1109/TPDS.2021.3097283
- e3nn（网络的定义与训练）：M. Geiger, T. Smidt, arXiv:2207.09453 (2022)
- 与 MACE / symmetrix 一起使用时：I. Batatia *et al.*, MACE, *NeurIPS* **35** (2022)；symmetrix，https://github.com/wcwitt/symmetrix

BibTeX 见 README.md。实现思路（手写 CUDA 内核、直接用 LAMMPS Kokkos 邻居表、径向函数样条化）参照 symmetrix；ghost 原子映射的写法参照 symmetrix 的 pair_symmetrix（GPLv2）。许可：GPL-2.0（与 LAMMPS 相同）。
