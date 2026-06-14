# PhysGaussian (Taichi 复现版) — 交互式软体仿真 Demo · **Server 版**

> Lab4 · 探索物理仿真与 3DGS 的融合 · Idea 1（基于 3DGS 粒子的交互式软体仿真）
>
> **这是 server 版**（部署到 CUDA 机器用）。相对 dev 版的改动：
> - 新增 3 种**真·弹塑性/颗粒本构**（plastic / sand / foam），物体落地后**保持形状不再摊饼**（修掉 dev 版已知限制 #2）
> - 材质从 5 种扩到 **9 种**，上游 6 种 MPM 本构现在**一一对应**（不再降级成软果冻）
> - 新增 **headless 离线渲染**（`--headless` → PNG 序列 → ffmpeg 合成 mp4），无需窗口，适合服务器批量出片
> - 新增 **逐帧状态导出**（`--export-state`），格式**严格对齐官方 PhysGaussian** 的 `save_data_at_frame`（h5: x/v/f_tensor/C/time + binary ply），下游可直接复用读官方输出的代码
> - 容量/网格分辨率**环境变量可配**（`PG_NMAX` / `PG_GRID` / `PG_ICO_SUBDIV`），CUDA 上可直接跑满 27 万高斯
> - dev 版保留原样存档在 `../physgaussian/`，本目录是改动版

把每个 3D Gaussian 同时当作**渲染单元**和**物理粒子**，用 MLS-MPM 驱动它的运动与形变，
形变梯度 `F` 既决定物理应力、又决定高斯椭球的形状——这就是 PhysGaussian 的核心思想
**"What you see is what you simulate"（所见即所仿真）**。

参考论文：Xie et al., *PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics*, CVPR 2024
（https://arxiv.org/abs/2311.12198）

---

## 快速运行

依赖：Python 3.10 + Taichi 1.7.x（macOS 走 Vulkan/MoltenVK）。

```bash
# 默认：果冻球
./run.sh

# 指定场景 + 材质
./run.sh cylinder putty
./run.sh torus rubber

# 或直接调用
conda run -n py310 python main.py --scene ball --preset liquid
```

启动后是一个 3D 交互窗口，左上角有控制面板。

---

## 加载 3DGS 真实物体（idea1 闭环）

3DGS 重建管线（`secsion1/`）把多视角图片 → 3DGS 训练 → 输出 `.ply` 点云 + PhysGaussian
格式的 `config.json`（含 VLM 推断的材质和物理参数）。本 demo 能直接读进来跑物理。
打包发布时，数据就放在本目录的 `data/<名字>/` 下（已内置 `data/chair/`）：

```bash
# 按名字加载（自动在 ./data/<名字>/ 里找；找不到再回退到 secsion1 训练树）
conda run -n py310 python main.py --gs chair

# 或直接给 .ply 路径（也可以是包含 point_cloud.ply 的文件夹）
conda run -n py310 python main.py --gs /abs/path/to/point_cloud.ply
conda run -n py310 python main.py --ply path/to/point_cloud.ply --config path/to/config.json

# 控制下采样后的粒子数（默认 6000；server 版上限由 PG_NMAX 控制，CUDA 上可到 27万+）
conda run -n py310 python main.py --gs chair --gs-particles 8000
```

> `--gs <名字>` 的查找顺序：① `./data/<名字>/point_cloud.ply`（打包用的扁平布局）
> → ② `./data/<名字>/point_cloud/iteration_*/point_cloud.ply`
> → ③ `../secsion1/running/output/<名字>/...`（3DGS 训练管线的原始输出树）。
> 找不到会**明确报错并打印找过的路径**，不再静默回退成默认球。
> 想换自己的物体：把 `point_cloud.ply` + `config.json` 丢进 `data/<名字>/` 即可。

加载时 `gs_loader.py` 做了这些桥接（两边格式/约定不一致，必须转换）：
- **二进制 .ply 解析**：纯 numpy 手写，零新依赖（环境没有 plyfile/torch）
- **下采样**：按 opacity 加权，把 20 万+高斯降到 1 万（保留实心部分，丢边缘噪点）
- **坐标系**：PhysGaussian 是 **Z-up**，本 demo 是 **Y-up**，自动转轴 + 归一化进 `[0,1]³`
- **量纲重标定**：上游 E 是真实 SI（木头 1e10、钢 2e11），demo 稳定区间是 `[8e3, 2e5]`；
  用 **log-线性映射** 压进稳定区间，同时**保留材质间的软硬相对关系**（果冻 < 木头 < 金属）。
  Server 版里金属/木头的"硬"主要靠**屈服强度 yield**体现，不再靠把 E 拉爆（拉过稳定上限会数值炸）
- **材质映射（server 版已做成 1:1 忠实映射）**：上游 6 种 MPM 本构 → demo 6 种
  （jelly→弹性，plasticine/metal/wood→**真弹塑性 PLASTIC**（按 yield 区分软硬），
  snow→雪塑性，sand→**Drucker-Prager 颗粒**，foam→**可压溃泡沫**，liquid→流体）。
  每种塑性材质还带一对 `(yield, friction)` 参数，让金属几乎不变形、橡皮泥一压一个坑

加载后物体会落到地面、按推断的材质变形，且**所有交互照常可用**（戳、障碍物、切材质、
按 R 重新加载）。面板会显示原始/下采样粒子数和 E 的真实值→demo 值。

> 已训好可直接用的：`chair`（27 万高斯，推断为木质 plasticine）。其他数据集
> （drums/lego/hotdog…）需要先用 `pipeline.py` 跑 3DGS 重建产出 `.ply`。

### 嫌画面闷 / frame 数涨得慢？调粒子数

左上角 `frame` 计数器涨多快 ≈ 实时帧率。gs 场景比程序场景重，粒子越多越慢：

| `--gs-particles` | 本地 mac 实测帧率 | 体感 |
|---|---|---|
| 4000 | ~22 fps | 丝滑，适合交互/录屏 |
| **6000（默认）** | ~12 fps | 流畅，精度/速度平衡 |
| 10000 | ~5 fps | 偏闷，但更密 |
| 12000 | ~4 fps | 最密，适合定格截图 |

> 以上是本地 mac（MoltenVK）的交互帧率。**CUDA 机器**用 `--headless` 离线出片时
> 不受实时帧率限制，`--gs-particles 270000` 跑满全部高斯也没问题。

**这只是下采样率，不影响物理。** 原始点云是 27 万高斯，本地实时跑不了那么多，
下采样是必须的；4000 和 270000 掉落变形的物理完全一样（物理精度由网格 `PG_GRID` 决定，
默认 64），区别只是渲染点的疏密。求流畅用 `--gs-particles 4000`，要展示精度就拉满。

### 已知限制（不是 bug）

1. **目前只有 `chair` 能加载。** 已建了 8 个数据集目录（chair/drums/ficus/hotdog/
   lego/materials/mic/ship），物理参数 `config.json` 都备好了，但只有 chair 训出了
   `.ply` 点云。要加载其他物体，需先跑 `pipeline.py --dataset <名> --material <材质>`。

2. **（dev 版的"软塌摊饼"问题，server 版已修复）** dev 版只有弹性/雪/流体三种本构，
   木质物体被降级成软弹性体，一掉就摊。server 版补了真正的 **von Mises 弹塑性（PLASTIC）**、
   **Drucker-Prager 颗粒（SAND）**、**可压溃泡沫（FOAM）**：物体撞击后会变形然后**保持新形状**，
   金属/木头几乎不变形。chair 现在映射成 plasticine（PLASTIC），落地后**保形不摊**。
   原理见下方"技术原理 ④"。

---

## Headless 离线渲染（server 版新增）

服务器没显示器也能出片：模拟 N 帧 → 每帧渲染成 PNG → ffmpeg 合成 mp4，全程无窗口。
需要 GPU 后端做离屏渲染（CUDA 机器用 `PG_ARCH=cuda`，本地 mac 用 `vulkan`）。

```bash
# 程序场景：橡皮泥球落地，椭球渲染，360 帧 @30fps，1280²
PG_ARCH=cuda conda run -n py310 python main.py \
    --scene ball --preset plasticine \
    --headless --frames 360 --fps 30 --res 1280 --hifi \
    --orbit 0.5 --out plasticine_ball.mp4

# 加载真实 3DGS chair，跑满 27 万高斯出高清片
PG_ARCH=cuda PG_NMAX=300000 conda run -n py310 python main.py \
    --gs chair --gs-particles 270000 \
    --headless --frames 600 --res 1920 --hifi --out chair.mp4
```

headless 相关参数：

| 参数 | 作用 | 默认 |
|---|---|---|
| `--headless` | 开启离屏模式（无窗口） | off |
| `--frames N` | 渲染帧数 | 360 |
| `--fps N` | 输出视频帧率 | 30 |
| `--out path` | 输出 mp4 路径 | out.mp4 |
| `--res N` | 方形渲染分辨率 | 1280 |
| `--orbit R` | 相机环绕速度（rad/秒），0=静止 | 0.5 |
| `--hifi` | 用 F 形变椭球渲染（更忠实、更重） | off（点splat） |

> 没装 ffmpeg 也不报错：会把 PNG 序列留在临时目录并打印路径，自己手动合成即可
> （`ffmpeg -framerate 30 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4`）。

## 逐帧状态导出（严格对齐官方 PhysGaussian）

除了出视频，还能把**每一帧的物理粒子状态**落盘，格式与官方 PhysGaussian
仓库的 `save_data_at_frame`（`mpm_solver_warp/engine_utils.py`）**字段级、命名级、
形状级完全一致**。下游（如 SDS 监督、物理分析、二次渲染）可以用**读官方输出的
同一套代码**直接读本 demo 的输出。

```bash
# 出视频 + 逐帧状态（ply + h5 都出）
PG_ARCH=cuda python main.py --gs chair \
    --headless --frames 360 --res 1280 --hifi --out chair.mp4 \
    --export-state ./chair_sim

# 只要 h5（SDS 下游通常读这个），不出 ply
... --export-state ./out --export-h5

# 程序场景也能导（不止 3DGS 物体）
python main.py --scene ball --export-state ./ball_sim --frames 120
```

> `--export-state` 会自动隐含 `--headless`（导出是离线批处理行为）。

导出参数：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--export-state DIR` | 开启导出，写入 `DIR/simulation_ply/` | off |
| `--export-ply` | 写逐帧 `.ply`（binary xyz） | 未指定格式时两者都出 |
| `--export-h5` | 写逐帧 `.h5`（需要 h5py） | 未指定格式时两者都出 |

输出目录结构与文件格式（**对齐官方**）：

```
DIR/simulation_ply/
├── sim_0000000000.h5   # frame 0 = 初始态，然后 1..N（与官方编号一致）
├── sim_0000000000.ply
├── sim_0000000001.h5
├── ...
└── transform.json      # 额外：sim↔原始 3DGS 世界坐标的还原参数（官方没有）
```

`.h5` 数据集（全部 channels-first，对齐官方的 `.transpose()`）：

| 数据集 | 形状 | 含义 |
|---|---|---|
| `x` | `(3, N)` | 粒子位置（sim 空间，与官方一致，未做反归一化） |
| `v` | `(3, N)` | 速度 |
| `f_tensor` | `(9, N)` | 形变梯度 F（3×3 行主序展平） |
| `C` | `(9, N)` | APIC 仿射速度场（3×3 行主序展平） |
| `time` | `(1, 1)` | 当前 sim 时间 |

`.ply`：`binary_little_endian`，`element vertex N`，仅 `float x/y/z`，与官方
`particle_position_to_ply` 字节布局一致。

> **坐标系说明**：官方 `save_data_at_frame` 存的是 **sim 空间**坐标（归一化、未
> 做 inverse transform）——本 demo 严格照此处理。若下游需要原始 3DGS 世界坐标
> （Z-up、真实尺度），用 `transform.json` 里的 `normalization`（axis_swap /
> center / extent / fill_frac / offset）逆变换即可。程序场景（非 3DGS 加载）
> 没有归一化，`normalization` 为 `null`。

> **依赖**：`--export-h5` 需要 `h5py`（官方同款依赖）。`pip install h5py`。
> 只用 `--export-ply` 则无需额外依赖。

> **不影响视频输出**：导出是渲染管线之外的纯增量步骤，不传 `--export-state`
> 时所有导出分支跳过，行为与改动前完全一致；带与不带导出，生成的 PNG/视频帧
> 逐帧相同。

## 部署到 CUDA 机器（环境变量）

本地 mac（MoltenVK）和服务器（CUDA）能力差很多，几个关键开关做成了环境变量，
**不用改代码**就能在大机器上放开规模：

| 环境变量 | 作用 | 默认 | 说明 |
|---|---|---|---|
| `PG_ARCH` | Taichi 后端 | 自动（cuda→vulkan→cpu） | 服务器显式设 `cuda` |
| `PG_NMAX` | 最大粒子数（显存上限） | 300000 | 已默认放开到 27万+ |
| `PG_GRID` | MPM 网格分辨率（物理精度） | 64 | CUDA 上可设 128 |
| `PG_SUBSTEP` | 每渲染帧的子步数 | 24 | — |
| `PG_ICO_SUBDIV` | 椭球细分级（0/1/2 → 12/42/162 顶点） | 0 | 高分辨率出片设 1 |
| `PG_HIFI` | 启动即用椭球渲染 | off | 等价 `--hifi` |
| `PG_NOVSYNC` | 关垂直同步（跑分用） | off | — |

```bash
# CUDA 机器典型配置
export PG_ARCH=cuda PG_NMAX=300000 PG_GRID=128
conda run -n py310 python main.py --gs chair --gs-particles 270000 --hifi
```

> `PG_GRID` 拉到 128 物理更细但 p2g/g2p 更慢；显存够再开。粒子数 `--gs-particles`
> 和 `PG_NMAX` 要配套（前者 ≤ 后者）。

---

## 操作说明

| 操作 | 按键 / 鼠标 |
|---|---|
| 旋转视角 | 右键拖拽（RMB） |
| 缩放 | W / S |
| 戳 / 推物体 | 左键拖拽（LMB） |
| 暂停 / 继续 | 空格 |
| 重置当前场景 | R |
| 切换材质 | 数字键 1-9，**或点面板按钮** |
| 切换场景 | B / X / M / C / T |
| 重力 减 / 加 | G / H |
| 点 / 椭球渲染切换 | F |
| 障碍物 开 / 关 | O，**或点面板按钮** |
| 移动障碍物（水平） | 方向键 |
| 障碍物 上 / 下 | U / N |
| 退出 | ESC |

> ⚠️ **数字键/字母键没反应时**：是 ImGui 面板抢走了键盘焦点。
> 解决办法：用鼠标点一下窗口里的 3D 场景空白处（不是面板），再按键；
> 或者直接点面板里的 **材质按钮 / 障碍物按钮**，按钮不受焦点影响。

---

## 场景（5 种）

每个场景都是程序生成的高斯粒子云（不依赖真实 3DGS 训练结果）。

| 按键 | 场景 | 说明 |
|---|---|---|
| B | `ball` | 球（默认） |
| X | `box` | 方块 |
| M | `multi` | 双球（左右分开，落地不重叠；右球用当前材质，左球固定果冻） |
| C | `cylinder` | 圆柱 |
| T | `torus` | 圆环 |

## 材质（9 种 preset）

| 按键 | preset | 模型 | E | ν | yield / fric | 行为 |
|---|---|---|---|---|---|---|
| 1 | `jelly`  | 弹性 (fixed-corotated) | 3.0e4 | 0.30 | — | 果冻，软、会回弹 |
| 2 | `rubber` | 弹性 | 6.0e4 | 0.42 | — | 橡胶，硬、弹性强 |
| 3 | `putty`  | 弹性 | 2.0e4 | 0.48 | — | 软泥，近不可压、慢回弹 |
| 4 | `snow`   | 雪塑性 (corotated + plasticity) | 8.0e4 | 0.20 | — | 雪，撞击留永久凹坑 |
| 5 | `liquid` | 弱可压缩流体 | 8.0e3 | 0.40 | — | 液体，瘫流、绕障碍分流 |
| 6 | `plasticine` | **弹塑性 (von Mises)** | 8.0e4 | 0.30 | yield 2.5e3 | 橡皮泥，一压一个坑、**保形** |
| 7 | `metal`  | **弹塑性 (von Mises)** | 1.2e5 | 0.35 | yield 3.0e4 | 金属，硬、几乎不变形 |
| 8 | `sand`   | **颗粒 (Drucker-Prager)** | 6.0e4 | 0.30 | fric 3.0 | 沙子，堆成沙堆、不抗拉 |
| 9 | `foam`   | **可压溃泡沫** | 1.2e4 | 0.10 | yield 6.0e2 | 泡沫，被压扁后**永久压实** |

> 还有一个 `wood`（PLASTIC, E 1.0e5, yield 1.5e4），没绑数字键，可在 gs 加载或
> `--preset wood` 用。
>
> 面板新增 `yield` / `sand fric` 两个滑条：`yield` 越大越硬（金属几乎不屈服），
> `sand fric` 越大沙堆越陡、越不"流"。只对对应材质生效。
>
> `Young E` / `Poisson` 滑条只调弹性参数，**改不出 liquid / 颗粒行为**——
> 那些是另一套本构，必须用按钮或数字键切换。

## 障碍物（碰撞代理）

按 `O` 开启一个**可移动的球形障碍物**（金黄色）。它作为碰撞代理，让高斯软体能与之
交互——物体砸上去会变形、绕流；用方向键 / U / N 把球推进物体里能看到实时挤压形变。
面板里也有 `obs X/Y/Z/radius` 滑条精确控制位置和大小。

这对应 ideas.md 的 **Idea 3（3DGS 碰撞代理）**：3DGS 没有天然碰撞面，这里用
"网格速度边界条件"作为隐式 collider，让视觉表示和物理交互统一在 MPM 网格上。

---

## 技术原理（怎么对应论文）

**① 同一批高斯既渲染又仿真。** 每个粒子（=高斯）携带：位置 `x`、速度 `v`、
形变梯度 `F`、APIC 仿射矩阵 `C`、材质 id、颜色。位置 `x` 既喂给物理求解、又直接喂给渲染，
没有第二套表示。

**② MLS-MPM 驱动。** 每帧跑 `SUBSTEP` 次子步，每个子步是标准三段：
```
clear_grid → p2g（粒子→网格，撒动量+应力）
           → grid_op（网格上加重力、解边界/碰撞）
           → g2p（网格→粒子，插值回速度、更新 F 和位置）
```

**③ 形变梯度 F 传输高斯形状（论文精髓）。** 高斯的形状由协方差 Σ 决定，
物理算出 `F` 后用 **Σ_t = F · Σ₀ · Fᵀ** 更新它。代码里体现为：椭球模式下单位球顶点被
`F` 拉成各向异性椭球（`x + F·(s·dir)`，法线用 `F⁻ᵀ·dir`）。物体被压扁/拉长时，
那块的高斯椭球就跟着压扁/拉长——**仿真形变直接成为渲染形变**。

**④ 六种本构模型。** 应力在 `kirchhoff_stress` 里算（弹性/雪/流体三支），
**塑性靠 g2p 里的"回映射"(return mapping) 实现**——这是 server 版保形的关键：

- **PLASTIC（plasticine/metal/wood）**：在 Hencky（对数）应变空间做 **von Mises 回映射**。
  `F` 存的是**弹性**形变梯度；超过屈服面 `yield` 的偏量应变被当作**永久塑性流动**剥离掉。
  结果：物体被撞会变形，然后**保持新形状**（不像弹性那样回弹、也不像旧版那样摊饼）。
  金属 vs 橡皮泥只差在 `yield` 大小（金属屈服面大到几乎不屈服）。
- **SAND**：**Drucker-Prager 回映射**（无黏聚颗粒，Klár 2016）。沙子按摩擦角堆成沙堆、
  不能受拉（受拉直接投影到锥尖 → 散开）。
- **FOAM**：低屈服的弹塑性，体积被压实后永久保持（可压溃泡沫）。

> 为什么硬度不靠 E：本 demo 的显式积分在 E 超过 ~2e5 会违反 CFL 稳定条件而数值炸
> （金属一开始设 E=2e5 实测直接 NaN）。所以"金属比橡皮泥硬"是用**更大的屈服强度**表达的，
> E 都压在稳定区间内。这点在调参时很关键。

---

## 渲染：两种模式（按 F 切换）

| | 点模式（默认） | 椭球模式（按 F） |
|---|---|---|
| 每个高斯画成 | 各向同性圆点 | 被 F 拉伸的椭球 |
| 速度 | 快（~30+ fps） | 慢（三角形 overdraw 重） |
| 能否看出各向异性形变 | 看不出 | **能看出** |

> macOS（MoltenVK）上完整椭球网格因为重叠三角形 overdraw 极慢（实测比点慢约 70×），
> 所以**默认用点模式**保证流畅。要展示"高斯被 F 拉成各向异性椭球"这个论文卖点时，
> 按 `F` 切到椭球模式定格观察。
>
> **注意**：点模式下物体下落、碰撞、形变的**物理一直在真实计算**（`F` 始终在更新），
> 只是没把椭球形状画出来。点模式仍然是 PhysGaussian。

**关于"一颗颗粒子感"**：真实 3DGS 是半透明、α 混合、连续表面的。本复现用不透明
点/椭球近似，所以能看到一颗颗——这是课程范围内的合理简化（ideas.md 也写明
"渲染可以先用 point splat / 简化 ellipsoid，不必完整复现 CUDA 3DGS rasterizer"）。
做真 α-blending 主要受限于 Taichi GGUI 不暴露自定义 splatting 管线，而非算力。
想让画面更"实心"：把面板 `size` 滑条调大，让点互相盖住缝隙即可。

---

## 常见现象答疑

**物体被撞会"瘪"——正常。** 这是弹塑性软体的正确行为：jelly/rubber 瘪后会回弹，
snow 会留下永久凹坑（塑性），liquid 本就没固定形状。

**演示建议。** 平时用点模式跑流畅；要展示"形变各向异性"时按 `F` 切椭球定格；
切 `liquid` + 开 `O` 障碍物，能看到流体绕球分流，效果最直观。

---

## 项目结构

```
physgaussian_server/          # ← 本目录（改动版，部署用）
├── main.py        # 全部逻辑：MPM 求解 + 回映射塑性 + 渲染 + 交互 + headless
├── gs_loader.py   # 3DGS .ply / config.json 的解析与材质/量纲桥接
├── run.sh         # 启动脚本
└── README.md      # 本文件
../physgaussian/               # ← dev 原版存档，未改动
```

`main.py` 关键函数：
- `kirchhoff_stress` — 弹性/雪/流体的应力模型
- `g2p` 里的 return mapping — **von Mises（PLASTIC/FOAM）+ Drucker-Prager（SAND）塑性**（server 新增）
- `p2g` / `grid_op` / `g2p` — MLS-MPM 三段；障碍物碰撞在 `grid_op` + `g2p` 里
- `fill_sphere/box/cylinder/torus` — 程序生成各场景的粒子
- `apply_preset` — 运行时切材质（同时设 E/ν/yield/fric 并重标记粒子）（server 新增）
- `build_points` / `build_mesh` — 点 / 椭球两种渲染
- `render_headless` — 离屏渲染 PNG 序列 + ffmpeg 合成 mp4（server 新增）
- `main` — 主循环（事件、仿真、绘制、面板）

---

