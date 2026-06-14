# 快速上手

PhysGaussian 复现 demo · 用 Taichi 把 3D 高斯当物理粒子跑 MLS-MPM 软体仿真。
拿到这个文件夹后，按下面三步就能跑。

---

## 1. 装环境（只做一次）

需要 conda + Python 3.10。在本文件夹里：

```bash
conda create -n py310 python=3.10 -y
conda activate py310
pip install "taichi==1.7.4" numpy
# 想导出 mp4 视频再装 ffmpeg（不装也能跑，只是出 PNG 序列）：
conda install -c conda-forge ffmpeg -y
```

> 用别的环境名也行，跑脚本时加 `PG_ENV=<环境名>` 即可。

---

## 2. 跑起来

脚本会自动选后端（Mac 走 Vulkan，有 N 卡的机器自动用 CUDA），**无需配置**。

### A. 交互模式（开窗口，最直观）
```bash
./run_local.sh                 # 加载椅子（真实 3DGS 重建），默认材质
./run_local.sh chair metal     # 椅子当金属，腿基本不弯
./run_local.sh ball jelly      # 程序生成的球，果冻材质
```
窗口里：右键拖=转视角，左键拖=戳物体，数字键 **1-9 切材质**，`F` 切椭球渲染，`R` 重置，`ESC` 退出。
（数字键没反应就用鼠标点一下 3D 场景空白处，或直接点左上角面板按钮。）

### B. 导出视频（不开窗口）
```bash
./render.sh                    # 椅子 -> chair.mp4
./render.sh chair metal        # 金属椅子 -> chair.mp4
./render.sh ball sand out.mp4  # 沙子球 -> out.mp4
```
调画质用环境变量：`PG_RES=1080 PG_PARTICLES=80000 PG_HIFI=1 ./render.sh chair`
（`PG_HIFI=1` 开椭球渲染，更忠实但更慢。）

### C. 不用真实椅子，纯看物理
```bash
./run_local.sh box snow        # 方块、雪
./run_local.sh torus liquid    # 圆环、液体
```
程序场景：`ball / box / multi / cylinder / torus`，任何机器都能立刻跑。

---

## 3. 9 种材质手感

数字键或脚本第二个参数指定：

| 键 | 材质 | 手感 |
|---|---|---|
| 1 | jelly | 果冻，软、**回弹** |
| 2 | rubber | 橡胶，硬、强回弹 |
| 3 | putty | 软泥 |
| 4 | snow | 雪，撞击留凹坑 |
| 5 | liquid | 液体，瘫流 |
| 6 | plasticine | 橡皮泥，一压一个坑、保形 |
| 7 | metal | 金属，几乎不变形 |
| 8 | sand | 沙子，散开堆沙堆 |
| 9 | foam | 泡沫，压扁后永久压实 |

> 椅子默认是 plasticine（来自它的 config），所以**落地腿会弯、不回弹**——这是橡皮泥的正确行为。
> 想看腿硬挺：`./run_local.sh chair metal`。想看回弹：`./run_local.sh chair jelly`。

---

## 常见问题

- **`cannot initialize GLFW`**：在没有显示器的服务器上跑了交互模式。服务器只能用 `./render.sh` 出视频。
- **看到的是球不是椅子**：椅子数据没找到，程序回退成默认球。终端会打印它找过哪些路径——确认 `data/chair/point_cloud.ply` 在不在。
- **CUDA out of memory**（大机器跑高粒子数）：降 `PG_PARTICLES`（如 40000）或 `PG_RES`（如 720）。
- **想换自己的 3DGS 物体**：把 `point_cloud.ply` + `config.json` 放进 `data/<名字>/`，然后 `./run_local.sh <名字>`。或直接 `--gs /绝对路径/point_cloud.ply`。

更详细的原理、参数、论文对应关系见 `README.md`。
