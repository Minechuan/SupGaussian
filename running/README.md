# 3DGS → PhysGaussian 工具链使用指南

## 环境要求

```bash
conda activate forCVGCoding
```

确保以下路径存在：
- `./running/` — 本工具脚本目录
- `./nerf_synthetic/` — NeRF 合成数据集（含 chair, drums, ficus, hotdog, lego, materials, mic, ship）
- `./model/Qwen2.5-VL-3B-Instruct/` — Qwen 视觉模型（可选，用于自动材料识别）
- `./PhysGaussian/` — PhysGaussian 代码仓库

---

## 文件说明

| 文件 | 用途 |
|---|---|
| `pipeline.py` | **主入口**：一键完成 3DGS 训练 + 生成 config.json |
| `train_3dgs.py` | 调用 PhysGaussian 的 train.py 进行 3DGS 训练 |
| `generate_config.py` | 生成 PhysGaussian 仿真所需的 config.json |
| `material_database.py` | 83 种材料的物理参数库（E, nu, density 等） |
| `run_simulation.py` | 运行 PhysGaussian MPM 物理仿真 |
| `filter_ply.py` | 按不透明度过滤 PLY 噪声点 |
| `infer_physics.py` | 用 Qwen2.5-VL 从图片自动识别材料类型 |

---

## 快速开始

### 1. 生成 PLY + 配置（以 chair 为例）

```bash
cd running

# 训练 3DGS（30000 轮）+ 生成 config.json
python pipeline.py --dataset chair --material wood
```

**参数说明：**
- `--dataset`：`nerf_synthetic/` 下的数据集名（chair, drums, lego…）
- `--material`：材料名，见下方材料表；不指定则自动用 Qwen 图片识别
- `--iterations`：训练轮数（默认 30000）
- `--skip-train`：跳过训练，只生成 config
- `--list-materials`：列出全部 83 种可用材料

### 2. 查看可用材料

```bash
python pipeline.py --list-materials
```

### 3. 运行物理仿真

```bash
python run_simulation.py --object_id chair
```

仿真输出在 `output/chair/sim_output/`：
- `0000.png ~ 0099.png`：100 帧渲染图
- `simulation_ply/`：101 帧粒子 PLY 文件

### 4. 过滤 PLY（可选）

```bash
python filter_ply.py
```

---

## 常用材料速查

| 材料名 | 类型 | 适用物体 |
|---|---|---|
| `wood` | 木头 | chair, table, ficus |
| `metal` | 金属 | drums, mic, ship |
| `hard_plastic` | 硬塑料 | lego, 玩具 |
| `plasticine` | 橡皮泥 | 雕塑, clay |
| `ceramic` | 陶瓷 | vase, cup |
| `glass` | 玻璃 | 瓶子, 窗户 |
| `rubber` | 橡胶 | 轮胎, 软玩具 |
| `cloth` | 布料 | 衣物, 沙发 |
| `leather` | 皮革 | 鞋子, 包 |
| `meat` | 肉 | hotdog, 食物 |
| `fruit` | 水果 | 苹果, 香蕉 |
| `bread` | 面包 | 蛋糕, 面包 |
| `snow` | 雪 | 雪人, 冰淇淋 |
| `sand` | 沙子 | 沙堆, 土壤 |

---

## 各数据集推荐材料

| 数据集 | 推荐材料 | 命令 |
|--------|----------|------|
| chair | wood | `python pipeline.py --dataset chair --material wood` |
| drums | metal | `python pipeline.py --dataset drums --material metal` |
| ficus | wood | `python pipeline.py --dataset ficus --material wood` |
| hotdog | meat | `python pipeline.py --dataset hotdog --material meat` |
| lego | hard_plastic | `python pipeline.py --dataset lego --material hard_plastic` |
| materials | metal | `python pipeline.py --dataset materials --material metal` |
| mic | metal | `python pipeline.py --dataset mic --material metal` |
| ship | metal | `python pipeline.py --dataset ship --material metal` |

> 不指定 `--material` 时会自动用 Qwen2.5-VL 从图片推理材料。已推理结果见 `inferred_materials.json`。

---

## chair 已有产出

```
output/chair/
├── point_cloud/iteration_7000/point_cloud.ply   ← 推荐（PSNR 35.94 dB）
├── point_cloud/iteration_30000/point_cloud.ply  ← 备用
└── config.json                                   ← 木材质配置
```

---

## Qwen 自动材料识别（可选）

```bash
python infer_physics.py --object chair
```

需要 `./model/Qwen2.5-VL-3B-Instruct/` 下的模型文件。首次加载约 30 秒，单张图推理约 80 秒。
