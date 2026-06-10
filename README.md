
## SupGaussian


通过可导的渲染，支持参数优化
> 本代码基于 Physics3D 原始的代码库进行修改
> 添加的功能：
> 1. 修正原本代码无法运行的问题（mpm 会爆，没有提供样例 gaussian model），按照自己的理解修改优化逻辑
> 2. 支持参数可选优化和更大的范围的优化
> 3. 支持根据颜色将原始的 Gaussian 进行聚类，在此基础上每个类可以进行优化


### 环境安装

```bash
conda create -n Physics3D python=3.9
conda activate Physics3D

pip install -r requirements.txt

# 先确保 torch 已安装；这两个 CUDA 扩展在构建阶段会直接 import torch
# 这里 requirements.txt 固定的是 torch==2.0.0 / torchvision==0.15.1，建议配套 CUDA 11.7
# 如果是新环境，建议先执行：pip install torch torchvision

# 安装 CUDA 11.7 编译链，建议使用 NVIDIA 的 11.7 label，避免 solver 拉到 12.x
# 如果 conda 下载一直卡在 0%，优先改用 mamba；cuda-cccl 提供 cuda/std 头文件
conda install -c nvidia/label/cuda-11.7.0 cuda-compiler=11.7.0 cuda-nvcc=11.7.64 cuda-cudart-dev=11.7.60 cuda-libraries-dev=11.7.0 cuda-cccl=11.7.58

# 让当前 shell 使用 conda 里的 CUDA
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 如果 cuda_runtime.h 仍然找不到，把 nvidia runtime 头文件目录加入 include path
export CPATH=$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/include:$CPATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/include:$CPLUS_INCLUDE_PATH

# cuda/std/type_traits 来自 cuda-cccl，补上它的 include 根目录
export CPATH=$CONDA_PREFIX/include/cccl:$CPATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/cccl:$CPLUS_INCLUDE_PATH

# 先检查一下：which nvcc && nvcc --version

git clone https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting
git submodule init

# diff-gaussian-rasterization 里还有嵌套子模块 glm，需要递归初始化
git submodule update --init --recursive

# 在 gaussian-splatting 目录内安装子模块，并关闭 build isolation
pip install -e submodules/diff-gaussian-rasterization/ --no-build-isolation
pip install -e submodules/simple-knn/ --no-build-isolation
```


### 算力要求：
最原始的配置：

>24G 4090: 由于需要 load diffusion model 会非常受限🚫
>stage_num = 4; frame_per_stage = 4; 同时

> 48G 4090: 可以支持更加多的参数和积累

### 运行代码

```bash
cd /home/featurize/work/SupGaussian
XDG_CACHE_HOME=$PWD/.cache \
HF_HOME=$PWD/.cache/huggingface \
TRANSFORMERS_CACHE=$PWD/.cache/huggingface/transformers \
python simulation.py \
    --model_path /home/featurize/work/gaussian_model/ficus_whitebg-trained \
    --prompt "a plant in the wind" \
    --output_path ./output \
    --physics_config ./config/ficus_config.json

```




