# 第三部分

## 摘要

本项目从静态 3D Gaussian Splatting 场景出发，通过可微物理仿真和视频扩散模型引导，自动优化物体的动态物理参数。系统首先读取预训练 Gaussian 场景，将可运动区域转换为 MPM 粒子；随后在 MPM 中模拟运动，把粒子状态重新映射为 Gaussian 的位置、协方差和旋转；最后渲染为视频，并利用文本条件 video diffusion model 计算 guidance loss，将梯度反向传播到物理参数。

整体目标可以写为：

$$
\min_{\theta}\ \mathcal{L}_{\mathrm{SDS}}
\left(
\mathrm{Render}(\mathrm{MPM}_{\theta}(\mathcal{G})), y
\right),
$$

其中 $\mathcal{G}$ 表示 3D Gaussian 场景，$y$ 是文本提示词，$\theta$ 是待优化材料参数，例如 Young's modulus $E$、Poisson ratio $\nu$、粘弹性参数 $\mu_N,\lambda_N$ 和 viscosity。当前实现还根据 Gaussian 颜色进行聚类，使不同外观区域拥有独立材料参数，从而支持非均匀材料优化。

## 1. 系统输入与调用链

主程序是 `simulation.py`。运行时主要输入包括：预训练 3D Gaussian 模型路径 `model_path`，物理配置文件 `physics_config`，文本提示词 `prompt`，输出路径 `output_path`，以及 diffusion guidance 配置 `guidance_config`。

系统调用链可以概括为：

1. 读取 3D Gaussian checkpoint；
2. 根据 opacity 和用户指定区域筛选可运动 Gaussian；
3. 将场景旋转、平移并归一化到 MPM 仿真空间；
4. 根据 Gaussian 颜色做 k-means 聚类；
5. 可选地执行 particle filling，补充体粒子；
6. 初始化 MPM solver，并将 cluster 参数广播到粒子；
7. 进行可微 MPM rollout；
8. 将粒子状态转换回 Gaussian 状态并渲染视频；
9. 使用 video diffusion guidance 计算 SDS loss；
10. 将梯度从图像反传到 MPM 状态，再反传到材料参数；
11. 按 cluster 聚合梯度并更新各区域材料参数。

## 2. SDS Guidance

Score Distillation Sampling 的作用是：不训练扩散模型本身，而是固定扩散模型参数 $\phi$，利用其文本条件先验优化当前渲染结果 $x=g(\theta)$。给定噪声时间步 $t$ 和随机噪声 $\epsilon$：

$$
x_t=\alpha_t x+\sigma_t\epsilon,\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

扩散模型预测噪声 $\epsilon_\phi(x_t;y,t)$。SDS 对渲染结果的梯度近似为：

$$
\nabla_x\mathcal{L}_{\mathrm{SDS}}
=
\mathbb{E}_{t,\epsilon}
\left[
w(t)
\left(
\epsilon_\phi(x_t;y,t)-\epsilon
\right)
\frac{\partial x_t}{\partial x}
\right].
$$

由于

$$
\frac{\partial x_t}{\partial x}=\alpha_t,
$$

该梯度会鼓励当前渲染视频朝着文本提示词对应的视频分布移动。进一步对物理参数求导得到：

$$
\nabla_{\theta}\mathcal{L}_{\mathrm{SDS}}
=
\nabla_x\mathcal{L}_{\mathrm{SDS}}
\frac{\partial x}{\partial\theta}.
$$

这里 $\partial x/\partial\theta$ 由 Gaussian rasterizer 和 differentiable MPM 共同提供。

## 3. 3D Gaussian 表示与渲染

### 3.1 Gaussian 状态

3D Gaussian Splatting 将场景表示为一组各向异性 Gaussian：

$$
G_i=(\mathbf{x}_i,\Sigma_i,\alpha_i,\mathbf{c}_i),
$$

其中 $\mathbf{x}_i$ 是中心位置，$\Sigma_i$ 是协方差矩阵，$\alpha_i$ 是不透明度，$\mathbf{c}_i$ 是由 spherical harmonics 表示的视角相关颜色。程序从 `point_cloud.ply` 中读取这些属性，并根据 opacity threshold 去除贡献较低的 Gaussian，以减少噪声和计算量。

### 3.2 可微渲染

每一帧渲染时，MPM 给出的粒子位置、协方差和旋转会被转换为 Gaussian rasterizer 的输入。颜色由 spherical harmonics 根据相机方向计算：

$$
\mathbf{c}_i(\mathbf{d}_i)=\mathrm{SH}_i(\mathbf{d}_i),
\qquad
\mathbf{d}_i=
\frac{\mathbf{x}_i-\mathbf{o}_{\mathrm{cam}}}
{\|\mathbf{x}_i-\mathbf{o}_{\mathrm{cam}}\|}.
$$

若 MPM 输出局部旋转 $R_i$，方向会被变换到局部坐标系，从而保持颜色与物体局部朝向的一致性。由于 rasterizer 可微，SDS loss 对图像的梯度可以继续传回 Gaussian 状态。

## 4. 从 Gaussian 到 MPM 粒子

### 4.1 坐标与协方差变换

原始 Gaussian 坐标通常处于重建场景的世界坐标系中。为了让 MPM 求解稳定，代码先将其旋转、平移并缩放到规范仿真空间；渲染时再执行逆变换回到原世界空间。

若旋转矩阵为 $R$，尺度因子为 $s$，协方差变换为：

$$
\Sigma' = s^2 R\Sigma R^\top.
$$

在物理仿真过程中，形变梯度 $F_t$ 会继续更新协方差：

$$
\Sigma_t = F_t\Sigma_0F_t^\top.
$$

因此，Gaussian 不仅会随粒子位置移动，也会随局部形变发生拉伸和旋转。

### 4.2 Particle Filling

3D Gaussian 通常主要分布在物体表面，而 MPM 需要体粒子来描述连续体。启用 particle filling 后，系统会在物体内部补充粒子，以获得更稳定的质量分布和体积响应。新增粒子继承最近 Gaussian 的 cluster 标签，因此后续仍然可以使用同一套区域材料参数。

## 5. MPM 物理模型

### 5.1 MPM 基本流程

Material Point Method 使用粒子携带质量、速度和形变状态，同时借助背景网格完成动量交换。一个时间步包含 P2G、网格更新和 G2P 三个阶段。P2G 将粒子质量、动量和应力传到网格；网格阶段加入重力并施加边界条件；G2P 再把网格速度插值回粒子，更新位置、速度和形变梯度。

### 5.2 APIC 传输

代码使用 APIC 风格的粒子-网格传输。P2G 中，粒子 $i$ 对网格节点 $g$ 的贡献为：

$$
m_g \mathrel{+}= w_{ig}m_i,
\qquad
\mathbf{p}_g \mathrel{+}=
w_{ig}m_i
\left(
\mathbf{v}_i+C_i(\mathbf{x}_g-\mathbf{x}_i)
\right)
\Delta t\,\mathbf{f}_i.
$$

其中 $w_{ig}$ 是 B-spline 插值权重，$C_i$ 是 APIC affine velocity matrix。G2P 阶段更新为：

$$
\mathbf{v}_i^{t+1}=\sum_g w_{ig}\mathbf{v}_g,\qquad
\mathbf{x}_i^{t+1}=\mathbf{x}_i^t+\Delta t\,\mathbf{v}_i^{t+1},
$$

$$
F_i^{\mathrm{trial}}=
\left(I+\Delta t\,\nabla\mathbf{v}_i\right)F_i^t.
$$

APIC 相比 PIC 更能保留局部旋转和角动量信息，适合模拟柔性物体的连续运动。

### 5.3 弹性模型

配置文件中的 $E$ 和 $\nu$ 先被转换为 Lamé 参数：

$$
\mu=\frac{10^7E}{2(1+\nu)},\qquad
\lambda=
\frac{10^7E\nu}{(1+\nu)(1-2\nu)}.
$$

其中 $\mu$ 是剪切模量，控制材料抵抗剪切形变的能力；$\lambda$ 主要影响体积压缩和膨胀响应。系数 $10^7$ 是代码中的工程尺度因子，用于把配置中的无量纲优化变量映射到 MPM 内部力学尺度。

对 jelly 材料，系统采用 fixed corotated elasticity，其 Kirchhoff stress 为：

$$
\tau_{\mathrm{elastic}}
=
2\mu(F-R)F^\top+\lambda J(J-1)I,
$$

其中 $F$ 是形变梯度，$R$ 是 $F$ 极分解得到的旋转部分，$J=\det(F)$，$I$ 是单位矩阵。

### 5.4 粘弹性模型

除普通弹性项外，代码还维护额外形变分量 $F_N$，用 $\mu_N$、$\lambda_N$ 和 viscosity 描述粘弹性响应。对 trial 形变做 SVD：

$$
F_N^{\mathrm{trial}}=U_N\operatorname{diag}(\sigma_N)V_N^\top,
\qquad
\epsilon_{\mathrm{trial}}=\log\sigma_N.
$$

随后在 logarithmic strain 空间中更新：

$$
\alpha=\frac{2\mu_N}{\eta},
\qquad
\beta=
\frac{2(2\mu_N+3\lambda_N)}{9\eta}
-
\frac{2\mu_N}{3\eta},
$$

$$
A=\frac{1}{1+\Delta t\,\alpha},
\qquad
B=
\frac{\Delta t\,\beta}
{1+\Delta t(\alpha+3\beta)},
$$

$$
\epsilon_N=
A
\left(
\epsilon_{\mathrm{trial}}
-
B\,\mathrm{tr}(\epsilon_{\mathrm{trial}})\mathbf{1}
\right),
$$

其中 $\eta$ 表示 viscosity。新的粘弹性形变为：

$$
F_N=U_N\operatorname{diag}(\exp\epsilon_N)V_N^\top.
$$

粘弹性应力写为：

$$
\tau_N=
2\mu_N\epsilon_N
+
\lambda_N\mathrm{tr}(\epsilon_{\mathrm{trial}})\mathbf{1}.
$$

最终应力由弹性项和粘弹性项共同构成：

$$
\tau=\tau_{\mathrm{elastic}}+\tau_N.
$$

### 5.5 优化参数的物理意义

当前实现中主要涉及五个物理参数：

$$
\theta=(E,\nu,\mu_N,\lambda_N,\eta).
$$

$E$ 是 Young's modulus，描述材料抵抗拉伸和压缩的能力。$E$ 越大，材料越硬，受力后形变越小；$E$ 越小，材料越软，更容易弯曲、塌陷或产生较大位移。

$\nu$ 是 Poisson ratio，描述材料被拉伸时横向收缩与纵向伸长的比例。$\nu$ 越接近 $0.5$，材料越接近不可压缩，此时 $\lambda$ 会快速增大。因此代码会将 $\nu$ 限制在安全范围内，避免数值奇异。

$\mu_N$ 和 $\lambda_N$ 是粘弹性分量中的类 Lamé 参数。$\mu_N$ 主要控制粘弹性剪切响应，$\lambda_N$ 主要控制粘弹性体积响应。它们不替代 $E$ 和 $\nu$，而是作用在额外形变 $F_N$ 上，用来表达材料的迟滞、内部阻抗和动态回弹。

viscosity $\eta$ 控制粘弹性形变的松弛速度。$\eta$ 较大时，$\alpha$ 和 $\beta$ 较小，历史形变保留更久，材料表现出更强阻尼和迟滞；$\eta$ 较小时，粘弹性形变衰减更快，响应更接近普通弹性。

在代码使用上，$E$ 和 $\nu$ 先转换为 $\mu,\lambda$ 并影响 $\tau_{\mathrm{elastic}}$；$\mu_N,\lambda_N$ 决定 $\tau_N$ 的强度；viscosity 决定 $\epsilon_N$ 的时间更新。配置文件中的 `param` 字段决定哪些参数参与优化，例如只优化 $E,\nu,\eta$ 时，$\mu_N$ 和 $\lambda_N$ 会保持初始值。

## 6. Video Diffusion Guidance

`ModelscopeGuidance` 加载 text-to-video / stable diffusion 风格模型，主要包含 VAE、UNet、DDIM scheduler 和 CLIP text encoder。VAE 将渲染视频编码为 latent，UNet 在扩散时间步上预测噪声，text encoder 将 prompt 转为文本条件。

给定渲染视频 $x$，VAE 编码得到 latent：

$$
z=\mathrm{VAE}(x).
$$

扩散过程采样：

$$
z_t=\sqrt{\bar{\alpha}_t}z+
\sqrt{1-\bar{\alpha}_t}\epsilon.
$$

使用 classifier-free guidance 后的噪声预测为：

$$
\hat{\epsilon}
=
\epsilon_{\mathrm{uncond}}
+
s(\epsilon_{\mathrm{text}}-\epsilon_{\mathrm{uncond}}),
$$

其中 $s$ 是 guidance scale。SDS 在 latent 空间中的梯度为：

$$
\nabla_z\mathcal{L}_{\mathrm{SDS}}
=
w(t)(\hat{\epsilon}-\epsilon).
$$

该梯度经 VAE decoder / encoder 链路传到 RGB 帧，再经 Gaussian rasterizer 和 MPM 传到物理参数。

`low_ram_vae` 用于降低显存占用。若一个 stage 有 16 帧，但只保留少数帧的 VAE 反向图，则其他帧仍参与视频上下文，但不直接提供完整 VAE 梯度。这是一种显存和梯度完整性之间的折中。

## 7. 梯度反传路径

整体反传链路为：

$$
\theta
\rightarrow
\mathrm{MPM}
\rightarrow
(\mathbf{x},\Sigma,R)
\rightarrow
\mathrm{Rasterizer}
\rightarrow
x
\rightarrow
\mathrm{VAE}
\rightarrow
\mathcal{L}_{\mathrm{SDS}}.
$$

对应梯度为：

$$
\frac{\partial\mathcal{L}}{\partial\theta}
=
\frac{\partial\mathcal{L}}{\partial x}
\frac{\partial x}{\partial(\mathbf{x},\Sigma,R)}
\frac{\partial(\mathbf{x},\Sigma,R)}{\partial \mathrm{MPM}}
\frac{\partial \mathrm{MPM}}{\partial\theta}.
$$

实现上，PyTorch autograd 负责从图像 loss 反传到粒子位置、协方差和旋转；Warp tape 再把这些粒子状态梯度作为终端梯度，穿过 MPM 时间积分，得到 $E,\nu,\mu_N,\lambda_N,\eta$ 等参数的梯度。

## 8. 基于颜色聚类的非均匀材料优化

### 8.1 聚类动机

若所有粒子共享一组材料参数，系统只能表达均质物体。但真实对象往往由不同部件组成，例如叶片、枝干和花盆具有不同硬度与阻尼。为此，当前实现使用 Gaussian 颜色进行 k-means 聚类，并为每个 cluster 维护独立材料参数。

### 8.2 颜色特征与 K-means

系统使用 spherical harmonics 的 DC 分量作为颜色特征，并进行 min-max normalization：

$$
\tilde{\mathbf{c}}_i=
\frac{\mathbf{c}_i-\min(\mathbf{c})}
{\max(\mathbf{c})-\min(\mathbf{c})+\varepsilon}.
$$

k-means 求解目标为：

$$
\min_{\{\ell_i\},\{\mathbf{m}_j\}}
\sum_i
\left\|
\tilde{\mathbf{c}}_i-\mathbf{m}_{\ell_i}
\right\|^2,
$$

其中 $\ell_i$ 是第 $i$ 个 Gaussian 的 cluster 标签，$\mathbf{m}_j$ 是第 $j$ 个 cluster 的颜色中心。

### 8.3 Cluster 参数广播

每个 cluster $j$ 有独立材料参数：

$$
\theta_j=(E_j,\nu_j,\mu_{N,j},\lambda_{N,j},\eta_j).
$$

若粒子 $i$ 属于 cluster $\ell_i$，则其 MPM 参数为：

$$
\theta_i=\theta_{\ell_i}.
$$

Particle filling 产生的新增粒子会继承最近 Gaussian 的 cluster 标签，因此它们也能获得对应区域的材料参数。

### 8.4 Cluster 梯度聚合

Warp 反传得到粒子级参数梯度后，系统按 cluster 求平均：

$$
\mathbf{g}_j=
\frac{1}{|C_j|}
\sum_{i\in C_j}
\mathbf{g}_i.
$$

随后只更新 `physics_config` 中 `param` 指定的参数。例如配置选择 $E,\nu,\eta$ 时，系统只更新这三个量，而 $\mu_N,\lambda_N$ 保持初始值。

### 8.5 参数更新与裁剪

对 $E,\mu_N,\lambda_N,\eta$，代码在 $\log_{10}$ 空间更新：

$$
\log_{10}\theta
\leftarrow
\mathrm{clip}
\left(
\log_{10}\theta
-
\gamma\hat{g},
l,u
\right),
\qquad
\theta\leftarrow 10^{\log_{10}\theta}.
$$

对 $\nu$，代码直接在线性空间更新：

$$
\nu
\leftarrow
\mathrm{clip}
(\nu-\gamma\hat{g},l,u).
$$

因此，如果 $E$ 的裁剪范围设置为 $[-1,2]$，实际物理范围是：

$$
E\in[10^{-1},10^2]=[0.1,100].
$$

这解释了为什么初始 $E=0.05$ 时，第一次更新后会被裁剪到 $0.1$。

## 9. 输出与可视化

程序会输出初始仿真视频、优化过程中的视频、普通 RGB 渲染帧、cluster mask 图像，以及记录聚类结果的 `.npy` 文件。Cluster mask 使用同一个 Gaussian rasterizer 渲染，但把 Gaussian 颜色替换为 cluster 调色板颜色，从而在单目图像中显示不同材料区域的位置。

这种可视化不保存额外 PLY，而是直接在渲染图像中展示 mask，更适合检查实际优化视角下的材料分区是否合理。

## 10. 数值稳定性设计

该系统结合了 diffusion guidance、可微 rasterization 和 differentiable MPM，梯度链路较长，因此代码加入了多种稳定性设计：渲染前检查 NaN/Inf，rasterizer OOM 时跳过异常 batch，autograd 梯度缺失时打印诊断信息，对物理参数进行裁剪，并将 $\nu$ 限制在安全范围内。此外，`low_ram_vae` 通过只保留部分帧的 VAE 梯度来控制显存占用。

## 11. 方法特点与局限

本方法的优点是把静态 3D Gaussian 场景扩展为可物理仿真的动态对象，不需要真实视频监督即可利用文本提示词优化材料参数；同时，颜色聚类使不同外观区域拥有独立物理属性，提高了非均匀材料表达能力。

主要局限在于：SDS 是弱监督，文本目标并不能唯一确定真实物理参数；颜色聚类也不一定严格对应真实材料分区，阴影和纹理可能造成误分；MPM 参数优化是非凸问题，对初始值、学习率和裁剪范围较敏感；低显存 VAE 模式虽然节省显存，但会削弱部分帧的直接梯度。

## 12. 总结

本项目实现了一个跨越 3D 表示、物理仿真、可微渲染和生成模型引导的优化系统。其核心思想是：用 MPM 产生物理运动，用 Gaussian Splatting 将运动状态渲染为视频，用 diffusion model 判断视频是否符合文本描述，再通过可微链路把误差反传到材料参数。

从优化角度看，系统求解的是：

$$
\min_{\theta}
\mathcal{L}_{\mathrm{SDS}}
\left(
\mathrm{Render}(\mathrm{MPM}_{\theta}(\mathcal{G})), y
\right).
$$

引入 cluster 后，$\theta$ 不再是单一全局材料参数，而是区域级参数集合 $\{\theta_j\}_{j=1}^k$。这种设计在保持优化变量数量较小的同时，增强了对非均匀材料的表达能力，是一种将生成式视觉先验用于物理参数反演的有效实践。
