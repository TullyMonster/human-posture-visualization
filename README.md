# 🧍 Human Posture Visualization

一个基于Web的交互式3D人体姿态可视化和编辑工具，支持SMPL/SMPLX模型格式，提供直观的姿态调节界面。

## ✨ 主要特性

### 🎯 **多数据集支持**

- **AMASS** - 自动识别SMPLX格式数据
- **3DPW** - 自动转换SMPL到SMPLX格式
- **HuMMan** - 智能适配多种人体模型
- **自适应处理** - 自动检测数据集类型并选择最佳转换策略

### 🖼️ **实时3D可视化**

- 基于PyRender的高质量3D渲染
- 实时姿态预览与调节
- 双模型对比显示（GT vs 调节后）
- 可自定义相机视角和光照

### 🎮 **交互式编辑**

- Web端直观的关节角度调节器
- 批量调节和单个关节精细控制
- 帧间调节参数复制功能
- 一键重置和实时预览

### ⚙️ **智能配置管理**

- YAML配置文件热重载
- 自动帧率检测和间隔计算
- 灵活的渲染参数配置
- 多模型自动选择

### 🔧 **开发友好**

- 模块化架构设计
- 完整的类型注解
- 详细的日志系统
- 易于扩展的适配器模式

## 🚀 快速开始

### 环境要求

- **Python**: 3.11+
- **操作系统**: Windows/Linux/macOS
- **GPU**: 应该可以支持CUDA的GPU（未测试、实现）

### 安装方式（使用 uv，推荐）

```bash
git clone https://github.com/your-org/human-posture-visualization.git
cd human-posture-visualization
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
uv run python interactive_pose_adjuster.py
```

### 模型文件准备

1. **下载SMPL/SMPLX模型文件**

   ```
   models/
   ├── smpl/
   │   ├── SMPL_MALE.npz
   │   ├── SMPL_FEMALE.npz
   │   └── SMPL_NEUTRAL.npz
   └── smplx/
       ├── SMPLX_MALE.npz
       ├── SMPLX_FEMALE.npz
       └── SMPLX_NEUTRAL.npz
   ```

2. **准备数据集文件**

   ```
   datasets/
   ├── AMASS/
   │   └── *.npz
   ├── 3DPW/
   │   └── *.pkl
   └── HuMMan/
       └── *.npz
   ```

## 📖 使用指南

### 基本使用流程

1. **配置数据集路径**

   ```yaml
   # config.yaml
   dataset:
     path: "./datasets/3DPW/office_phoneCall_00.pkl"
     type: "3DPW"
     gender: "neutral"
   ```

2. **启动应用**

   ```bash
   uv run python interactive_pose_adjuster.py
   ```

3. **打开Web界面**

   ```
   浏览器访问: http://localhost:5000
   ```

4. **调节姿态参数**
    - 选择目标关节
    - 拖动滑块调节角度
    - 实时预览变化效果
    - 保存或重置调节

### 配置文件说明

```yaml
# 数据集配置
dataset:
  path: "./datasets/3DPW/office_phoneCall_00.pkl"  # 数据文件路径
  type: "3DPW"                                     # 数据集类型
  gender: "neutral"                                # 性别偏好

# 帧序列参数
frames:
  start_frame: 10          # 起始帧
  frame_interval: 5        # 帧间隔
  num_frames: 3            # 显示帧数
  frame_offset: 0          # 帧偏移
  time_interval_ms: 100    # 时间间隔（毫秒）

# 渲染配置
rendering:
  width: 1200              # 渲染宽度
  height: 1200             # 渲染高度
  materials: # 材质配置
    gt: # Ground Truth材质
      color: [ 0.29, 0.33, 0.60, 0.7 ]
    predicted: # 预测材质
      color: [ 0.60, 0.33, 0.29, 0.8 ]

# 服务器配置
server:
  host: "0.0.0.0"
  port: 5000
  debug: true
```

### API接口

应用提供RESTful API接口：

- `GET /api/render` - 获取当前渲染图像
- `POST /api/adjust` - 调节关节角度
- `POST /api/navigate` - 帧导航
- `POST /api/reset` - 重置调节
- `GET /api/current_angles` - 获取当前角度值

## 🏗️ 项目架构

### 核心模块

```
├── interactive_pose_adjuster.py    # 主应用入口和Flask服务
├── config_manager.py              # 配置管理和热重载
├── dataset_adapter.py             # 数据集适配器
├── model_selector.py              # 模型自动选择
├── camera_adjuster_console.py     # 相机调节工具
├── config.yaml                    # 主配置文件
├── static/
│   ├── js/pose_adjuster.js        # 前端JavaScript逻辑
│   └── joint_config.json          # 关节配置
└── templates/
    └── pose_adjuster.html          # Web界面模板
```

### 技术栈

- **后端**: Flask + PyTorch + PyRender
- **前端**: HTML + JavaScript + CSS
- **3D渲染**: PyRender + Trimesh
- **人体模型**: SMPL/SMPLX
- **配置管理**: YAML + Watchdog
- **数据处理**: NumPy + Pillow

### 设计模式

- **适配器模式**: 统一不同数据集格式
- **策略模式**: 灵活的数据转换策略
- **观察者模式**: 配置文件热重载
- **工厂模式**: 模型自动选择机制

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 致谢

- [SMPL-X](https://smpl-x.is.tue.mpg.de/) - 人体模型
- [PyRender](https://pyrender.readthedocs.io/) - 3D渲染引擎
- [Flask](https://flask.palletsprojects.com/) - Web框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 联系方式

如有问题或建议，请通过以下方式联系我们：

- 📧 Email: [zeng-qi-hang@qq.com](mailto:zeng-qi-hang@qq.com)
- 🐛 Issues: [GitHub Issues](https://github.com/TullyMonster/human-posture-visualization/issues)
