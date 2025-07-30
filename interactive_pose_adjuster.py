"""
交互式人体姿态调节器
基于Flask + pyrender实现Web端SMPLX姿态调节
支持多种数据集格式：AMASS (SMPLX), 3DPW (SMPL), HuMMan (SMPL)
使用YAML配置文件管理所有参数，无硬编码路径
支持配置文件热重载功能
"""

import atexit
import base64
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image
from flask import Flask, request, jsonify, render_template

from config_manager import get_config_manager, DatasetConfig
# 导入数据集适配器、模型选择器和配置管理器
from dataset_adapter import DatasetAdapter

# ==================== 配置初始化 ====================
# 获取全局配置管理器
config_manager = get_config_manager()

# 全局变量，将在初始化函数中设置
dataset_config = None
render_config = None
RENDER_WIDTH = None
RENDER_HEIGHT = None
engine = None
_cleanup_registered = False  # 防止重复注册退出清理


def load_global_config():
    """加载全局配置变量"""
    global dataset_config, render_config, RENDER_WIDTH, RENDER_HEIGHT

    # 从配置文件获取当前数据集和渲染配置
    dataset_config = config_manager.get_current_dataset_config()
    render_config = config_manager.get_render_config()

    # 渲染配置：从配置文件获取
    RENDER_WIDTH = render_config.width
    RENDER_HEIGHT = render_config.height

    print(f'📐 渲染尺寸已更新: {RENDER_WIDTH}x{RENDER_HEIGHT}')


def on_config_changed():
    """配置文件变更回调函数"""
    try:
        # 重新加载全局配置
        load_global_config()

        # 重新初始化引擎
        initialize_engine()

        print('🔄 应用完成')

    except Exception as e:
        print(f'❌ 配置应用失败: {e}')


# 初始化全局配置
load_global_config()

# 加载关节配置
with open('./static/joint_config.json', 'r', encoding='utf-8') as f:
    joint_config_data = json.load(f)
    CORE_JOINTS = joint_config_data['core_joints']

# ==================== Flask应用初始化 ====================
app = Flask(__name__)


class PoseAdjusterEngine:
    """
    姿态调节引擎核心类
    """

    def __init__(self, dataset_config: DatasetConfig):
        """
        初始化姿态调节引擎
        
        :param dataset_config: 数据集配置对象
        """
        self.poses = None
        self.betas = None
        self.model = None
        self.frame_indices = []
        self.current_frame_idx = 0

        # 从配置对象获取序列配置
        self.dataset_config = dataset_config
        self.start_frame = dataset_config.start_frame
        self.frame_interval = dataset_config.frame_interval
        self.num_frames = dataset_config.num_frames
        self.frame_offset = dataset_config.frame_offset

        # 当前调节状态：存储每帧的姿态调节
        self.adjusted_poses = {}  # {frame_idx: adjusted_pose_tensor}

        # 从配置获取相机姿态矩阵
        self.camera_pose = config_manager.get_camera_pose_matrix()

        # 数据集信息
        self.total_frames = 0
        self.framerate = 30.0

        self.load_data()

    @property
    def render_width(self) -> int:
        """动态获取渲染宽度"""
        return config_manager.get_render_config().width

    @property
    def render_height(self) -> int:
        """动态获取渲染高度"""
        return config_manager.get_render_config().height

    def load_data(self):
        """加载数据和模型（支持多种数据集格式）"""
        print(f'数据集: {self.dataset_config.path}')

        # 首先尝试智能适配器（推荐方式）
        adapter_success = False
        recommended_model_config = None

        try:
            data, model_config = DatasetAdapter.smart_convert(
                data_path=self.dataset_config.path,
                models_dir=Path('./models'),
                preferred_gender=config_manager.get_current_gender()
            )

            self.poses = data['poses']
            self.betas = data['betas']
            recommended_model_config = model_config
            self.total_frames = self.poses.shape[0]
            self.framerate = float(data.get('mocap_framerate', 30.0))
            adapter_success = True

            print(f'{self.total_frames}帧, {self.poses.shape[1]}维姿态参数')

        except Exception as e:
            # 回退到标准适配器
            try:
                data = DatasetAdapter.convert_to_smplx_format(self.dataset_config.path)
                self.poses = data['poses']
                self.betas = data['betas']
                self.total_frames = self.poses.shape[0]
                self.framerate = float(data.get('mocap_framerate', 30.0))

                if self.poses.shape[1] == 156:
                    adapter_success = True

            except Exception as std_error:
                pass

                # 如果适配器失败，回退到原始加载方式
        if not adapter_success:
            try:
                if self.dataset_config.path.suffix == '.npz':
                    data = np.load(self.dataset_config.path, allow_pickle=True)
                elif self.dataset_config.path.suffix == '.pkl':
                    import pickle
                    with open(self.dataset_config.path, 'rb') as f:
                        data = pickle.load(f, encoding='latin1')
                else:
                    raise ValueError(f'不支持的文件格式: {self.dataset_config.path.suffix}')

                # 处理可能的list格式数据
                poses = data['poses']
                betas = data['betas']

                # 处理多人数据（在转换为numpy数组之前）
                if isinstance(poses, list) and len(poses) > 0:
                    # 检查是否是多人数据（list的每个元素是一个人的数据）
                    if isinstance(poses[0], (list, np.ndarray)):
                        poses = poses[0]  # 取第一个人

                if isinstance(betas, list) and len(betas) > 0:
                    # 检查是否是多人数据
                    if isinstance(betas[0], (list, np.ndarray)):
                        betas = betas[0]  # 取第一个人

                # 确保poses是numpy数组
                if not isinstance(poses, np.ndarray):
                    poses = np.array(poses)

                # 处理poses的额外维度检查
                if len(poses.shape) > 2:
                    poses = poses[0]  # 取第一个人（如果还有多维）

                # 确保betas是numpy数组
                if not isinstance(betas, np.ndarray):
                    betas = np.array(betas)

                # 处理betas的额外维度检查
                if len(betas.shape) > 1:
                    betas = betas[0]  # 取第一个人（如果还有多维）

                self.poses = poses
                self.betas = betas[:10]  # 只取前10个beta参数
                self.total_frames = self.poses.shape[0]
                self.framerate = float(data.get('mocap_framerate', 30.0))

            except Exception as fallback_error:
                raise RuntimeError(f'数据加载失败: {self.dataset_config.path}')

        # 计算帧序列（严格模式：只使用真实存在的原始请求帧）
        calculated_frames = [self.start_frame + i * self.frame_interval for i in range(self.num_frames)]

        # 过滤掉超出数据范围的帧（严格模式：不补充任何帧）
        valid_frames = [f for f in calculated_frames if f < self.total_frames]

        if len(valid_frames) < self.num_frames:
            # 严格模式：只使用真实存在的帧
            self.frame_indices = valid_frames
            self.num_frames = len(valid_frames)
        else:
            self.frame_indices = valid_frames

        # 最终保护：确保至少有一帧
        if len(self.frame_indices) == 0:
            self.frame_indices = [self.total_frames - 1]
            self.num_frames = 1

        print(f'帧序列: {self.frame_indices}')

        # 加载推荐的模型
        if recommended_model_config:
            model_path = recommended_model_config.model_path
            gender = recommended_model_config.gender.lower()

            print(f'模型: {model_path}')

            if recommended_model_config.model_type == 'SMPLX':
                from smplx import SMPLX
                self.model = SMPLX(
                    model_path=str(model_path),
                    gender=gender,
                    num_betas=10,
                    use_pca=False,
                    flat_hand_mean=True
                )
            else:  # SMPL
                from smplx import SMPL
                self.model = SMPL(
                    model_path=str(model_path),
                    gender=gender,
                    num_betas=10
                )
        else:
            # 回退到默认SMPLX模型
            default_model_path = "./models/smplx/SMPLX_NEUTRAL.npz"
            print(f'⚠️ 使用默认模型: {default_model_path}')

            from smplx import SMPLX
            self.model = SMPLX(
                model_path=default_model_path,
                gender="neutral",
                num_betas=10,
                use_pca=False,
                flat_hand_mean=True
            )

    def get_current_poses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前帧的GT和调节后的姿态（考虑帧偏移）
        
        :return: (gt_pose, adjusted_pose)
        """
        # 确保当前帧索引在有效范围内
        if self.current_frame_idx >= len(self.frame_indices):
            print(
                f'⚠️ 帧索引越界: current_frame_idx={self.current_frame_idx}, frame_indices长度={len(self.frame_indices)}')
            self.current_frame_idx = len(self.frame_indices) - 1

        frame_idx = self.frame_indices[self.current_frame_idx]

        # 确保帧索引在数据范围内
        if frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1

        # 调试信息已简化

        # GT姿态（始终使用原始帧）
        poses_slice = self.poses[frame_idx:frame_idx + 1]
        if hasattr(poses_slice, 'clone'):
            # 如果是PyTorch tensor
            gt_pose = poses_slice.clone().float()
        else:
            # 如果是numpy array
            gt_pose = torch.tensor(poses_slice, dtype=torch.float32)

        # 验证gt_pose的维度
        if gt_pose.shape[0] == 0:
            print(f'❌ GT姿态为空: frame_idx={frame_idx}, poses.shape={self.poses.shape}')
            # 使用最后一个有效帧
            valid_frame_idx = min(frame_idx, self.total_frames - 1)
            gt_pose = torch.tensor(self.poses[valid_frame_idx:valid_frame_idx + 1], dtype=torch.float32)


        # 预测姿态的基础帧（考虑偏移）
        predicted_base_frame = frame_idx + self.frame_offset

        # 确保偏移后的帧在有效范围内
        if predicted_base_frame < 0:

            predicted_base_frame = 0
        elif predicted_base_frame >= self.total_frames:
            print(
                f'⚠️ 偏移帧{predicted_base_frame}超出范围(总帧数{self.total_frames})，调整为最后一帧{self.total_frames - 1}')
            predicted_base_frame = self.total_frames - 1

        # 预测姿态始终从偏移后的基础帧开始
        poses_slice = self.poses[predicted_base_frame:predicted_base_frame + 1]
        if hasattr(poses_slice, 'clone'):
            # 如果是PyTorch tensor
            predicted_pose = poses_slice.clone().float()
        else:
            # 如果是numpy array
            predicted_pose = torch.tensor(poses_slice, dtype=torch.float32)

        # 验证predicted_pose的维度
        if predicted_pose.shape[0] == 0:
            # 使用GT姿态作为备选
            predicted_pose = gt_pose.clone()

        # 如果当前帧有用户调节，这些调节应该是存储为相对于GT的修改
        # 我们需要将这些修改应用到预测基础姿态上
        if self.current_frame_idx in self.adjusted_poses:
            # 获取用户的调节数据（这应该是绝对角度）
            user_adjusted_pose = self.adjusted_poses[self.current_frame_idx].clone()

            # 确保调节后的姿态有完整的维度
            if user_adjusted_pose.shape[1] < gt_pose.shape[1]:
                # 如果维度不足，创建完整维度的姿态，用GT补充
                full_adjusted_pose = gt_pose.clone()
                # 只替换已调节的部分
                full_adjusted_pose[0, :user_adjusted_pose.shape[1]] = user_adjusted_pose[0, :]
                predicted_pose = full_adjusted_pose
            else:
                # 直接使用用户调节的绝对角度作为最终预测姿态
                predicted_pose = user_adjusted_pose

        return gt_pose, predicted_pose

    def adjust_joint(self, joint_name: str, axis: int, angle_degrees: float, operation: str = 'set'):
        """
        调节指定关节的角度 - 简化版本，直接操作绝对角度
        
        :param joint_name: 关节名称
        :param axis: 轴向索引 (0, 1, 2)
        :param angle_degrees: 绝对角度值（度数）
        :param operation: 操作类型 ('set', 'add', 'reset')
        """
        if joint_name not in CORE_JOINTS:
            raise ValueError(f'不支持的关节: {joint_name}')

        joint_config = CORE_JOINTS[joint_name]
        if axis < 0 or axis >= len(joint_config['indices']):
            raise ValueError(f'无效的轴向索引: {axis}')

        # 获取当前姿态
        gt_pose, current_adjusted = self.get_current_poses()

        # 获取姿态索引
        pose_index = joint_config['indices'][axis]
        angle_radians = math.radians(angle_degrees)

        if operation == 'set':
            # 直接设置绝对角度
            current_adjusted[0, pose_index] = angle_radians
        elif operation == 'add':
            # 增量调节
            current_adjusted[0, pose_index] += angle_radians
        elif operation == 'reset':
            # 重置到偏移后的基础姿态
            frame_idx = self.frame_indices[self.current_frame_idx]
            reset_base_frame = frame_idx + self.frame_offset

            # 确保偏移后的帧不超出数据范围
            if reset_base_frame < 0 or reset_base_frame >= self.total_frames:
                reset_base_frame = frame_idx

            poses_slice = self.poses[reset_base_frame:reset_base_frame + 1]
            if hasattr(poses_slice, 'clone'):
                # 如果是PyTorch tensor
                reset_base_pose = poses_slice.clone().float()
            else:
                # 如果是numpy array
                reset_base_pose = torch.tensor(poses_slice, dtype=torch.float32)
            current_adjusted[0, pose_index] = reset_base_pose[0, pose_index]

        # 保存调节结果
        self.adjusted_poses[self.current_frame_idx] = current_adjusted

    def copy_prev_frame_adjustments(self):
        """
        复制上一帧的调节变化量到当前帧
        计算上一帧预测相对于GT的差异，然后应用到当前帧的GT上
        """
        if self.current_frame_idx == 0:
            return False  # 第一帧没有上一帧

        prev_frame_idx = self.current_frame_idx - 1
        if prev_frame_idx in self.adjusted_poses:
            # 获取上一帧的GT姿态
            prev_actual_frame = self.frame_indices[prev_frame_idx]
            poses_slice = self.poses[prev_actual_frame:prev_actual_frame + 1]
            if hasattr(poses_slice, 'clone'):
                prev_gt_pose = poses_slice.clone().float()
            else:
                prev_gt_pose = torch.tensor(poses_slice, dtype=torch.float32)

            # 获取上一帧的调节后姿态
            prev_adjusted_pose = self.adjusted_poses[prev_frame_idx]

            # 计算调节变化量（相对调节量）
            adjustment_delta = prev_adjusted_pose - prev_gt_pose

            # 获取当前帧的GT姿态
            current_actual_frame = self.frame_indices[self.current_frame_idx]
            poses_slice = self.poses[current_actual_frame:current_actual_frame + 1]
            if hasattr(poses_slice, 'clone'):
                current_gt_pose = poses_slice.clone().float()
            else:
                current_gt_pose = torch.tensor(poses_slice, dtype=torch.float32)

            # 将调节变化量应用到当前帧GT上
            current_adjusted_pose = current_gt_pose + adjustment_delta

            # 保存到当前帧
            self.adjusted_poses[self.current_frame_idx] = current_adjusted_pose
            return True
        return False

    def create_body_mesh(self, pose: torch.Tensor, material: pyrender.Material) -> pyrender.Mesh:
        """
        创建人体3D网格（兼容SMPL和SMPLX）
        
        :param pose: 姿态参数
        :param material: 渲染材质
        :return: pyrender.Mesh对象
        """
        # 处理betas参数，确保正确的形状和数据类型
        if len(self.betas.shape) == 1:
            # betas是1D数组，表示单个人的身体形状参数
            if len(self.betas) >= 10:
                betas_tensor = torch.tensor(self.betas[:10][None], dtype=torch.float32)
            else:
                # 如果betas不足10个，用零填充
                betas_padded = np.zeros(10)
                betas_padded[:len(self.betas)] = self.betas
                betas_tensor = torch.tensor(betas_padded[None], dtype=torch.float32)
        else:
            # betas是2D数组，使用第一个人的数据
            if self.betas.shape[0] > 0 and self.betas.shape[1] >= 10:
                betas_tensor = torch.tensor(self.betas[0:1, :10], dtype=torch.float32)
            else:
                # 默认使用零值betas
                betas_tensor = torch.zeros(1, 10, dtype=torch.float32)

        transl = torch.zeros(1, 3)

        # 检测模型类型
        model_type = type(self.model).__name__
        is_smplx_model = (
                model_type == 'SMPLX' or
                'SMPLX' in model_type or
                hasattr(self.model, 'left_hand_pose') or
                hasattr(self.model, 'right_hand_pose')
        )

        try:
            if is_smplx_model and pose.shape[1] >= 156:
                # SMPLX模型且有完整的156维姿态数据
                output = self.model(
                    betas=betas_tensor,
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:66],
                    left_hand_pose=pose[:, 66:111],
                    right_hand_pose=pose[:, 111:156],
                    transl=transl
                )
            elif is_smplx_model:
                # SMPLX模型但姿态数据不足156维，只使用身体部分
                output = self.model(
                    betas=betas_tensor,
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:min(66, pose.shape[1])],
                    transl=transl
                )
            else:
                # SMPL模型，只使用身体参数
                if pose.shape[1] >= 66:
                    output = self.model(
                        betas=betas_tensor,
                        global_orient=pose[:, :3],
                        body_pose=pose[:, 3:66],
                        transl=transl
                    )
                else:
                    # 姿态维度不足，扩展到66维
                    extended_pose = torch.zeros(1, 66)
                    extended_pose[:, :pose.shape[1]] = pose
                    output = self.model(
                        betas=betas_tensor,
                        global_orient=extended_pose[:, :3],
                        body_pose=extended_pose[:, 3:66],
                        transl=transl
                    )

        except Exception as e:
            print(f'❌ 模型调用失败: {str(e)}')
            # 回退策略：只使用身体关节
            try:
                output = self.model(
                    betas=betas_tensor,
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:66],
                    transl=transl
                )
            except Exception as fallback_error:
                raise fallback_error

        vertices = output.vertices.detach().cpu().numpy().squeeze()
        body_mesh = trimesh.Trimesh(vertices, self.model.faces)
        return pyrender.Mesh.from_trimesh(body_mesh, material=material, smooth=False)

    def render_single_frame(self, frame_idx: int = None) -> np.ndarray:
        """
        渲染单帧图像（包含GT和调节后的人体，支持不同模型格式）
        
        :param frame_idx: 帧索引，如果为None则使用当前帧
        :return: 渲染的图像数组
        """
        if frame_idx is None:
            frame_idx = self.current_frame_idx

        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f'帧索引超出范围: {frame_idx}')

        scene = pyrender.Scene()

        # 设置环境光
        scene.ambient_light = [0.3, 0.3, 0.3]

        # 获取当前帧的GT和预测姿态（考虑帧偏移）
        original_frame_idx = self.current_frame_idx
        self.current_frame_idx = frame_idx  # 临时设置帧索引
        gt_pose, predicted_pose = self.get_current_poses()
        self.current_frame_idx = original_frame_idx  # 恢复原始帧索引

        # 获取材质配置
        render_config = config_manager.get_render_config()
        gt_material_config = render_config.gt_material
        predicted_material_config = render_config.predicted_material

        # GT材质（蓝色半透明）
        gt_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=gt_material_config['color'],
            metallicFactor=gt_material_config['metallic'],
            roughnessFactor=gt_material_config['roughness']
        )

        # 预测材质（棕色半透明）
        predicted_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=predicted_material_config['color'],
            metallicFactor=predicted_material_config['metallic'],
            roughnessFactor=predicted_material_config['roughness']
        )

        # 生成GT网格（叠加显示）
        gt_mesh = self.create_body_mesh(gt_pose, gt_material)
        scene.add(gt_mesh)

        # 生成预测网格（叠加显示，始终渲染，显示帧偏移效果）
        predicted_mesh = self.create_body_mesh(predicted_pose, predicted_material)
        scene.add(predicted_mesh)

        # 获取光照配置
        lighting_config = render_config.lighting

        # 设置主光源
        directional_light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=lighting_config['directional_intensity']
        )
        light_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]])
        scene.add(directional_light, pose=light_pose)

        # 设置补充光
        fill_light = pyrender.DirectionalLight(
            color=lighting_config['fill_light_color'],
            intensity=lighting_config['fill_light_intensity']
        )
        fill_light_pose = np.eye(4)
        fill_light_pose[:3, 3] = np.array([2, 1, 2])
        scene.add(fill_light, pose=fill_light_pose)

        # 设置相机
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        scene.add(camera, pose=self.camera_pose)

        # 渲染
        renderer = pyrender.OffscreenRenderer(self.render_width, self.render_height)
        color, _ = renderer.render(scene)
        renderer.delete()

        return color

    def render_sequence(self) -> str:
        """
        渲染多帧序列（1行NUM_FRAMES列）
        
        :return: Base64编码的PNG图像
        """
        frames = []

        # 渲染每一帧
        for i in range(self.num_frames):
            frame_image = self.render_single_frame(i)
            frames.append(frame_image)

        # 水平拼接所有帧
        sequence_width = self.render_width * self.num_frames
        sequence_height = self.render_height
        sequence_image = np.zeros((sequence_height, sequence_width, 3), dtype=np.uint8)

        for i, frame in enumerate(frames):
            x_start = i * self.render_width
            x_end = (i + 1) * self.render_width
            sequence_image[:, x_start:x_end, :] = frame

        # 在当前编辑帧周围添加高亮边框
        current_x_start = self.current_frame_idx * self.render_width
        current_x_end = (self.current_frame_idx + 1) * self.render_width

        # 绘制红色边框表示当前编辑帧
        border_width = 5
        sequence_image[:border_width, current_x_start:current_x_end, :] = [255, 0, 0]  # 上边框
        sequence_image[-border_width:, current_x_start:current_x_end, :] = [255, 0, 0]  # 下边框
        sequence_image[:, current_x_start:current_x_start + border_width, :] = [255, 0, 0]  # 左边框
        sequence_image[:, current_x_end - border_width:current_x_end, :] = [255, 0, 0]  # 右边框

        # 转换为Base64
        image = Image.fromarray(sequence_image)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return f'data:image/png;base64,{image_base64}'

    def render_sequence_for_export(self) -> str:
        """
        渲染多帧序列用于导出（无边框标识）
        
        :return: Base64编码的PNG图像
        """
        frames = []

        # 渲染每一帧
        for i in range(self.num_frames):
            frame_image = self.render_single_frame(i)
            frames.append(frame_image)

        # 水平拼接所有帧（无边框）
        sequence_width = self.render_width * self.num_frames
        sequence_height = self.render_height
        sequence_image = np.zeros((sequence_height, sequence_width, 3), dtype=np.uint8)

        for i, frame in enumerate(frames):
            x_start = i * self.render_width
            x_end = (i + 1) * self.render_width
            sequence_image[:, x_start:x_end, :] = frame

        # 转换为Base64
        image = Image.fromarray(sequence_image)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return f'data:image/png;base64,{image_base64}'


# ==================== Flask路由 ====================

@app.route('/')
def index():
    """主页面"""
    return render_template('pose_adjuster.html')


@app.route('/api/render')
def api_render():
    """渲染序列"""
    try:
        image_base64 = engine.render_sequence()

        # 添加时间戳防止缓存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒级时间戳

        return jsonify({
            'success': True,
            'image': image_base64,
            'frame_idx': engine.current_frame_idx,
            'frame_count': len(engine.frame_indices),
            'actual_frame': engine.frame_indices[engine.current_frame_idx],
            'modified': engine.current_frame_idx in engine.adjusted_poses,
            'timestamp': timestamp
        })
    except Exception as e:
        print(f'❌ 渲染API错误: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/adjust', methods=['POST'])
def api_adjust():
    """调节关节角度"""
    try:
        data = request.json
        joint_name = data['joint_name']
        axis = int(data['axis'])
        angle = float(data['angle'])
        operation = data.get('operation', 'set')

        engine.adjust_joint(joint_name, axis, angle, operation)

        # 重新渲染
        image_base64 = engine.render_sequence()

        return jsonify({
            'success': True,
            'image': image_base64,
            'modified': True
        })
    except Exception as e:
        import traceback
        error_msg = f"adjust错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"调试信息: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': error_msg}), 500


@app.route('/api/adjust_batch', methods=['POST'])
def api_adjust_batch():
    """批量调节关节角度"""
    try:
        data = request.json
        changes = data['changes']

        # 批量应用所有调整
        for change in changes:
            joint_name = change['joint_name']
            axis = int(change['axis'])
            angle = float(change['angle'])
            operation = change.get('operation', 'set')
            engine.adjust_joint(joint_name, axis, angle, operation)

        # 只渲染一次（而不是每个调整都渲染）
        image_base64 = engine.render_sequence()

        return jsonify({
            'success': True,
            'image': image_base64,
            'modified': True
        })
    except Exception as e:
        import traceback
        error_msg = f"adjust_batch错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"调试信息: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': error_msg}), 500


@app.route('/api/navigate', methods=['POST'])
def api_navigate():
    """帧导航"""
    try:
        data = request.json
        direction = data['direction']  # 'prev', 'next', 'goto'

        if direction == 'prev':
            engine.current_frame_idx = max(0, engine.current_frame_idx - 1)
        elif direction == 'next':
            engine.current_frame_idx = min(len(engine.frame_indices) - 1, engine.current_frame_idx + 1)
        elif direction == 'goto':
            target_idx = int(data['target_idx'])
            engine.current_frame_idx = max(0, min(len(engine.frame_indices) - 1, target_idx))

        # 重新渲染
        image_base64 = engine.render_sequence()

        return jsonify({
            'success': True,
            'image': image_base64,
            'frame_idx': engine.current_frame_idx,
            'frame_count': len(engine.frame_indices),
            'actual_frame': engine.frame_indices[engine.current_frame_idx],
            'modified': engine.current_frame_idx in engine.adjusted_poses
        })
    except Exception as e:
        import traceback
        error_msg = f"navigate错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"调试信息: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': error_msg}), 500


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """重置当前帧"""
    try:
        # 移除当前帧的调节
        if engine.current_frame_idx in engine.adjusted_poses:
            del engine.adjusted_poses[engine.current_frame_idx]

        # 重新渲染
        image_base64 = engine.render_sequence()

        return jsonify({
            'success': True,
            'image': image_base64,
            'modified': False
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export')
def api_export():
    """导出当前渲染结果"""
    try:
        image_base64 = engine.render_sequence_for_export()

        # 这里可以扩展为保存到文件等
        return jsonify({
            'success': True,
            'image': image_base64,
            'filename': f'pose_adjustment_sequence_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reset_joint', methods=['POST'])
def api_reset_joint():
    """重置单个关节"""
    try:
        data = request.json
        joint_name = data['joint_name']
        axis = int(data['axis'])

        engine.adjust_joint(joint_name, axis, 0, 'reset')

        # 重新渲染
        image_base64 = engine.render_sequence()

        return jsonify({
            'success': True,
            'image': image_base64,
            'modified': engine.current_frame_idx in engine.adjusted_poses
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data_info')
def api_data_info():
    """获取数据集信息"""
    try:
        return jsonify({
            'success': True,
            'total_frames': engine.total_frames,
            'framerate': engine.framerate,
            'current_sequence': engine.frame_indices
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sequence_config')
def api_sequence_config():
    """获取当前序列配置参数"""
    try:
        return jsonify({
            'success': True,
            'start_frame': engine.start_frame,
            'frame_interval': engine.frame_interval,
            'num_frames': engine.num_frames,
            'frame_offset': engine.frame_offset
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/copy_prev', methods=['POST'])
def api_copy_prev():
    """复制上一帧的调节参数"""
    try:
        copied = engine.copy_prev_frame_adjustments()

        if copied:
            # 重新渲染
            image_base64 = engine.render_sequence()

            return jsonify({
                'success': True,
                'image': image_base64,
                'modified': True,
                'message': '已复制上一帧参数'
            })
        else:
            return jsonify({
                'success': False,
                'error': '无法复制：当前为第一帧或上一帧无调节'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/play_frame', methods=['POST'])
def api_play_frame():
    """播放指定帧（用于循环播放）"""
    try:
        data = request.json
        frame_idx = int(data['frame_idx'])

        if 0 <= frame_idx < len(engine.frame_indices):
            frame_image = engine.render_single_frame(frame_idx)

            # 转换为Base64
            image = Image.fromarray(frame_image)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{image_base64}',
                'frame_idx': frame_idx,
                'frame_count': len(engine.frame_indices),
                'actual_frame': engine.frame_indices[frame_idx]
            })
        else:
            return jsonify({'success': False, 'error': '帧索引超出范围'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/update_sequence', methods=['POST'])
def api_update_sequence():
    """更新序列参数"""
    try:
        data = request.json
        start_frame = int(data['start_frame'])
        frame_interval = int(data['frame_interval'])
        num_frames = int(data['num_frames'])
        frame_offset = int(data.get('frame_offset', 0))

        # 验证参数
        if start_frame < 0 or start_frame >= engine.total_frames:
            return jsonify({'success': False, 'error': f'起始帧必须在0-{engine.total_frames - 1}范围内'})

        if frame_interval < 1:
            return jsonify({'success': False, 'error': '帧间隔必须大于0'})

        if num_frames < 1:
            return jsonify({'success': False, 'error': '帧数必须大于0'})

        # 检查序列是否超出数据范围
        max_frame = start_frame + (num_frames - 1) * frame_interval
        if max_frame >= engine.total_frames:
            return jsonify({
                'success': False,
                'error': f'序列超出数据范围，最大帧为{max_frame}，数据总帧数为{engine.total_frames}'
            })

        # 检查帧偏移是否会导致超出范围
        max_offset_frame = max_frame + abs(frame_offset)
        if max_offset_frame >= engine.total_frames:
            return jsonify({
                'success': False,
                'error': f'帧偏移导致超出数据范围，最大访问帧为{max_offset_frame}，数据总帧数为{engine.total_frames}'
            })

        # 更新引擎配置
        old_frame_offset = engine.frame_offset
        old_adjusted_poses_count = len(engine.adjusted_poses)

        engine.start_frame = start_frame
        engine.frame_interval = frame_interval
        engine.num_frames = num_frames
        engine.frame_offset = frame_offset
        engine.frame_indices = [start_frame + i * frame_interval for i in range(num_frames)]
        engine.current_frame_idx = 0

        # 总是清空调节数据，确保帧偏移效果能够显现
        engine.adjusted_poses = {}  # 清空之前的调节

        return jsonify({'success': True, 'message': '参数更新成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/current_angles')
def api_current_angles():
    """获取当前帧的关节角度值（简化版本 - 直接返回绝对角度）"""
    try:
        # 获取当前姿态（GT + 调节）
        gt_pose, adjusted_pose = engine.get_current_poses()
        current_pose = adjusted_pose.detach().cpu().numpy().squeeze()

        # 计算每个关节的绝对角度值（弧度转角度）
        joint_angles = {}
        for joint_name, config in CORE_JOINTS.items():
            joint_angles[joint_name] = {}
            for axis_idx, pose_idx in enumerate(config['indices']):
                # 当前绝对角度
                current_angle_radians = current_pose[pose_idx]
                current_angle_degrees = math.degrees(current_angle_radians)
                joint_angles[joint_name][axis_idx] = round(current_angle_degrees, 1)

        return jsonify({
            'success': True,
            'frame_idx': engine.current_frame_idx,
            'actual_frame': engine.frame_indices[engine.current_frame_idx],
            'joint_angles': joint_angles,
            'modified': engine.current_frame_idx in engine.adjusted_poses
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def initialize_engine():
    """初始化全局引擎实例（基于配置文件）"""
    global engine

    # 获取当前数据集配置
    current_dataset_config = config_manager.get_current_dataset_config()
    server_config = config_manager.get_server_config()

    # 创建引擎实例
    engine = PoseAdjusterEngine(current_dataset_config)

    # 在数据加载后重新计算帧间隔（如果配置了时间间隔控制）
    if current_dataset_config.time_interval_ms > 0:
        # 使用数据集的真实帧率
        actual_framerate = engine.framerate
        frame_time_ms = 1000.0 / actual_framerate
        calculated_frame_interval = max(1, round(current_dataset_config.time_interval_ms / frame_time_ms))

        # 重新计算帧序列
        if calculated_frame_interval != engine.frame_interval:
            engine.frame_interval = calculated_frame_interval
            engine.frame_indices = [engine.start_frame + i * calculated_frame_interval for i in
                                    range(engine.num_frames)]
            # 过滤掉超出数据范围的帧
            valid_frames = [f for f in engine.frame_indices if f < engine.total_frames]
            if len(valid_frames) < engine.num_frames:
                engine.frame_indices = valid_frames
                engine.num_frames = len(valid_frames)
            print(
                f'帧率{actual_framerate}fps: {calculated_frame_interval}帧/{current_dataset_config.time_interval_ms}ms')
            print(f'📋 更新后帧序列: {engine.frame_indices}')
        else:
            print(
                f'⏱️ 帧间隔无需调整: {calculated_frame_interval}帧/{current_dataset_config.time_interval_ms}ms ({actual_framerate}fps)')

    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 5000)
    print(f'启动服务器: http://localhost:{port}')


def start_config_monitoring():
    """启动配置文件监控"""
    global _cleanup_registered
    try:
        # 注册配置变更回调（避免重复注册）
        config_manager.register_change_callback(on_config_changed)

        # 启动配置文件监控
        config_manager.start_monitoring()

        # 只注册一次退出清理
        if not _cleanup_registered:
            atexit.register(stop_config_monitoring)
            _cleanup_registered = True

    except Exception as e:
        print(f'❌ 启动配置文件监控失败: {e}')


def stop_config_monitoring():
    """停止配置文件监控"""
    try:
        config_manager.stop_monitoring()
        # 移除重复输出，由config_manager.stop_monitoring()自己输出
    except Exception as e:
        print(f'❌ 停止配置文件监控失败: {e}')


if __name__ == '__main__':
    # 启动配置文件监控
    start_config_monitoring()

    # 初始化引擎
    initialize_engine()

    # 从配置获取服务器设置
    server_config = config_manager.get_server_config()
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 5000)
    debug = server_config.get('debug', True)

    try:
        app.run(debug=debug, host=host, port=port)
    finally:
        # 确保停止监控
        stop_config_monitoring()
