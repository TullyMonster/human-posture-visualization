"""
交互式人体姿态调节器
基于Flask + pyrender实现Web端SMPLX姿态调节
"""

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
from smplx import SMPLX

# ==================== 配置常量 ====================
# 数据集文件：包含人体动作序列的姿态参数
# 要求：NPZ格式，包含poses[T,156]和betas[10]数组
# T=帧数，156=SMPLX参数维度（global_orient+body_pose+hand_poses）
DATA_SET: Path = Path(r'./datasets/AMASS/G18 push kick right poses.npz')

# SMPLX人体模型文件：定义人体网格拓扑和蒙皮权重
# 支持：MALE/FEMALE/NEUTRAL，影响体型和比例
SMPL_MODEL: Path = Path(r'./models/smplx/SMPLX_MALE.npz')

# 渲染配置：单帧图像分辨率
# 序列图像总宽度 = RENDER_WIDTH × 帧数
RENDER_WIDTH = 1200
RENDER_HEIGHT = 1200

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

    def __init__(self, start_frame: int, frame_interval: int, num_frames: int, frame_offset: int = 0):
        self.poses = None
        self.betas = None
        self.model = None
        self.frame_indices = []
        self.current_frame_idx = 0

        # 序列配置
        self.start_frame = start_frame
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.frame_offset = frame_offset  # 预测数据相对于GT的偏移帧数

        # 当前调节状态：存储每帧的姿态调节
        self.adjusted_poses = {}  # {frame_idx: adjusted_pose_tensor}

        # 相机状态
        self.camera_pose = np.array([
            [0.6102284363588065, 0.22558472023756496, -0.7594292524352928, -1.8948403428657872],
            [-0.7864855217646244, 0.0573172953978972, -0.614943291452878, -1.4792258307646078],
            [-0.09519337956872702, 0.972535995037516, 0.2123957599451424, 0.40015489940524757],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # 数据集信息
        self.total_frames = 0
        self.framerate = 30.0

        self.load_data()

    def load_data(self):
        """加载数据和模型"""
        print(f'🔄 加载数据集: {DATA_SET}')
        data = np.load(DATA_SET)
        self.poses = data['poses']
        self.betas = data['betas']

        # 获取数据集信息
        self.total_frames = self.poses.shape[0]
        try:
            self.framerate = float(data['mocap_framerate'])
        except KeyError:
            self.framerate = 30.0  # 默认帧率

        # 计算帧序列
        self.frame_indices = [self.start_frame + i * self.frame_interval for i in range(self.num_frames)]
        print(f'🎯 选择帧序列: {self.frame_indices}')

        # 加载SMPLX模型
        print('🔄 加载SMPLX模型...')
        self.model = SMPLX(
            model_path=SMPL_MODEL.as_posix(),
            gender="male",
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True
        )
        print('✅ 模型加载完成')

    def get_current_poses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前帧的GT和调节后的姿态（考虑帧偏移）
        
        :return: (gt_pose, adjusted_pose)
        """
        frame_idx = self.frame_indices[self.current_frame_idx]

        # GT姿态（始终使用原始帧）
        gt_pose = torch.tensor(self.poses[frame_idx:frame_idx + 1], dtype=torch.float32)

        # 预测姿态的基础帧（考虑偏移）
        predicted_base_frame = frame_idx + self.frame_offset

        # 确保偏移后的帧不超出数据范围
        if predicted_base_frame < 0 or predicted_base_frame >= self.total_frames:
            # 如果偏移超出范围，使用原始帧作为基础
            predicted_base_frame = frame_idx

        # 预测姿态始终从偏移后的基础帧开始
        predicted_pose = torch.tensor(self.poses[predicted_base_frame:predicted_base_frame + 1], dtype=torch.float32)

        # 如果当前帧有用户调节，这些调节应该是存储为相对于GT的修改
        # 我们需要将这些修改应用到预测基础姿态上
        if self.current_frame_idx in self.adjusted_poses:
            # 获取用户的调节数据（这应该是绝对角度）
            user_adjusted_pose = self.adjusted_poses[self.current_frame_idx].clone()

            # 直接使用用户调节的绝对角度作为最终预测姿态
            # （这里假设用户调节是想要的最终角度）
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

            reset_base_pose = torch.tensor(self.poses[reset_base_frame:reset_base_frame + 1], dtype=torch.float32)
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
            prev_gt_pose = torch.tensor(self.poses[prev_actual_frame:prev_actual_frame + 1], dtype=torch.float32)

            # 获取上一帧的调节后姿态
            prev_adjusted_pose = self.adjusted_poses[prev_frame_idx]

            # 计算调节变化量（相对调节量）
            adjustment_delta = prev_adjusted_pose - prev_gt_pose

            # 获取当前帧的GT姿态
            current_actual_frame = self.frame_indices[self.current_frame_idx]
            current_gt_pose = torch.tensor(self.poses[current_actual_frame:current_actual_frame + 1],
                                           dtype=torch.float32)

            # 将调节变化量应用到当前帧GT上
            current_adjusted_pose = current_gt_pose + adjustment_delta

            # 保存到当前帧
            self.adjusted_poses[self.current_frame_idx] = current_adjusted_pose
            return True
        return False

    def create_body_mesh(self, pose: torch.Tensor, material: pyrender.Material) -> pyrender.Mesh:
        """
        创建人体3D网格
        
        :param pose: 姿态参数
        :param material: 渲染材质
        :return: pyrender.Mesh对象
        """
        betas_tensor = torch.tensor(self.betas[:10][None], dtype=torch.float32)
        transl = torch.zeros(1, 3)

        output = self.model(
            betas=betas_tensor,
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:66],
            left_hand_pose=pose[:, 66:111],
            right_hand_pose=pose[:, 111:],
            transl=transl
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        body_mesh = trimesh.Trimesh(vertices, self.model.faces)
        return pyrender.Mesh.from_trimesh(body_mesh, material=material, smooth=False)

    def render_single_frame(self, frame_idx: int) -> np.ndarray:
        """
        渲染单帧双层模型（GT + 预测偏移后）
        
        :param frame_idx: 帧索引
        :return: 渲染的图像数组
        """
        # 临时切换到指定帧来获取正确的姿态
        original_frame_idx = self.current_frame_idx
        self.current_frame_idx = frame_idx

        # 使用get_current_poses方法获取正确的GT和预测姿态（包含偏移）
        gt_pose, predicted_pose = self.get_current_poses()

        # 恢复原始帧索引
        self.current_frame_idx = original_frame_idx

        # 创建材质
        gt_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            baseColorFactor=(74 / 255, 84 / 255, 153 / 255, 0.7)  # 蓝色，半透明
        )

        pred_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            baseColorFactor=(153 / 255, 84 / 255, 74 / 255, 0.8)  # 棕色，半透明
        )

        # 创建网格
        gt_mesh = self.create_body_mesh(gt_pose, gt_material)
        pred_mesh = self.create_body_mesh(predicted_pose, pred_material)

        # 创建场景
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0, 1.0])

        # 添加网格
        scene.add(gt_mesh)
        scene.add(pred_mesh)

        # 添加光源
        directional_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]])
        scene.add(directional_light, pose=light_pose)

        fill_light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.9], intensity=2.0)
        fill_light_pose = np.eye(4)
        fill_light_pose[:3, 3] = np.array([2, 1, 2])
        scene.add(fill_light, pose=fill_light_pose)

        # 设置相机
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        scene.add(camera, pose=self.camera_pose)

        # 渲染
        renderer = pyrender.OffscreenRenderer(RENDER_WIDTH, RENDER_HEIGHT)
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
        sequence_width = RENDER_WIDTH * self.num_frames
        sequence_height = RENDER_HEIGHT
        sequence_image = np.zeros((sequence_height, sequence_width, 3), dtype=np.uint8)

        for i, frame in enumerate(frames):
            x_start = i * RENDER_WIDTH
            x_end = (i + 1) * RENDER_WIDTH
            sequence_image[:, x_start:x_end, :] = frame

        # 在当前编辑帧周围添加高亮边框
        current_x_start = self.current_frame_idx * RENDER_WIDTH
        current_x_end = (self.current_frame_idx + 1) * RENDER_WIDTH

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
        sequence_width = RENDER_WIDTH * self.num_frames
        sequence_height = RENDER_HEIGHT
        sequence_image = np.zeros((sequence_height, sequence_width, 3), dtype=np.uint8)

        for i, frame in enumerate(frames):
            x_start = i * RENDER_WIDTH
            x_end = (i + 1) * RENDER_WIDTH
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
        return jsonify({'success': False, 'error': str(e)}), 500


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
        return jsonify({'success': False, 'error': str(e)}), 500


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

        if num_frames < 1 or num_frames > 20:
            return jsonify({'success': False, 'error': '帧数必须在1-20范围内'})

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
    """初始化全局引擎实例"""
    global engine
    # 序列配置（用户启动时的参数）
    START_FRAME = 40
    TARGET_TIME_INTERVAL_MS = 100  # 目标时间间隔（毫秒）
    FRAME_INTERVAL = 20  # 手动帧间隔（如果不使用时间间隔）
    NUM_FRAMES = 11
    FRAME_OFFSET = 0  # 预测数据相对于GT的偏移帧数

    # 计算实际使用的帧间隔
    if TARGET_TIME_INTERVAL_MS > 0:
        # 快速获取帧率信息，避免重复加载数据
        print(f'🔄 加载数据集: {DATA_SET}')
        data = np.load(DATA_SET)
        try:
            mocap_framerate = float(data['mocap_framerate'])
        except KeyError:
            mocap_framerate = 30.0  # 默认帧率

        frame_time_ms = 1000.0 / mocap_framerate  # 每帧时间（毫秒）
        calculated_frame_interval = max(1, round(TARGET_TIME_INTERVAL_MS / frame_time_ms))
        actual_time_interval_ms = calculated_frame_interval * frame_time_ms

        print(f'⏱️ 时间间隔控制: 目标={TARGET_TIME_INTERVAL_MS}ms, 实际={actual_time_interval_ms:.1f}ms')
        print(f'📐 自动计算帧间隔: {calculated_frame_interval} (覆盖手动设置={FRAME_INTERVAL})')

        frame_interval_to_use = calculated_frame_interval
    else:
        print(f'📐 使用手动设置帧间隔: {FRAME_INTERVAL}')
        frame_interval_to_use = FRAME_INTERVAL

    # 创建引擎实例
    engine = PoseAdjusterEngine(START_FRAME, frame_interval_to_use, NUM_FRAMES, FRAME_OFFSET)

    print('🚀 启动交互式人体姿态调节器...')
    print(f'📂 数据文件: {DATA_SET}')
    print(f'🤖 模型文件: {SMPL_MODEL}')
    print(f'🎯 帧序列: {engine.frame_indices}')
    print(f'⚡ 帧偏移: {FRAME_OFFSET}')
    print('🌐 服务器地址: http://localhost:5000')
    print('-' * 50)


if __name__ == '__main__':
    initialize_engine()
    app.run(debug=True, host='0.0.0.0', port=5000)
