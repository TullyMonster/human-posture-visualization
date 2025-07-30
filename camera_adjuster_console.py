"""
控制台控制版本的相机调整器
支持AMASS、3DPW、HuMMan等多种数据集格式
从config.yaml配置文件读取所有设置
"""

from pathlib import Path

import numpy as np

# NumPy 2.0兼容性补丁
if not hasattr(np, 'infty'):
    np.infty = np.inf

import pyrender
import torch
import trimesh
import yaml

from config_manager import get_config_manager
from dataset_adapter import DatasetAdapter


class MultiDatasetCameraAdjuster:
    """
    多数据集支持的相机调节器
    支持从config.yaml读取配置并自动适配不同数据集格式
    """

    def __init__(self):
        """初始化相机调节器"""
        self.config_manager = get_config_manager()
        self.dataset_config = self.config_manager.get_current_dataset_config()
        self.render_config = self.config_manager.get_render_config()

        # 数据相关
        self.poses = None
        self.betas = None
        self.model = None
        self.model_config = None
        self.total_frames = 0
        self.frame_indices = []

        # 渲染相关
        self.current_frame_idx = 0
        self.scene = None
        self.mesh_node = None
        self.viewer = None
        self.running = True

        # 相机配置
        self.initial_camera_pose = self.config_manager.get_camera_pose_matrix()

        # 帧导航配置
        self.frame_step = 10

        print("=" * 60)
        print("多数据集相机调节器")
        print("=" * 60)
        print(f"数据集: {self.dataset_config.path}")
        print(f"类型: {self.dataset_config.type}")
        print(f"性别: {self.config_manager.get_current_gender()}")
        print("=" * 60)
        print("说明：")
        print("1. 3D窗口显示人体模型，用鼠标调整视角")
        print("2. 在控制台输入命令来切换帧")
        print("3. 输入 'p' 获取当前相机矩阵")
        print("4. 输入 's' 保存相机矩阵到config.yaml")
        print("=" * 60)

    def load_data(self):
        """加载数据和模型（支持多种数据集格式）"""
        print(f'加载数据集: {self.dataset_config.path}')

        try:
            # 使用智能适配器
            data, model_config = DatasetAdapter.smart_convert(
                data_path=self.dataset_config.path,
                models_dir=Path('./models'),
                preferred_gender=self.config_manager.get_current_gender()
            )

            self.poses = data['poses']
            self.betas = data['betas']
            self.model_config = model_config
            self.total_frames = self.poses.shape[0]

            print(f'✅ 智能转换完成: {self.total_frames}帧, {self.poses.shape[1]}维')

        except Exception as e:
            print(f'❌ 智能转换失败: {str(e)}')
            print(f'🔄 回退到标准转换...')

            # 回退到标准转换
            try:
                data = DatasetAdapter.convert_to_smplx_format(self.dataset_config.path)
                self.poses = data['poses']
                self.betas = data['betas']
                self.total_frames = self.poses.shape[0]

                # 使用默认SMPLX模型配置
                from dataset_adapter import ModelConfig
                self.model_config = ModelConfig(
                    model_type='SMPLX',
                    model_path=Path('./models/smplx/SMPLX_NEUTRAL.npz'),
                    gender='neutral',
                    pose_dim=156,
                    supports_hands=True,
                    supports_face=True
                )

            except Exception as fallback_error:
                raise RuntimeError(f'数据加载完全失败: {str(fallback_error)}')

        print(f'数据集形状: poses={self.poses.shape}, betas={self.betas.shape}')

        # 计算帧序列
        self.calculate_frame_sequence()

        # 加载模型
        self.load_model()

    def calculate_frame_sequence(self):
        """计算帧序列"""
        start_frame = self.dataset_config.start_frame
        frame_interval = self.dataset_config.frame_interval
        num_frames = self.dataset_config.num_frames

        # 计算帧序列
        calculated_frames = [start_frame + i * frame_interval for i in range(num_frames)]

        # 过滤有效帧
        self.frame_indices = [f for f in calculated_frames if f < self.total_frames]

        if len(self.frame_indices) == 0:
            self.frame_indices = [self.total_frames - 1]

        print(f'帧序列: {self.frame_indices}')

        # 设置初始帧
        initial_frame = self.dataset_config.start_frame
        if initial_frame in self.frame_indices:
            self.current_frame_idx = self.frame_indices.index(initial_frame)
        else:
            self.current_frame_idx = 0

    def load_model(self):
        """加载推荐的模型"""
        print(f'加载模型: {self.model_config.model_path} ({self.model_config.gender})')

        if self.model_config.model_type == 'SMPLX':
            from smplx import SMPLX
            self.model = SMPLX(
                model_path=str(self.model_config.model_path),
                gender=self.model_config.gender.lower(),
                num_betas=10,
                use_pca=False,
                flat_hand_mean=True
            )
        else:  # SMPL
            from smplx import SMPL
            self.model = SMPL(
                model_path=str(self.model_config.model_path),
                gender=self.model_config.gender.lower(),
                num_betas=10
            )

    def create_mesh(self, frame_sequence_idx: int):
        """创建指定帧的3D网格"""
        actual_frame = self.frame_indices[frame_sequence_idx]
        pose = torch.tensor(self.poses[actual_frame:actual_frame + 1], dtype=torch.float32)
        betas_tensor = torch.tensor(self.betas[:10][None], dtype=torch.float32)
        transl = torch.zeros(1, 3)

        # 根据模型类型调用
        if self.model_config.model_type == 'SMPLX' and pose.shape[1] >= 156:
            output = self.model(
                betas=betas_tensor,
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:66],
                left_hand_pose=pose[:, 66:111],
                right_hand_pose=pose[:, 111:156],
                transl=transl
            )
        else:
            # SMPL或简化SMPLX
            output = self.model(
                betas=betas_tensor,
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:min(66, pose.shape[1])],
                transl=transl
            )

        vertices = output.vertices.detach().cpu().numpy().squeeze()
        body_mesh = trimesh.Trimesh(vertices, self.model.faces)

        # 使用配置中的材质
        gt_material = self.render_config.gt_material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=gt_material['metallic'],
            roughnessFactor=gt_material['roughness'],
            alphaMode='OPAQUE',
            baseColorFactor=gt_material['color']
        )

        return pyrender.Mesh.from_trimesh(body_mesh, material=material, smooth=False)

    def update_frame(self, new_frame_idx: int):
        """更新显示的帧"""
        new_frame_idx = max(0, min(new_frame_idx, len(self.frame_indices) - 1))

        if new_frame_idx != self.current_frame_idx:
            self.current_frame_idx = new_frame_idx
            actual_frame = self.frame_indices[self.current_frame_idx]
            new_mesh = self.create_mesh(self.current_frame_idx)

            if self.mesh_node and self.scene:
                self.mesh_node.mesh = new_mesh
                print(f"已切换到序列帧 {self.current_frame_idx} (实际帧 {actual_frame})")

    def setup_scene(self):
        """设置3D场景"""
        # 创建初始网格
        mesh = self.create_mesh(self.current_frame_idx)

        # 使用配置中的渲染设置
        lighting = self.render_config.lighting
        self.scene = pyrender.Scene(
            ambient_light=lighting['ambient'],
            bg_color=[1.0, 1.0, 1.0, 1.0]
        )
        self.mesh_node = self.scene.add(mesh)

        # 添加光源
        directional_light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=lighting['directional_intensity']
        )
        light_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]])
        self.scene.add(directional_light, pose=light_pose)

        fill_light = pyrender.DirectionalLight(
            color=lighting['fill_light_color'],
            intensity=lighting['fill_light_intensity']
        )
        fill_light_pose = np.eye(4)
        fill_light_pose[:3, 3] = np.array([2, 1, 2])
        self.scene.add(fill_light, pose=fill_light_pose)

    def print_camera_matrix(self):
        """输出当前相机矩阵"""
        if self.viewer and hasattr(self.viewer, '_camera_node'):
            actual_frame = self.frame_indices[self.current_frame_idx]
            print("\n" + "=" * 60)
            print(f"当前序列帧：{self.current_frame_idx} (实际帧: {actual_frame})")
            print("相机矩阵：")
            camera_pose = self.viewer._camera_node.matrix

            # 格式化输出，便于复制到config.yaml
            print("相机矩阵（可直接复制到config.yaml）：")
            print("camera:")
            print("  matrix: [")
            for row in camera_pose:
                formatted_row = "[" + ", ".join([f"{val}" for val in row]) + "]"
                print(f"    {formatted_row},")
            print("  ]")
            print("=" * 60)
        else:
            print("无法获取相机矩阵，请确保3D窗口已打开")

    def save_camera_matrix(self):
        """保存当前相机矩阵到config.yaml"""
        if self.viewer and hasattr(self.viewer, '_camera_node'):
            try:
                camera_pose = self.viewer._camera_node.matrix

                # 读取当前配置
                config_path = Path('config.yaml')
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # 更新相机矩阵
                config['camera']['matrix'] = camera_pose.tolist()

                # 保存配置
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)

                print("✅ 相机矩阵已保存到 config.yaml")

            except Exception as e:
                print(f"❌ 保存相机矩阵失败: {e}")
        else:
            print("无法获取相机矩阵，请确保3D窗口已打开")

    def console_control_loop(self):
        """控制台命令循环"""
        print("\n" + "-" * 60)
        print("控制台命令说明：")
        print("w/up    - 下一帧 (+1)")
        print("s/down  - 上一帧 (-1)")
        print("a/left  - 大幅后退 (-5)")
        print("d/right - 大幅前进 (+5)")
        print("0-9     - 跳转到百分比位置")
        print("f <num> - 跳转到序列帧号")
        print("p       - 输出相机矩阵")
        print("save/s  - 保存相机矩阵到config.yaml")
        print("q/exit  - 退出")
        actual_frame = self.frame_indices[self.current_frame_idx]
        print(f"当前：序列帧{self.current_frame_idx} (实际帧{actual_frame})，共{len(self.frame_indices)}个序列帧")
        print("-" * 60)

        while self.running:
            try:
                actual_frame = self.frame_indices[self.current_frame_idx]
                command = input(f"[序列帧{self.current_frame_idx}/实际帧{actual_frame}] 输入命令: ").strip().lower()

                if command in ['q', 'exit', 'quit']:
                    self.running = False
                    break

                elif command in ['w', 'up']:
                    self.update_frame(self.current_frame_idx + 1)

                elif command in ['s', 'down']:
                    self.update_frame(self.current_frame_idx - 1)

                elif command in ['a', 'left']:
                    self.update_frame(self.current_frame_idx - 5)

                elif command in ['d', 'right']:
                    self.update_frame(self.current_frame_idx + 5)

                elif command.isdigit() and len(command) == 1:
                    percent = int(command) / 9.0
                    target_frame = int(percent * (len(self.frame_indices) - 1))
                    self.update_frame(target_frame)
                    print(f"跳转到{percent * 100:.0f}%位置")

                elif command.startswith('f '):
                    try:
                        target_frame = int(command.split()[1])
                        self.update_frame(target_frame)
                    except (IndexError, ValueError):
                        print("无效的帧号，使用格式：f 0")

                elif command == 'p':
                    self.print_camera_matrix()

                elif command in ['save', 's']:
                    self.save_camera_matrix()

                elif command in ['help', 'h']:
                    print("控制台命令说明：")
                    print("w/up    - 下一帧 (+1)")
                    print("s/down  - 上一帧 (-1)")
                    print("a/left  - 大幅后退 (-5)")
                    print("d/right - 大幅前进 (+5)")
                    print("0-9     - 跳转到百分比位置")
                    print("f <num> - 跳转到序列帧号")
                    print("p       - 输出相机矩阵")
                    print("save/s  - 保存相机矩阵到config.yaml")
                    print("q/exit  - 退出")

                else:
                    print("未知命令，输入 'help' 查看帮助")

            except KeyboardInterrupt:
                print("\n退出...")
                self.running = False
                break
            except Exception as e:
                print(f"错误: {e}")

    def start_viewer(self):
        """启动3D查看器"""
        try:
            print('启动3D查看器...')

            # 创建相机并设置初始位置
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            self.scene.add(camera, pose=self.initial_camera_pose)

            actual_frame = self.frame_indices[self.current_frame_idx]
            self.viewer = pyrender.Viewer(
                self.scene,
                use_raymond_lighting=True,
                viewport_size=(self.render_config.width, self.render_config.height),
                window_title=f"相机调节器 - 序列帧{self.current_frame_idx}/实际帧{actual_frame}",
                run_in_thread=True  # 在后台线程运行
            )
            return True
        except Exception as e:
            print(f"启动查看器失败: {e}")
            return False

    def run(self):
        """主运行函数"""
        try:
            self.load_data()
            self.setup_scene()

            # 启动3D查看器（后台线程）
            if self.start_viewer():
                # 在主线程运行控制台循环
                self.console_control_loop()
            else:
                print("无法启动3D查看器")

        except Exception as e:
            print(f"运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("程序结束")


if __name__ == "__main__":
    adjuster = MultiDatasetCameraAdjuster()
    adjuster.run()
