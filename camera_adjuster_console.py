"""
控制台控制版本的相机调整器
用于解决pyrender键盘事件不工作的问题
"""

from pathlib import Path

import numpy as np
import pyrender
import torch
import trimesh
from smplx import SMPLX

# ==================== 配置常量 ====================
DATA_SET: Path = Path(r'./G18 push kick right poses.npz')
SMPLX_MALE: Path = Path(r'./models\smplx\SMPLX_MALE.npz')

# 预览配置
INITIAL_FRAME = 106
FRAME_STEP = 10

print("=" * 60)
print("控制台控制版相机调整器")
print("=" * 60)
print("说明：")
print("1. 3D窗口显示人体模型，用鼠标调整视角")
print("2. 在控制台输入命令来切换帧")
print("3. 输入 'p' 获取当前相机矩阵")
print("=" * 60)


class ConsoleControlledViewer:
    """
    控制台控制的查看器
    """

    def __init__(self):
        self.poses = None
        self.betas = None
        self.model = None
        self.current_frame = INITIAL_FRAME
        self.scene = None
        self.mesh_node = None
        self.viewer = None
        self.running = True

    def load_data(self):
        """加载数据和模型"""
        print('加载数据集...')
        data = np.load(DATA_SET)
        self.poses = data['poses']
        self.betas = data['betas']

        print(f'数据集形状: poses={self.poses.shape}, betas={self.betas.shape}')

        # 检查帧索引
        if INITIAL_FRAME >= self.poses.shape[0]:
            print(f'警告: 帧索引 {INITIAL_FRAME} 超出范围，使用第 100 帧')
            self.current_frame = 100

        # 加载SMPLX模型
        print('加载SMPLX模型...')
        self.model = SMPLX(
            model_path=SMPLX_MALE.as_posix(),
            gender="male",
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True
        )

    def create_mesh(self, frame_idx):
        """创建指定帧的3D网格"""
        pose = torch.tensor(self.poses[frame_idx:frame_idx + 1], dtype=torch.float32)
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

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            baseColorFactor=(74 / 255, 84 / 255, 153 / 255, 1.0)
        )

        return pyrender.Mesh.from_trimesh(body_mesh, material=material, smooth=False)

    def update_frame(self, new_frame):
        """更新显示的帧"""
        max_frame = self.poses.shape[0] - 1
        new_frame = max(0, min(new_frame, max_frame))

        if new_frame != self.current_frame:
            self.current_frame = new_frame
            new_mesh = self.create_mesh(self.current_frame)

            if self.mesh_node and self.scene:
                self.mesh_node.mesh = new_mesh
                print(f"已切换到第 {self.current_frame} 帧")

    def setup_scene(self):
        """设置3D场景"""
        # 创建初始网格
        mesh = self.create_mesh(self.current_frame)

        # 创建场景
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0, 1.0])
        self.mesh_node = self.scene.add(mesh)

        # 添加光源
        directional_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]])
        self.scene.add(directional_light, pose=light_pose)

        fill_light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.9], intensity=2.0)
        fill_light_pose = np.eye(4)
        fill_light_pose[:3, 3] = np.array([2, 1, 2])
        self.scene.add(fill_light, pose=fill_light_pose)

    def print_camera_matrix(self):
        """输出当前相机矩阵"""
        if self.viewer and hasattr(self.viewer, '_camera_node'):
            print("\n" + "=" * 60)
            print(f"当前帧：{self.current_frame}")
            print("相机矩阵：")
            camera_pose = self.viewer._camera_node.matrix
            print("camera_pose = np.array([")
            for row in camera_pose:
                print(f"    {list(row)},")
            print("])")
            print("=" * 60)
        else:
            print("无法获取相机矩阵，请确保3D窗口已打开")

    def console_control_loop(self):
        """控制台命令循环"""
        print("\n" + "-" * 60)
        print("控制台命令说明：")
        print("w/up    - 下一帧 (+%d)" % FRAME_STEP)
        print("s/down  - 上一帧 (-%d)" % FRAME_STEP)
        print("a/left  - 大幅后退 (-%d)" % (FRAME_STEP * 5))
        print("d/right - 大幅前进 (+%d)" % (FRAME_STEP * 5))
        print("0-9     - 跳转到百分比位置")
        print("f <num> - 跳转到指定帧号")
        print("p       - 输出相机矩阵")
        print("q/exit  - 退出")
        print(f"当前帧：{self.current_frame}（共{self.poses.shape[0]}帧）")
        print("-" * 60)

        while self.running:
            try:
                command = input(f"[帧{self.current_frame}] 输入命令: ").strip().lower()

                if command in ['q', 'exit', 'quit']:
                    self.running = False
                    break

                elif command in ['w', 'up']:
                    self.update_frame(self.current_frame + FRAME_STEP)

                elif command in ['s', 'down']:
                    self.update_frame(self.current_frame - FRAME_STEP)

                elif command in ['a', 'left']:
                    self.update_frame(self.current_frame - FRAME_STEP * 5)

                elif command in ['d', 'right']:
                    self.update_frame(self.current_frame + FRAME_STEP * 5)

                elif command.isdigit() and len(command) == 1:
                    percent = int(command) / 9.0
                    target_frame = int(percent * (self.poses.shape[0] - 1))
                    self.update_frame(target_frame)
                    print(f"跳转到{percent * 100:.0f}%位置")

                elif command.startswith('f '):
                    try:
                        target_frame = int(command.split()[1])
                        self.update_frame(target_frame)
                    except (IndexError, ValueError):
                        print("无效的帧号，使用格式：f 123")

                elif command == 'p':
                    self.print_camera_matrix()

                elif command == 'help' or command == 'h':
                    print("控制台命令说明：")
                    print("w/up    - 下一帧 (+%d)" % FRAME_STEP)
                    print("s/down  - 上一帧 (-%d)" % FRAME_STEP)
                    print("a/left  - 大幅后退 (-%d)" % (FRAME_STEP * 5))
                    print("d/right - 大幅前进 (+%d)" % (FRAME_STEP * 5))
                    print("0-9     - 跳转到百分比位置")
                    print("f <num> - 跳转到指定帧号")
                    print("p       - 输出相机矩阵")
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
            self.viewer = pyrender.Viewer(
                self.scene,
                use_raymond_lighting=True,
                viewport_size=(800, 800),
                window_title=f"相机调整器 - 帧{self.current_frame}",
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
        finally:
            print("程序结束")


if __name__ == "__main__":
    viewer = ConsoleControlledViewer()
    viewer.run()
