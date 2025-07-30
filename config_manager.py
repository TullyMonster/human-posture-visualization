"""
配置管理器
负责加载和管理YAML配置文件，提供统一的配置访问接口
支持配置文件热重载功能
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Callable

import numpy as np
import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


@dataclass
class DatasetConfig:
    """数据集配置类"""
    path: Path
    type: str
    gender: str

    # 帧序列参数
    start_frame: int
    frame_interval: int
    num_frames: int
    frame_offset: int
    time_interval_ms: int

    # 相机配置
    camera_matrix: List[List[float]]


@dataclass
class ModelConfig:
    """模型配置类"""
    smpl_paths: Dict[str, str]
    smplx_paths: Dict[str, str]


@dataclass
class RenderConfig:
    """渲染配置类"""
    width: int
    height: int
    gt_material: Dict[str, Any]
    predicted_material: Dict[str, Any]
    lighting: Dict[str, Any]


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监控处理器"""

    def __init__(self, config_manager: 'ConfigManager'):
        """
        初始化配置文件监控器
        
        :param config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.last_modified = 0
        self.debounce_delay = 1.0  # 防抖延时（秒），增加到1秒避免编辑器重复触发

    def on_modified(self, event):
        """
        文件修改事件处理
        
        :param event: 文件系统事件
        """
        if event.is_directory:
            return

        # 检查是否是我们监控的配置文件（使用绝对路径比较）
        if Path(event.src_path).resolve() == self.config_manager.config_path.resolve():
            current_time = time.time()
            # 防抖处理：避免短时间内多次触发
            if current_time - self.last_modified > self.debounce_delay:
                self.last_modified = current_time
                print(f'🔄 检测到配置文件变更: {event.src_path}')
                self.config_manager._reload_config()


class ConfigManager:
    """配置管理器主类"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        :param config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

        # 配置变更回调函数列表
        self._change_callbacks: List[Callable[[], None]] = []

        # 文件监控相关
        self._observer = None
        self._watcher = None
        self._monitoring = False

    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f'配置文件未找到: {self.config_path}')
        except yaml.YAMLError as e:
            raise ValueError(f'配置文件格式错误: {e}')

    def _reload_config(self):
        """重新加载配置文件"""
        try:
            old_config = self.config.copy()
            self.config = self._load_config()
            self._validate_config()

            # 检查配置是否真的发生了变化
            if old_config != self.config:
                print('✅ 配置已更新')
                # 调用所有注册的回调函数
                for callback in self._change_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        print(f'❌ 配置变更回调执行失败: {e}')
            else:
                print('⚡ 配置无变化，跳过重载')

        except Exception as e:
            print(f'❌ 配置文件重新加载失败: {e}')

    def register_change_callback(self, callback: Callable[[], None]):
        """
        注册配置变更回调函数
        
        :param callback: 当配置发生变更时调用的回调函数
        """
        if callback not in self._change_callbacks:
            self._change_callbacks.append(callback)

    def unregister_change_callback(self, callback: Callable[[], None]):
        """
        取消注册配置变更回调函数
        
        :param callback: 要取消的回调函数
        """
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
            print(f'✅ 已取消配置变更回调: {callback.__name__}')

    def start_monitoring(self):
        """开始监控配置文件变化"""
        if self._monitoring:
            print('⚠️ 配置文件监控已在运行')
            return

        try:
            # 创建监控器和观察者
            self._watcher = ConfigFileWatcher(self)
            self._observer = Observer()

            # 监控配置文件所在目录
            config_dir = self.config_path.parent.absolute()
            self._observer.schedule(self._watcher, str(config_dir), recursive=False)

            # 启动监控
            self._observer.start()
            self._monitoring = True

        except Exception as e:
            print(f'❌ 启动配置文件监控失败: {e}')

    def stop_monitoring(self):
        """停止监控配置文件变化"""
        if not self._monitoring:
            # 静默处理，避免重复输出
            return

        try:
            if self._observer:
                self._observer.stop()
                self._observer.join()
                self._observer = None

            self._watcher = None
            self._monitoring = False
            print('🛑 已停止配置文件监控')

        except Exception as e:
            print(f'❌ 停止配置文件监控失败: {e}')

    def is_monitoring(self) -> bool:
        """检查是否正在监控配置文件"""
        return self._monitoring

    def _validate_config(self):
        """验证配置文件的完整性"""
        required_sections = ['dataset', 'frames', 'camera', 'models', 'rendering']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f'配置文件缺少必需部分: {section}')

        # 验证数据集配置必需字段
        dataset_required = ['path', 'type', 'gender']
        for field in dataset_required:
            if field not in self.config['dataset']:
                raise ValueError(f'数据集配置缺少必需字段: {field}')

    def get_current_dataset_config(self) -> DatasetConfig:
        """获取当前数据集配置"""
        dataset_data = self.config['dataset']
        frames_data = self.config['frames']
        camera_data = self.config['camera']

        return DatasetConfig(
            path=Path(dataset_data['path']),
            type=dataset_data['type'],
            gender=dataset_data['gender'],

            start_frame=frames_data['start_frame'],
            frame_interval=frames_data['frame_interval'],
            num_frames=frames_data['num_frames'],
            frame_offset=frames_data['frame_offset'],
            time_interval_ms=frames_data['time_interval_ms'],

            camera_matrix=camera_data['matrix']
        )

    def get_model_config(self) -> ModelConfig:
        """获取模型文件配置"""
        models_data = self.config['models']

        return ModelConfig(
            smpl_paths=models_data['smpl'],
            smplx_paths=models_data['smplx']
        )

    def get_model_path(self, dataset_type: str, gender: str) -> Path:
        """
        根据数据集类型和性别获取模型文件路径
        
        :param dataset_type: 数据集类型 (HuMMan/AMASS/3DPW)
        :param gender: 性别 (male/female/neutral)
        :return: 模型文件路径
        """
        model_config = self.get_model_config()

        # 根据数据集类型确定模型类型
        if dataset_type in ['HuMMan', '3DPW']:
            # HuMMan和3DPW使用SMPL模型
            model_type = 'SMPL'
        elif dataset_type == 'AMASS':
            # AMASS使用SMPLX模型
            model_type = 'SMPLX'
        else:
            # 默认使用SMPLX
            model_type = 'SMPLX'

        if model_type == 'SMPL':
            if gender not in model_config.smpl_paths:
                raise ValueError(f'SMPL模型不支持性别: {gender}')
            return Path(model_config.smpl_paths[gender])
        else:  # SMPLX
            if gender not in model_config.smplx_paths:
                raise ValueError(f'SMPLX模型不支持性别: {gender}')
            return Path(model_config.smplx_paths[gender])

    def get_render_config(self) -> RenderConfig:
        """获取渲染配置"""
        render_data = self.config['rendering']

        return RenderConfig(
            width=render_data['width'],
            height=render_data['height'],
            gt_material=render_data['materials']['gt'],
            predicted_material=render_data['materials']['predicted'],
            lighting=render_data['lighting']
        )

    def get_camera_pose_matrix(self) -> np.ndarray:
        """
        获取相机姿态矩阵
        
        :return: 4x4相机姿态矩阵
        """
        dataset_config = self.get_current_dataset_config()
        return np.array(dataset_config.camera_matrix, dtype=np.float32)

    def get_current_gender(self) -> str:
        """获取当前性别偏好"""
        return self.config['dataset']['gender']

    def get_server_config(self) -> Dict[str, Any]:
        """获取服务器配置"""
        return self.config.get('server', {
            'host': '0.0.0.0',
            'port': 5000,
            'debug': True
        })

    def update_dataset_config(self, path: str = None, dataset_type: str = None, gender: str = None):
        """
        更新数据集配置
        
        :param path: 数据集路径
        :param dataset_type: 数据集类型
        :param gender: 性别
        """
        if path is not None:
            self.config['dataset']['path'] = path
            print(f'✅ 已更新数据集路径: {path}')

        if dataset_type is not None:
            self.config['dataset']['type'] = dataset_type
            print(f'✅ 已更新数据集类型: {dataset_type}')

        if gender is not None:
            self.config['dataset']['gender'] = gender
            print(f'✅ 已更新性别: {gender}')

    def save_config(self):
        """保存当前配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f'✅ 配置已保存到: {self.config_path}')
        except Exception as e:
            print(f'❌ 配置保存失败: {e}')

    def print_current_config(self):
        """打印当前配置摘要"""
        dataset_config = self.get_current_dataset_config()
        model_config = self.get_model_config()
        render_config = self.get_render_config()

        print("\n" + "=" * 60)
        print("📋 当前配置摘要")
        print("=" * 60)

        print(f"📁 数据路径: {dataset_config.path}")
        print(f"📊 数据集类型: {dataset_config.type}")
        print(f"👤 性别: {dataset_config.gender}")
        print(
            f"📊 帧配置: 起始{dataset_config.start_frame}, 间隔{dataset_config.frame_interval}, 数量{dataset_config.num_frames}")

        print(f"📷 相机: 4x4矩阵配置")

        print(f"🖼️ 渲染尺寸: {render_config.width}x{render_config.height}")

        # 推荐的模型路径
        try:
            model_path = self.get_model_path(dataset_config.type, dataset_config.gender)
            print(f"🤖 推荐模型: {model_path}")
        except Exception:
            print(f"🤖 推荐模型: 自动选择")

        print("=" * 60)


# 全局配置管理器实例
_config_manager = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


if __name__ == '__main__':
    # 测试配置管理器
    try:
        config_mgr = ConfigManager()
        config_mgr.print_current_config()

        # 测试更新配置
        print(f"\n🔄 测试更新配置...")
        config_mgr.update_dataset_config(gender="male")

        # 测试相机矩阵生成
        print(f"\n📷 测试相机矩阵生成...")
        camera_matrix = config_mgr.get_camera_pose_matrix()
        print(f"相机矩阵形状: {camera_matrix.shape}")

        # 测试文件监控
        print(f"\n🔍 测试文件监控...")


        def test_callback():
            print("🔔 配置变更回调被触发！")


        config_mgr.register_change_callback(test_callback)
        config_mgr.start_monitoring()

        print("监控运行中，按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在停止监控...")
            config_mgr.stop_monitoring()

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
