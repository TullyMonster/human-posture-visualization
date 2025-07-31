"""
数据集适配器模块
智能转换不同格式的数据集，根据可用模型自动选择最佳格式
支持：AMASS (SMPLX), 3DPW (SMPL), HuMMan (SMPL)
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

from model_selector import ModelConfig, auto_select_model_config


class DatasetAdapter:
    """
    数据集适配器：将不同格式的数据集转换为统一的SMPLX格式
    """

    SUPPORTED_DATASETS = ['AMASS', '3DPW', 'HuMMan']

    @staticmethod
    def detect_dataset_type(data_path: Path, data: Optional[Dict] = None) -> str:
        """
        自动检测数据集类型
        
        :param data_path: 数据文件路径
        :param data: 可选的预加载数据
        :return: 数据集类型字符串
        """
        try:
            if data is None:
                data = DatasetAdapter.load_raw_data(data_path)

            # 检查键值来判断数据集类型
            keys = set(data.keys())

            # HuMMan 数据集特征：分离的global_orient和body_pose（优先检查，特征最明显）
            if 'global_orient' in keys and 'body_pose' in keys and 'transl' in keys:
                return 'HuMMan'

            # AMASS 数据集特征：有完整的SMPLX参数
            if 'poses' in keys:
                poses_data = data.get('poses', [])
                try:
                    # 安全地获取姿态维度
                    pose_dim = DatasetAdapter._safe_get_pose_dimension(poses_data)

                    if pose_dim >= 156:
                        if 'dmpls' in keys or 'mocap_framerate' in keys:
                            return 'AMASS'
                        # 即使没有特征键，156维以上很可能是AMASS
                        return 'AMASS'
                except Exception:
                    pass

            # 3DPW 数据集特征：有SMPL格式poses + 相机参数
            if 'poses' in keys and 'betas' in keys:
                try:
                    poses_data = data.get('poses', [])
                    pose_dim = DatasetAdapter._safe_get_pose_dimension(poses_data)

                    if pose_dim == 72:
                        # 检查3DPW特有的键
                        if any(k in keys for k in ['cam_poses', 'cam_intrinsics', 'genders']):
                            return '3DPW'
                        # 即使没有相机参数，72维+betas很可能是3DPW或类似SMPL数据
                        return '3DPW'
                except Exception:
                    pass

            # 最后尝试根据文件路径推断
            path_str = str(data_path).lower()
            if '3dpw' in path_str:
                return '3DPW'
            elif 'humman' in path_str:
                return 'HuMMan'
            elif 'amass' in path_str:
                return 'AMASS'

            # 如果都无法识别，默认返回错误
            raise ValueError(f'无法识别数据集类型，路径: {data_path}')

        except Exception as e:
            # 增强错误处理，提供更多调试信息
            available_keys = list(data.keys()) if data else "无法加载数据"
            raise ValueError(f'数据集类型检测失败，路径: {data_path}, 可用键: {available_keys}, 错误: {str(e)}')

    @staticmethod
    def _safe_get_pose_dimension(poses_data) -> int:
        """
        安全地获取姿态数据的维度
        
        :param poses_data: 姿态数据（可能是list、numpy数组等）
        :return: 姿态参数维度
        """
        if poses_data is None:
            return 0

        # 处理numpy数组
        if hasattr(poses_data, 'shape'):
            shape = poses_data.shape
            if len(shape) >= 2:
                return shape[-1]
            elif len(shape) == 1:
                return shape[0]
            return 0

        # 处理Python列表
        if isinstance(poses_data, list) and len(poses_data) > 0:
            try:
                # 转换为numpy数组并获取维度
                poses_array = np.array(poses_data)

                # 处理多人数据（3D数组）
                if len(poses_array.shape) > 2:
                    poses_array = poses_array[0]  # 取第一个人的数据

                if len(poses_array.shape) >= 2:
                    return poses_array.shape[-1]
                elif len(poses_array.shape) == 1:
                    return poses_array.shape[0]

            except Exception:
                # 如果转换失败，尝试直接从列表结构推断
                first_item = poses_data[0]
                if isinstance(first_item, (list, np.ndarray)):
                    if hasattr(first_item, '__len__'):
                        return len(first_item)

        return 0

    @staticmethod
    def load_raw_data(data_path: Path) -> Dict[str, Any]:
        """
        加载原始数据文件
        
        :param data_path: 数据文件路径
        :return: 原始数据字典
        """
        data_path = Path(data_path)

        if data_path.suffix == '.npz':
            return dict(np.load(data_path, allow_pickle=True))
        elif data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        else:
            raise ValueError(f'不支持的文件格式: {data_path.suffix}')

    @staticmethod
    def smart_convert(data_path: Path,
                      models_dir: Path = Path('./models'),
                      preferred_gender: str = 'MALE',
                      dataset_type: Optional[str] = None) -> Tuple[Dict[str, Any], ModelConfig]:
        """
        智能数据转换：根据可用模型自动选择最佳数据格式和模型配置
        
        :param data_path: 数据文件路径
        :param models_dir: 模型文件目录
        :param preferred_gender: 偏好的性别
        :param dataset_type: 数据集类型，如果为None则自动检测
        :return: (转换后的数据, 推荐的模型配置)
        """
        try:
            # 加载原始数据
            raw_data = DatasetAdapter.load_raw_data(data_path)

            # 自动检测数据集类型
            if dataset_type is None:
                dataset_type = DatasetAdapter.detect_dataset_type(data_path, raw_data)

            # 获取最佳模型配置和转换策略
            model_config, strategy = auto_select_model_config(
                dataset_type, models_dir, preferred_gender
            )

            # 根据策略进行数据转换
            if strategy == 'keep_smplx':
                # 保持SMPLX格式，主要进行验证和标准化
                converted_data = DatasetAdapter._convert_amass(raw_data)
            elif strategy == 'convert_to_smplx':
                # 转换为SMPLX格式
                if dataset_type == '3DPW':
                    converted_data = DatasetAdapter._convert_3dpw(raw_data)
                elif dataset_type == 'HuMMan':
                    converted_data = DatasetAdapter._convert_humman(raw_data)
                else:
                    converted_data = DatasetAdapter._convert_generic(raw_data, data_path)
            elif strategy == 'keep_smpl':
                # 保持SMPL格式
                converted_data = DatasetAdapter._convert_to_smpl_format(raw_data, dataset_type)
            elif strategy == 'convert_to_smpl':
                # 转换为SMPL格式
                converted_data = DatasetAdapter._convert_to_smpl_format(raw_data, dataset_type)
            else:
                raise ValueError(f'未知的转换策略: {strategy}')

            # 验证转换结果
            expected_dim = model_config.pose_dim
            actual_dim = converted_data['poses'].shape[1]

            if actual_dim != expected_dim:
                print(f'⚠️ 维度不匹配: 期望{expected_dim}维，实际{actual_dim}维，进行调整...')
                if expected_dim == 156 and actual_dim == 72:
                    # 扩展SMPL到SMPLX
                    converted_data['poses'] = DatasetAdapter._extend_poses_to_smplx(converted_data['poses'])
                elif expected_dim == 72 and actual_dim == 156:
                    # 压缩SMPLX到SMPL
                    converted_data['poses'] = converted_data['poses'][:, :72]
                    print(f'📉 SMPLX→SMPL: 丢弃手部和面部数据')

            return converted_data, model_config

        except Exception as e:
            print(f'❌ 智能转换失败: {str(e)}')
            print(f'🔄 回退到通用转换...')

            # 回退到通用转换
            try:
                raw_data = DatasetAdapter.load_raw_data(data_path)
                converted_data = DatasetAdapter._convert_generic(raw_data, data_path)

                # 创建默认模型配置
                default_config = ModelConfig(
                    model_type='SMPLX',
                    model_path=models_dir / 'smplx' / 'SMPLX_MALE.npz',
                    gender='MALE',
                    pose_dim=156,
                    supports_hands=True,
                    supports_face=True
                )

                return converted_data, default_config

            except Exception as fallback_error:
                raise RuntimeError(f'智能转换完全失败: 原始错误={str(e)}, 回退错误={str(fallback_error)}')

    @staticmethod
    def convert_to_smplx_format(data_path: Path,
                                dataset_type: Optional[str] = None) -> Dict[str, Any]:
        """
        将数据集转换为SMPLX格式
        
        :param data_path: 数据文件路径
        :param dataset_type: 数据集类型，如果为None则自动检测
        :return: 转换后的SMPLX格式数据
        """
        try:
            # 加载原始数据
            raw_data = DatasetAdapter.load_raw_data(data_path)

            # 自动检测数据集类型
            if dataset_type is None:
                dataset_type = DatasetAdapter.detect_dataset_type(data_path, raw_data)

            # 根据类型调用相应的转换函数
            if dataset_type == 'AMASS':
                return DatasetAdapter._convert_amass(raw_data)
            elif dataset_type == '3DPW':
                return DatasetAdapter._convert_3dpw(raw_data)
            elif dataset_type == 'HuMMan':
                return DatasetAdapter._convert_humman(raw_data)
            else:
                raise ValueError(f'不支持的数据集类型: {dataset_type}')

        except Exception as e:
            # 如果适配器处理失败，尝试通用方式处理
            print(f'⚠️ 数据集适配器处理失败: {str(e)}')
            print(f'🔄 尝试通用数据格式处理...')

            try:
                raw_data = DatasetAdapter.load_raw_data(data_path)
                return DatasetAdapter._convert_generic(raw_data, data_path)
            except Exception as fallback_error:
                # 最终失败，抛出详细错误信息
                raise RuntimeError(f'数据集转换完全失败: 原始错误={str(e)}, 回退错误={str(fallback_error)}')

    @staticmethod
    def _convert_generic(data: Dict[str, Any], data_path: Path) -> Dict[str, Any]:
        """
        通用数据格式转换（当自动检测失败时的回退方案）
        
        :param data: 原始数据
        :param data_path: 数据文件路径  
        :return: 尽力转换的SMPLX格式数据
        """
        converted = {}

        print(f'📋 可用数据键: {list(data.keys())}')

        # 尝试获取姿态参数
        if 'poses' in data:
            poses = data['poses']

            # 处理不规则多人数据（在转换为numpy数组之前）
            if isinstance(poses, list) and len(poses) > 0:
                # 检查是否是多人数据（list的每个元素是一个人的数据）
                if isinstance(poses[0], (list, np.ndarray)):
                    poses = poses[0]  # 取第一个人
                    print(f'⚠️ 检测到不规则多人数据，使用第一个人的数据')

            # 处理各种可能的格式
            if isinstance(poses, list):
                poses = np.array(poses)

            # 处理额外的多人数据维度
            if len(poses.shape) > 2:
                poses = poses[0]  # 取第一个人

            # 根据维度判断并扩展
            if poses.shape[-1] == 72:
                # SMPL格式，扩展到SMPLX
                poses = DatasetAdapter._extend_poses_to_smplx(poses)
                print(f'🔄 SMPL格式检测，扩展72维→156维')
            elif poses.shape[-1] >= 156:
                # 已经是SMPLX格式
                poses = poses[:, :156]  # 确保是156维
                print(f'✅ SMPLX格式检测')
            else:
                # 未知格式，尝试扩展
                poses = DatasetAdapter._extend_poses_to_smplx(poses)
                print(f'⚠️ 未知姿态维度 {poses.shape[-1]}，尝试扩展到156维')

            converted['poses'] = poses.astype(np.float32)

        elif 'global_orient' in data and 'body_pose' in data:
            # HuMMan格式
            print(f'🔄 检测到分离姿态参数，重组中...')
            global_orient = np.array(data['global_orient'])
            body_pose = np.array(data['body_pose'])

            # 处理不同格式的body_pose
            if len(body_pose.shape) == 3 and body_pose.shape[1:] == (23, 3):
                # 格式：(N, 23, 3) -> (N, 69)
                body_pose = body_pose.reshape(body_pose.shape[0], -1)
                print(f'🔄 重塑body_pose: (N, 23, 3) → (N, 69)')
            
            smpl_poses = np.concatenate([global_orient, body_pose], axis=-1)
            smplx_poses = DatasetAdapter._extend_poses_to_smplx(smpl_poses)
            converted['poses'] = smplx_poses.astype(np.float32)
        else:
            raise ValueError('无法找到有效的姿态参数')

        # 尝试获取形状参数
        if 'betas' in data:
            betas = data['betas']

            # 处理不规则多人betas数据（在转换为numpy数组之前）
            if isinstance(betas, list) and len(betas) > 0:
                # 检查是否是多人数据
                if isinstance(betas[0], (list, np.ndarray)):
                    betas = betas[0]  # 取第一个人

            if isinstance(betas, list):
                betas = np.array(betas)
            if len(betas.shape) > 1:
                betas = betas[0]  # 取第一个人
            converted['betas'] = betas[:10].astype(np.float32)
        else:
            # 生成默认形状参数
            converted['betas'] = np.zeros(10, dtype=np.float32)
            print(f'⚠️ 未找到形状参数，使用默认值')

        # 尝试获取平移参数
        trans_key = 'transl' if 'transl' in data else 'trans'
        if trans_key in data:
            trans = data[trans_key]
            if isinstance(trans, list):
                trans = np.array(trans)
            if len(trans.shape) > 2:
                trans = trans[0]  # 取第一个人
            converted['trans'] = trans.astype(np.float32)
        else:
            # 生成零平移
            num_frames = converted['poses'].shape[0]
            converted['trans'] = np.zeros((num_frames, 3), dtype=np.float32)
            print(f'⚠️ 未找到平移参数，使用零平移')

        # 设置默认属性
        converted['gender'] = 'neutral'
        converted['mocap_framerate'] = 30.0

        print(f'✅ 通用转换完成: {converted["poses"].shape[0]}帧, {converted["poses"].shape[1]}维')
        return converted

    @staticmethod
    def _convert_to_smpl_format(data: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
        """
        转换数据为SMPL格式（72维）
        
        :param data: 原始数据
        :param dataset_type: 数据集类型
        :return: SMPL格式数据
        """
        converted = {}

        # 获取姿态参数并转换为72维
        if 'poses' in data:
            poses = data['poses']
            if isinstance(poses, list):
                poses = np.array(poses)

            # 处理多人数据
            if len(poses.shape) > 2:
                poses = poses[0]

            # 根据原始维度处理
            if poses.shape[-1] >= 156:
                # SMPLX格式，提取前72维（身体部分）
                poses = poses[:, :72]
                print(f'📉 SMPLX→SMPL: 提取身体姿态，丢弃手部和面部数据')
            elif poses.shape[-1] == 72:
                # 已经是SMPL格式
                print(f'✅ 保持SMPL格式')
            else:
                # 其他维度，尝试扩展或截取
                if poses.shape[-1] < 72:
                    # 扩展到72维
                    target_poses = np.zeros((poses.shape[0], 72), dtype=poses.dtype)
                    target_poses[:, :poses.shape[-1]] = poses
                    poses = target_poses
                    print(f'📈 扩展到SMPL格式: {poses.shape[-1]}→72维')
                else:
                    # 截取到72维
                    poses = poses[:, :72]
                    print(f'📉 截取到SMPL格式: {poses.shape[-1]}→72维')

            converted['poses'] = poses.astype(np.float32)

        elif 'global_orient' in data and 'body_pose' in data:
            # HuMMan格式，直接组合
            global_orient = np.array(data['global_orient'])
            body_pose = np.array(data['body_pose'])

            # 处理不同格式的body_pose
            if len(body_pose.shape) == 3 and body_pose.shape[1:] == (23, 3):
                # 格式：(N, 23, 3) -> (N, 69)
                body_pose = body_pose.reshape(body_pose.shape[0], -1)
                print(f'🔄 重塑body_pose: (N, 23, 3) → (N, 69)')
            elif body_pose.shape[-1] > 69:
                # 截取body_pose到适当维度（69维）
                body_pose = body_pose[:, :69]

            poses = np.concatenate([global_orient, body_pose], axis=-1)
            converted['poses'] = poses.astype(np.float32)
            print(f'🔄 重组为SMPL格式: {poses.shape[-1]}维')
        else:
            raise ValueError('无法找到有效的姿态参数')

        # 处理其他参数（与通用转换类似）
        if 'betas' in data:
            betas = data['betas']
            if isinstance(betas, list):
                betas = np.array(betas)
            if len(betas.shape) > 1:
                betas = betas[0]
            converted['betas'] = betas[:10].astype(np.float32)
        else:
            converted['betas'] = np.zeros(10, dtype=np.float32)

        # 平移参数
        trans_key = 'transl' if 'transl' in data else 'trans'
        if trans_key in data:
            trans = data[trans_key]
            if isinstance(trans, list):
                trans = np.array(trans)
            if len(trans.shape) > 2:
                trans = trans[0]
            converted['trans'] = trans.astype(np.float32)
        else:
            num_frames = converted['poses'].shape[0]
            converted['trans'] = np.zeros((num_frames, 3), dtype=np.float32)

        # 默认属性（注意：这里的gender是数据的性别属性，不影响模型选择）
        converted['gender'] = 'neutral'  
        converted['mocap_framerate'] = 30.0

        print(f'✅ SMPL格式转换完成: {converted["poses"].shape[0]}帧, {converted["poses"].shape[1]}维')
        return converted

    @staticmethod
    def _convert_amass(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换AMASS数据集（已经是SMPLX格式，主要做验证和标准化）
        
        :param data: AMASS原始数据
        :return: 标准化的SMPLX格式数据
        """
        converted = {}

        # 姿态参数
        poses = data['poses']
        if poses.shape[-1] < 156:
            warnings.warn(f'AMASS姿态参数维度不足: {poses.shape[-1]}, 期望156维')
            # 扩展到156维
            poses = DatasetAdapter._extend_poses_to_smplx(poses)

        converted['poses'] = poses.astype(np.float32)

        # 形状参数
        betas = data['betas']
        if len(betas.shape) == 1:
            # 如果是1维数组，保持原样
            converted['betas'] = betas[:10].astype(np.float32)
        else:
            # 如果是2维数组，取第一行
            converted['betas'] = betas[0, :10].astype(np.float32)

        # 全局平移
        if 'trans' in data:
            converted['trans'] = data['trans'].astype(np.float32)
        else:
            # 生成零平移
            converted['trans'] = np.zeros((poses.shape[0], 3), dtype=np.float32)

        # 性别信息
        if 'gender' in data:
            if isinstance(data['gender'], (bytes, np.bytes_)):
                converted['gender'] = data['gender'].decode('utf-8')
            else:
                converted['gender'] = str(data['gender'])
        else:
            converted['gender'] = 'neutral'

        # 帧率信息
        converted['mocap_framerate'] = float(data.get('mocap_framerate', 30.0))

        # DMPL参数（如果存在）
        if 'dmpls' in data:
            converted['dmpls'] = data['dmpls'].astype(np.float32)

        print(f'✅ AMASS转换完成: {poses.shape[0]}帧, {poses.shape[1]}维姿态参数')
        return converted

    @staticmethod
    def _convert_3dpw(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换3DPW数据集（SMPL格式 → SMPLX格式）
        
        :param data: 3DPW原始数据
        :return: SMPLX格式数据
        """
        converted = {}

        # 姿态参数：从72维扩展到156维
        smpl_poses = data['poses']  # (N, 72) 或 list

        # 处理3DPW数据可能是不规则多人格式的情况
        if isinstance(smpl_poses, list) and len(smpl_poses) > 0:
            # 检查是否是多人数据（list的每个元素是一个人的数据）
            if isinstance(smpl_poses[0], (list, np.ndarray)):
                smpl_poses = smpl_poses[0]  # 取第一个人

        # 确保是numpy数组
        if not isinstance(smpl_poses, np.ndarray):
            smpl_poses = np.array(smpl_poses)

        # 处理还有额外维度的情况
        if len(smpl_poses.shape) > 2:
            smpl_poses = smpl_poses[0]  # 取第一个人的数据

        if smpl_poses.shape[-1] != 72:
            raise ValueError(f'3DPW姿态参数维度错误: {smpl_poses.shape[-1]}, 期望72维')

        smplx_poses = DatasetAdapter._extend_poses_to_smplx(smpl_poses)
        converted['poses'] = smplx_poses.astype(np.float32)

        # 形状参数
        betas = data['betas']

        # 处理betas可能是不规则多人格式的情况
        if isinstance(betas, list) and len(betas) > 0:
            # 检查是否是多人数据（list的每个元素是一个人的数据）
            if isinstance(betas[0], (list, np.ndarray)):
                betas = betas[0]  # 取第一个人

        # 确保是numpy数组
        if not isinstance(betas, np.ndarray):
            betas = np.array(betas)

        if len(betas.shape) > 1:
            # 多人数据，取第一个人
            converted['betas'] = betas[0, :10].astype(np.float32)
        else:
            converted['betas'] = betas[:10].astype(np.float32)

        # 全局平移
        trans = data.get('trans', np.zeros((smpl_poses.shape[0], 3)))

        # 确保是numpy数组
        if not isinstance(trans, np.ndarray):
            trans = np.array(trans)

        if len(trans.shape) > 2:
            # 多人数据，取第一个人
            trans = trans[0]
        converted['trans'] = trans.astype(np.float32)

        # 性别信息
        if 'genders' in data:
            genders = data['genders']
            if isinstance(genders, (list, np.ndarray)) and len(genders) > 0:
                gender = genders[0]
                if isinstance(gender, (bytes, np.bytes_)):
                    converted['gender'] = gender.decode('utf-8')
                else:
                    converted['gender'] = str(gender)
            else:
                converted['gender'] = 'neutral'
        else:
            converted['gender'] = 'neutral'

        # 帧率信息（3DPW通常没有，使用默认值）
        converted['mocap_framerate'] = 30.0

        return converted

    @staticmethod
    def _convert_humman(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换HuMMan数据集（分离的SMPL格式 → SMPLX格式）
        
        :param data: HuMMan原始数据
        :return: SMPLX格式数据
        """
        converted = {}

        # 重组姿态参数：global_orient + body_pose
        global_orient = data['global_orient']  # (N, 3)
        body_pose = data['body_pose']  # (N, 69) 或 (N, 23, 3)

        # 检查维度
        if global_orient.shape[-1] != 3:
            raise ValueError(f'HuMMan global_orient维度错误: {global_orient.shape[-1]}, 期望3维')

        # 处理不同格式的body_pose
        if len(body_pose.shape) == 3 and body_pose.shape[1:] == (23, 3):
            # 格式：(N, 23, 3) -> (N, 69)
            body_pose = body_pose.reshape(body_pose.shape[0], -1)
            print(f'🔄 重塑body_pose: (N, 23, 3) → (N, 69)')
        elif body_pose.shape[-1] != 69:
            raise ValueError(f'HuMMan body_pose维度错误: {body_pose.shape}, 期望(N, 69)或(N, 23, 3)')

        # 合并为72维SMPL姿态参数
        smpl_poses = np.concatenate([global_orient, body_pose], axis=-1)  # (N, 72)

        # 扩展到156维SMPLX格式
        smplx_poses = DatasetAdapter._extend_poses_to_smplx(smpl_poses)
        converted['poses'] = smplx_poses.astype(np.float32)

        # 形状参数
        betas = data['betas']
        if len(betas.shape) > 1:
            converted['betas'] = betas[0, :10].astype(np.float32)
        else:
            converted['betas'] = betas[:10].astype(np.float32)

        # 全局平移（注意键名是transl）
        converted['trans'] = data['transl'].astype(np.float32)

        # 性别信息（HuMMan通常没有，使用默认值，注意：这是数据的性别属性）
        converted['gender'] = 'neutral'

        # 帧率信息
        converted['mocap_framerate'] = 30.0

        return converted

    @staticmethod
    def _extend_poses_to_smplx(smpl_poses: np.ndarray) -> np.ndarray:
        """
        将SMPL姿态参数扩展为SMPLX格式
        
        :param smpl_poses: SMPL姿态参数 (N, 72)
        :return: SMPLX姿态参数 (N, 156)
        """
        N = smpl_poses.shape[0]
        smplx_poses = np.zeros((N, 156), dtype=smpl_poses.dtype)

        # 复制SMPL的身体姿态参数
        if smpl_poses.shape[-1] == 72:
            # 标准SMPL格式：前66维是身体姿态（包括全局旋转）
            smplx_poses[:, :66] = smpl_poses[:, :66]
        elif smpl_poses.shape[-1] == 69:
            # body_pose格式：不包含全局旋转
            smplx_poses[:, 3:66] = smpl_poses[:, :63]  # 跳过全局旋转
        else:
            # 直接复制可用的维度
            copy_dims = min(smpl_poses.shape[-1], 66)
            smplx_poses[:, :copy_dims] = smpl_poses[:, :copy_dims]

        # 手部和面部参数保持为零（默认姿态）
        # smplx_poses[:, 66:156] = 0  # 已经是零

        return smplx_poses

    @staticmethod
    def save_converted_data(data: Dict[str, Any], output_path: Path):
        """
        保存转换后的数据
        
        :param data: 转换后的数据
        :param output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.npz':
            np.savez(output_path, **data)
        elif output_path.suffix == '.pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            # 默认保存为npz格式
            output_path = output_path.with_suffix('.npz')
            np.savez(output_path, **data)

        print(f'💾 转换后数据已保存到: {output_path}')


def convert_dataset_file(input_path: str,
                         output_path: Optional[str] = None,
                         dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：转换单个数据集文件
    
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径（可选）
    :param dataset_type: 数据集类型（可选，自动检测）
    :return: 转换后的数据
    """
    input_path = Path(input_path)

    # 转换数据
    converted_data = DatasetAdapter.convert_to_smplx_format(input_path, dataset_type)

    # 保存转换后的数据
    if output_path:
        DatasetAdapter.save_converted_data(converted_data, Path(output_path))

    return converted_data


def batch_convert_datasets(dataset_dir: str,
                           output_dir: str,
                           pattern: str = '*.pkl'):
    """
    批量转换数据集文件
    
    :param dataset_dir: 数据集目录
    :param output_dir: 输出目录
    :param pattern: 文件匹配模式
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(dataset_dir.glob(pattern))
    if not files:
        print(f'❌ 在 {dataset_dir} 中未找到匹配 {pattern} 的文件')
        return

    print(f'🔄 开始批量转换 {len(files)} 个文件...')

    for file_path in files:
        try:
            output_path = output_dir / file_path.name
            converted_data = convert_dataset_file(file_path, output_path)
            print(f'✅ 已转换: {file_path.name}')
        except Exception as e:
            print(f'❌ 转换失败 {file_path.name}: {str(e)}')

    print(f'🎉 批量转换完成！输出目录: {output_dir}')


if __name__ == '__main__':
    # 示例用法
    print('🚀 数据集适配器测试')

    # 测试3DPW数据集转换
    try:
        test_file = Path('./datasets/3DPW/courtyard_basketball_00.pkl')
        if test_file.exists():
            print(f'\n📁 测试转换: {test_file}')
            converted = convert_dataset_file(test_file)
            print(f'✅ 转换成功！姿态参数形状: {converted["poses"].shape}')
        else:
            print(f'⚠️ 测试文件不存在: {test_file}')
    except Exception as e:
        print(f'❌ 测试转换失败: {str(e)}')
