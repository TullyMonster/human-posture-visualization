"""
智能模型选择器
根据可用模型文件和数据集类型自动选择最佳的模型和数据格式组合
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List


@dataclass
class ModelConfig:
    """模型配置类"""
    model_type: str  # 'SMPL' 或 'SMPLX'
    model_path: Path
    gender: str  # 'MALE', 'FEMALE', 'NEUTRAL'
    pose_dim: int  # 期望的姿态参数维度
    supports_hands: bool  # 是否支持手部
    supports_face: bool  # 是否支持面部


class ModelSelector:
    """智能模型选择器"""

    # 标准模型文件名模式
    SMPL_PATTERNS = [
        'SMPL_MALE.pkl', 'SMPL_MALE.npz',
        'SMPL_FEMALE.pkl', 'SMPL_FEMALE.npz',
        'SMPL_NEUTRAL.pkl', 'SMPL_NEUTRAL.npz'
    ]

    SMPLX_PATTERNS = [
        'SMPLX_MALE.pkl', 'SMPLX_MALE.npz',
        'SMPLX_FEMALE.pkl', 'SMPLX_FEMALE.npz',
        'SMPLX_NEUTRAL.pkl', 'SMPLX_NEUTRAL.npz'
    ]

    def __init__(self, models_dir: Path = Path('./models')):
        """
        初始化模型选择器
        
        :param models_dir: 模型文件根目录
        """
        self.models_dir = Path(models_dir)
        self.available_models = self._scan_available_models()

    def _scan_available_models(self) -> Dict[str, List[ModelConfig]]:
        """
        扫描可用的模型文件
        
        :return: 按类型分组的可用模型配置
        """
        available = {'SMPL': [], 'SMPLX': []}

        # 扫描SMPL模型
        smpl_dir = self.models_dir / 'smpl'
        if smpl_dir.exists():
            for pattern in self.SMPL_PATTERNS:
                model_path = smpl_dir / pattern
                if model_path.exists():
                    gender = self._extract_gender_from_filename(pattern)
                    config = ModelConfig(
                        model_type='SMPL',
                        model_path=model_path,
                        gender=gender,
                        pose_dim=72,
                        supports_hands=False,
                        supports_face=False
                    )
                    available['SMPL'].append(config)

        # 扫描SMPLX模型
        smplx_dir = self.models_dir / 'smplx'
        if smplx_dir.exists():
            for pattern in self.SMPLX_PATTERNS:
                model_path = smplx_dir / pattern
                if model_path.exists():
                    gender = self._extract_gender_from_filename(pattern)
                    config = ModelConfig(
                        model_type='SMPLX',
                        model_path=model_path,
                        gender=gender,
                        pose_dim=156,
                        supports_hands=True,
                        supports_face=True
                    )
                    available['SMPLX'].append(config)

        return available

    def _extract_gender_from_filename(self, filename: str) -> str:
        """从文件名提取性别信息"""
        filename_upper = filename.upper()
        if 'MALE' in filename_upper and 'FEMALE' not in filename_upper:
            return 'MALE'
        elif 'FEMALE' in filename_upper:
            return 'FEMALE'
        elif 'NEUTRAL' in filename_upper:
            return 'NEUTRAL'
        return 'NEUTRAL'  # 默认

    def get_optimal_config(self,
                           dataset_type: str,
                           preferred_gender: str = 'MALE',
                           force_model_type: Optional[str] = None) -> Tuple[ModelConfig, str]:
        """
        获取最佳的模型配置和数据转换策略
        
        :param dataset_type: 数据集类型 ('AMASS', '3DPW', 'HuMMan')
        :param preferred_gender: 偏好的性别 ('MALE', 'FEMALE', 'NEUTRAL')
        :param force_model_type: 强制指定模型类型 ('SMPL', 'SMPLX', None)
        :return: (最佳模型配置, 推荐的数据转换策略)
        """

        # 数据集特性分析
        dataset_characteristics = {
            'AMASS': {'native_format': 'SMPLX', 'has_hands': True, 'has_face': True},
            '3DPW': {'native_format': 'SMPL', 'has_hands': False, 'has_face': False},
            'HuMMan': {'native_format': 'SMPL', 'has_hands': False, 'has_face': False}
        }

        dataset_info = dataset_characteristics.get(dataset_type, {
            'native_format': 'SMPL', 'has_hands': False, 'has_face': False
        })

        # 选择策略
        if force_model_type:
            # 强制指定模型类型
            candidates = self.available_models.get(force_model_type, [])
            if not candidates:
                raise ValueError(f'强制指定的模型类型 {force_model_type} 不可用')

            model_config = self._select_best_gender(candidates, preferred_gender)
            strategy = self._determine_conversion_strategy(dataset_info, model_config)

        elif dataset_info['native_format'] == 'SMPLX' and self.available_models['SMPLX']:
            # AMASS数据优先使用SMPLX模型
            model_config = self._select_best_gender(self.available_models['SMPLX'], preferred_gender)
            strategy = 'keep_smplx'  # 保持SMPLX格式

        elif self.available_models['SMPLX']:
            # 如果有SMPLX模型，转换数据以充分利用功能
            model_config = self._select_best_gender(self.available_models['SMPLX'], preferred_gender)
            strategy = 'convert_to_smplx'  # 转换为SMPLX格式

        elif self.available_models['SMPL']:
            # 只有SMPL模型时，保持SMPL格式
            model_config = self._select_best_gender(self.available_models['SMPL'], preferred_gender)
            strategy = 'keep_smpl'  # 保持SMPL格式

        else:
            raise ValueError('没有找到可用的模型文件，请检查models目录')

        return model_config, strategy

    def _select_best_gender(self, candidates: List[ModelConfig], preferred_gender: str) -> ModelConfig:
        """从候选模型中选择最佳性别匹配"""

        # 统一转换为大写进行匹配
        preferred_gender_upper = preferred_gender.upper()
        
        # 首先尝试精确匹配
        for config in candidates:
            if config.gender == preferred_gender_upper:
                return config

        # 如果没有精确匹配，按优先级选择
        gender_priority = ['NEUTRAL', 'MALE', 'FEMALE']
        for gender in gender_priority:
            for config in candidates:
                if config.gender == gender:
                    return config

        # 如果都没有，返回第一个
        return candidates[0]

    def _determine_conversion_strategy(self, dataset_info: Dict, model_config: ModelConfig) -> str:
        """确定数据转换策略"""

        if model_config.model_type == 'SMPLX':
            if dataset_info['native_format'] == 'SMPLX':
                return 'keep_smplx'
            else:
                return 'convert_to_smplx'
        else:  # SMPL
            if dataset_info['native_format'] == 'SMPL':
                return 'keep_smpl'
            else:
                return 'convert_to_smpl'

    def get_recommended_pose_dimension(self, strategy: str) -> int:
        """根据策略获取推荐的姿态参数维度"""
        strategy_dimensions = {
            'keep_smplx': 156,
            'convert_to_smplx': 156,
            'keep_smpl': 72,
            'convert_to_smpl': 72
        }
        return strategy_dimensions.get(strategy, 156)

    def print_available_models(self):
        """打印可用模型的详细信息"""
        print("🔍 可用模型扫描结果:")
        print("-" * 50)

        for model_type, configs in self.available_models.items():
            print(f"\n📁 {model_type} 模型:")
            if configs:
                for config in configs:
                    print(f"  ✅ {config.gender} - {config.model_path}")
                    print(
                        f"     姿态维度: {config.pose_dim}, 手部: {config.supports_hands}, 面部: {config.supports_face}")
            else:
                print(f"  ❌ 未找到 {model_type} 模型文件")

    def get_config_summary(self, model_config: ModelConfig, strategy: str) -> str:
        """获取配置摘要信息"""
        return (f"模型: {model_config.model_type} ({model_config.gender}), "
                f"策略: {strategy}, "
                f"维度: {model_config.pose_dim}, "
                f"路径: {model_config.model_path}")


def auto_select_model_config(dataset_type: str,
                             models_dir: Path = Path('./models'),
                             preferred_gender: str = 'MALE') -> Tuple[ModelConfig, str]:
    """
    便捷函数：自动选择最佳模型配置
    
    :param dataset_type: 数据集类型
    :param models_dir: 模型目录 
    :param preferred_gender: 偏好性别
    :return: (模型配置, 转换策略)
    """
    selector = ModelSelector(models_dir)
    return selector.get_optimal_config(dataset_type, preferred_gender)


if __name__ == '__main__':
    # 测试功能
    print("🚀 模型选择器测试")

    selector = ModelSelector()
    selector.print_available_models()

    # 测试不同数据集的推荐配置
    test_datasets = ['AMASS', '3DPW', 'HuMMan']

    print(f"\n🎯 推荐配置测试:")
    for dataset in test_datasets:
        try:
            config, strategy = selector.get_optimal_config(dataset)
            summary = selector.get_config_summary(config, strategy)
            print(f"\n{dataset}: {summary}")
        except Exception as e:
            print(f"\n{dataset}: ❌ {str(e)}")
