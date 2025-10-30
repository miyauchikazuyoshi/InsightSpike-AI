"""
統合インデックスの設定
"""

from typing import Optional
from pydantic import BaseModel, Field


class IntegratedIndexConfig(BaseModel):
    """統合インデックスの設定"""
    
    # 基本設定
    enabled: bool = Field(
        default=False,
        description="統合インデックスを有効化するか"
    )
    
    dimension: int = Field(
        default=768,
        description="ベクトル次元数"
    )
    
    # 性能設定
    similarity_threshold: float = Field(
        default=0.3,
        description="類似度閾値（グラフエッジ作成用）"
    )
    
    use_faiss: bool = Field(
        default=True,
        description="大規模データでFAISSを使用するか"
    )
    
    faiss_threshold: int = Field(
        default=100000,
        description="FAISS使用開始のベクトル数閾値"
    )
    
    # 移行設定
    migration_mode: str = Field(
        default="shadow",
        description="移行モード: shadow, partial, full"
    )
    
    # 自動保存設定
    auto_save: bool = Field(
        default=True,
        description="自動保存を有効化するか"
    )
    
    save_interval: int = Field(
        default=1000,
        description="自動保存の間隔（エピソード数）"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "dimension": 768,
                "similarity_threshold": 0.3,
                "use_faiss": True,
                "faiss_threshold": 100000,
                "migration_mode": "shadow",
                "auto_save": True,
                "save_interval": 1000
            }
        }


class IndexFeatureFlags(BaseModel):
    """機能フラグ設定"""
    
    use_integrated_index: bool = Field(
        default=False,
        description="統合インデックスを使用（段階的ロールアウト用）"
    )
    
    rollout_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        description="ロールアウト割合（%）"
    )
    
    enable_performance_monitoring: bool = Field(
        default=True,
        description="性能モニタリングを有効化"
    )
    
    enable_rollback: bool = Field(
        default=True,
        description="ロールバック機能を有効化"
    )