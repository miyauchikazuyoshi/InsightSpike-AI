# マルチユーザー対応設計

## 概要

InsightSpike-AIを複数のユーザーが同時に利用できるようにするための設計案。現在のシングルユーザー前提の実装から、マルチユーザー・マルチテナント対応への拡張方針を示す。

## 現状の課題

### 1. データの分離がない
- すべてのエピソードが同一のnamespace（"default"）に保存される
- ユーザー間でデータが混在してしまう
- プライバシーとセキュリティの懸念

### 2. 設定の共有
- グローバルな設定ファイルを使用
- ユーザー固有の設定（LLMプロバイダー、APIキーなど）が管理できない

### 3. 同時アクセスの考慮不足
- SQLiteのWALモードは有効だが、大量の同時書き込みには限界
- ベクトルインデックスの更新で競合の可能性

## マルチユーザー対応アーキテクチャ

### 1. ユーザー管理層

```python
# src/insightspike/core/user.py
class User:
    """ユーザーエンティティ"""
    id: str
    email: str
    created_at: datetime
    settings: UserSettings
    
class UserSettings:
    """ユーザー固有の設定"""
    llm_provider: str
    llm_api_key: Optional[str]  # 暗号化して保存
    max_episodes: int = 10000
    default_language: str = "ja"
```

### 2. データ分離戦略

#### 2.1 Namespace方式（推奨）
```python
# 各ユーザーごとにnamespaceを分離
user_namespace = f"user_{user_id}"
datastore.save_episodes(episodes, namespace=user_namespace)
```

**利点：**
- 実装が簡単
- 既存のコードベースへの影響が最小
- SQLiteでも十分対応可能

**欠点：**
- 完全な分離ではない（同一DB内）
- ユーザー数が増えるとインデックスが肥大化

#### 2.2 Database分離方式
```python
# 各ユーザーごとに別のデータベース
user_db_path = f"./data/users/{user_id}/insightspike.db"
user_datastore = SQLiteDataStore(user_db_path)
```

**利点：**
- 完全なデータ分離
- ユーザーごとのバックアップ・削除が容易
- パフォーマンスの独立性

**欠点：**
- リソース使用量が増加
- 管理の複雑化

### 3. 認証・認可システム

```python
# src/insightspike/auth/authenticator.py
class Authenticator:
    """認証管理"""
    
    async def authenticate(self, token: str) -> Optional[User]:
        """トークンからユーザーを認証"""
        # JWTトークンの検証など
        pass
    
    async def authorize(self, user: User, resource: str, action: str) -> bool:
        """リソースへのアクセス権限を確認"""
        pass
```

### 4. APIレイヤーの追加

```python
# src/insightspike/api/server.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer

app = FastAPI()
security = HTTPBearer()

@app.post("/api/v1/process")
async def process_text(
    text: str,
    user: User = Depends(get_current_user),
    token: str = Depends(security)
):
    """ユーザー固有のコンテキストで処理"""
    # ユーザー専用のDataStoreを取得
    user_datastore = get_user_datastore(user.id)
    
    # ユーザー設定でエージェントを初期化
    agent = DataStoreMainAgent(
        datastore=user_datastore,
        config=user.get_config()
    )
    
    # 処理実行
    result = agent.process(text)
    return result
```

### 5. セッション管理

```python
# src/insightspike/core/session.py
class UserSession:
    """ユーザーセッション管理"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.datastore = self._get_user_datastore()
        self.agent = None
        self._cache = {}  # セッション内キャッシュ
    
    def get_agent(self) -> DataStoreMainAgent:
        """セッション用のエージェントを取得（キャッシュ済み）"""
        if not self.agent:
            self.agent = DataStoreMainAgent(
                datastore=self.datastore,
                config=self._get_user_config()
            )
        return self.agent
```

## 実装フェーズ

### Phase 1: 基本的なユーザー分離（1週間）
1. Userモデルの実装
2. Namespace方式でのデータ分離
3. 簡単な認証システム（APIキー方式）

### Phase 2: API化（1週間）
1. FastAPIサーバーの実装
2. REST APIエンドポイントの定義
3. 認証ミドルウェアの実装

### Phase 3: 高度な機能（2週間）
1. ユーザー間でのデータ共有機能
2. 組織/チーム機能
3. 使用量制限とクォータ管理
4. 監査ログ

### Phase 4: スケーラビリティ対応（オプション）
1. PostgreSQL/MySQLへの移行
2. Redis/Memcachedでのキャッシング
3. 水平スケーリング対応

## セキュリティ考慮事項

### 1. データアクセス制御
- ユーザーは自分のデータのみアクセス可能
- 管理者権限の実装
- データ共有時の権限管理

### 2. APIセキュリティ
- HTTPS必須
- レート制限
- APIキーの定期的なローテーション
- CORS設定

### 3. データ保護
- APIキーの暗号化保存
- 機密データのマスキング
- 定期的なセキュリティ監査

## パフォーマンス最適化

### 1. キャッシング戦略
```python
# ユーザーごとのキャッシュ
user_cache = {
    "embeddings": LRUCache(max_size=1000),
    "search_results": TTLCache(ttl=300),  # 5分
}
```

### 2. リソース制限
```python
# ユーザーごとの制限
USER_LIMITS = {
    "max_episodes": 100000,
    "max_requests_per_minute": 60,
    "max_concurrent_requests": 5,
}
```

### 3. 非同期処理の活用
- すべてのI/O操作を非同期化
- バックグラウンドタスクでの重い処理
- WebSocketsでのリアルタイム通信

## 移行計画

### 既存ユーザーの移行
1. 現在のデータを"default"ユーザーとして扱う
2. 新規ユーザーから新方式を適用
3. 段階的に既存データを移行

### 後方互換性
- CLIは引き続きシングルユーザーモードで動作
- APIモードでマルチユーザー対応
- 設定で切り替え可能

## まとめ

マルチユーザー対応により、InsightSpike-AIは：

1. **エンタープライズ対応**: 組織での利用が可能に
2. **SaaS化**: クラウドサービスとしての提供が可能
3. **コラボレーション**: ユーザー間での知識共有
4. **スケーラビリティ**: 大規模利用への対応

実装は段階的に進め、まずは基本的なユーザー分離から始めることを推奨します。