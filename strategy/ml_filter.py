"""
ML 시그널 필터.
과거 거래 데이터를 학습하여 시그널 품질을 예측한다.
최소 50건의 거래 데이터가 누적된 후 활성화된다.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import json

import numpy as np
from loguru import logger


@dataclass
class MLPrediction:
    """ML 예측 결과"""
    win_probability: float    # 승리 확률 (0~1)
    should_trade: bool        # 거래 실행 여부
    features_used: int        # 사용된 피처 수
    model_type: str           # 모델 타입 ("logistic", "statistical", "disabled")

    def summary(self) -> str:
        status = "PASS" if self.should_trade else "BLOCK"
        return f"ML필터 [{status}] 승률예측={self.win_probability:.1%} 모델={self.model_type}"


class MLSignalFilter:
    """ML 기반 시그널 필터링"""

    MIN_TRADES_FOR_ML = 50          # ML 활성화 최소 거래 수
    MIN_WIN_PROBABILITY = 0.55      # 최소 예측 승률
    RETRAIN_INTERVAL_HOURS = 168    # 재학습 주기 (7일)

    def __init__(self):
        self._model = None
        self._scaler = None
        self._is_trained = False
        self._last_train_time: Optional[datetime] = None
        self._feature_importances: Optional[dict] = None
        self._model_type = "disabled"

    def predict(
        self,
        score_total: float,
        trend_score: float = 50.0,
        momentum_score: float = 50.0,
        volume_score: float = 50.0,
        volatility_score: float = 50.0,
        sentiment_score: float = 50.0,
        macro_score: float = 50.0,
        atr_pct: float = 0.01,
        hour_of_day: int = 12,
        day_of_week: int = 3,
        regime: str = "TRENDING",
    ) -> MLPrediction:
        """시그널 피처를 입력받아 승률을 예측한다.

        Returns:
            MLPrediction (model이 학습되지 않았으면 should_trade=True)
        """
        if not self._is_trained:
            return MLPrediction(
                win_probability=0.5,
                should_trade=True,
                features_used=0,
                model_type="disabled",
            )

        features = self._build_features(
            score_total, trend_score, momentum_score, volume_score,
            volatility_score, sentiment_score, macro_score,
            atr_pct, hour_of_day, day_of_week, regime,
        )

        try:
            if self._model_type == "logistic":
                features_scaled = self._scaler.transform([features])
                prob = self._model.predict_proba(features_scaled)[0][1]
            else:
                prob = self._statistical_predict(features)

            should_trade = prob >= self.MIN_WIN_PROBABILITY

            return MLPrediction(
                win_probability=float(prob),
                should_trade=should_trade,
                features_used=len(features),
                model_type=self._model_type,
            )
        except Exception as e:
            logger.warning(f"ML 예측 실패: {e} — 기본 통과 처리")
            return MLPrediction(
                win_probability=0.5,
                should_trade=True,
                features_used=0,
                model_type="error",
            )

    def train(self, trades: list[dict]) -> bool:
        """과거 거래 데이터로 모델을 학습한다.

        Args:
            trades: 거래 기록 리스트. 각 딕셔너리에 다음 키 필요:
                - pnl_usdt: 거래 P&L
                - signal_confidence: 시그널 신뢰도
                - entry_time: 진입 시각 (ISO format)
                - atr_at_entry: 진입 시 ATR

        Returns:
            학습 성공 여부
        """
        if len(trades) < self.MIN_TRADES_FOR_ML:
            logger.info(f"[ML필터] 거래 수 부족 ({len(trades)}/{self.MIN_TRADES_FOR_ML}) — 학습 스킵")
            return False

        # 피처/라벨 구성
        X = []
        y = []
        for t in trades:
            pnl = t.get("pnl_usdt", 0) or 0
            confidence = t.get("signal_confidence", 0.5) or 0.5
            atr = t.get("atr_at_entry", 0) or 0
            entry_time = t.get("entry_time", "")

            # 시간 피처 추출
            hour = 12
            dow = 3
            try:
                if entry_time:
                    dt = datetime.fromisoformat(str(entry_time))
                    hour = dt.hour
                    dow = dt.weekday()
            except (ValueError, TypeError):
                pass

            features = [
                confidence * 100,  # score_total 근사
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  # 개별 스코어 (unavailable → 중립)
                atr / 75000 if atr > 0 else 0.01,  # atr_pct 근사
                hour,
                dow,
                1.0,  # regime = TRENDING (default)
            ]
            X.append(features)
            y.append(1 if pnl > 0 else 0)

        X = np.array(X)
        y = np.array(y)

        # sklearn 사용 시도
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            self._model = LogisticRegression(
                max_iter=1000,
                C=0.1,  # 정규화 강화 (과적합 방지)
                class_weight="balanced",
            )
            self._model.fit(X_scaled, y)
            self._model_type = "logistic"

            # 피처 중요도 추출
            feature_names = [
                "score_total", "trend", "momentum", "volume",
                "volatility", "sentiment", "macro",
                "atr_pct", "hour", "day_of_week", "regime",
            ]
            importances = np.abs(self._model.coef_[0])
            self._feature_importances = dict(zip(feature_names, importances))

            logger.info(f"[ML필터] Logistic Regression 학습 완료 ({len(X)}건)")
            logger.info(f"[ML필터] 피처 중요도: {self._feature_importances}")

        except ImportError:
            # sklearn 미설치 시 통계 기반 대체
            self._model_type = "statistical"
            self._stat_means = {
                "win_mean_score": float(X[y == 1, 0].mean()) if y.sum() > 0 else 50,
                "loss_mean_score": float(X[y == 0, 0].mean()) if (1 - y).sum() > 0 else 50,
                "overall_win_rate": float(y.mean()),
            }
            logger.info(f"[ML필터] 통계 모델 사용 (sklearn 미설치): {self._stat_means}")

        self._is_trained = True
        self._last_train_time = datetime.now(timezone.utc)
        return True

    def needs_retrain(self) -> bool:
        """재학습이 필요한지 확인한다."""
        if not self._is_trained or self._last_train_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_train_time).total_seconds() / 3600
        return elapsed >= self.RETRAIN_INTERVAL_HOURS

    def get_feature_importances(self) -> Optional[dict]:
        """피처 중요도를 반환한다 (학습 후에만 가용)."""
        return self._feature_importances

    def _build_features(
        self,
        score_total: float,
        trend_score: float,
        momentum_score: float,
        volume_score: float,
        volatility_score: float,
        sentiment_score: float,
        macro_score: float,
        atr_pct: float,
        hour_of_day: int,
        day_of_week: int,
        regime: str,
    ) -> list[float]:
        """피처 벡터를 구성한다."""
        regime_map = {"TRENDING": 1.0, "RANGING": 0.0, "VOLATILE": -1.0}
        return [
            score_total,
            trend_score,
            momentum_score,
            volume_score,
            volatility_score,
            sentiment_score,
            macro_score,
            atr_pct,
            float(hour_of_day),
            float(day_of_week),
            regime_map.get(regime, 0.5),
        ]

    def _statistical_predict(self, features: list[float]) -> float:
        """sklearn 없이 통계 기반 예측 (대체 모델)."""
        if not hasattr(self, "_stat_means"):
            return 0.5

        score = features[0]
        base_wr = self._stat_means.get("overall_win_rate", 0.5)
        win_mean = self._stat_means.get("win_mean_score", 70)
        loss_mean = self._stat_means.get("loss_mean_score", 50)

        if win_mean == loss_mean:
            return base_wr

        # 스코어가 승리 평균에 가까울수록 높은 확률
        distance_to_win = abs(score - win_mean)
        distance_to_loss = abs(score - loss_mean)
        total_distance = distance_to_win + distance_to_loss

        if total_distance == 0:
            return base_wr

        prob = distance_to_loss / total_distance
        # base_wr 방향으로 보정
        prob = prob * 0.7 + base_wr * 0.3
        return max(0.0, min(1.0, prob))
