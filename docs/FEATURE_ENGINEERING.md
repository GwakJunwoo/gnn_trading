
# Feature Engineering Layer (Phase‑1)

생성 일시 : 2025-07-02T03:14:18

이 레이어는 **원천 파케이** → **모델 학습용 Feature** 변환을 담당합니다.

## 주요 산출
- `market_<freq>.parquet` : 리샘플 OHLCV + 수익률  
- `macro_daily.parquet`  : 거시지표 일단위 Forward‑fill
