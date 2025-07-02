# gnn‑trading 📈
한국 주식·국채 선물에 **그래프 신경망(GNN)** 을 적용해 하루 ~ 일주일 수익률을 예측하고, 백테스트·실시간 API까지 한 번에 돌릴 수 있는 파이프라인입니다.  
👉 *데이터 인제스트 → 피처 엔지니어링 → 그래프 구축 → TGAT 학습 → 백테스트 / API* 단계가 모두 포함된 **올‑인‑원 패키지**.

---

## 1. 설치
```bash
# 1) 압축 해제 후 프로젝트 폴더로 이동
cd gnn_trading
# 2) 의존성 설치 (Poetry 필요)
poetry install
```
> **Python 3.11** 기반 / GPU 옵션(선택)까지 torch >= 2.3 설치를 가정합니다.

---

## 2. 빠른 파이프라인 실행
```bash
# ① 데이터 수집 (시장 6월, 거시 2020~현재 예)
poetry run python -m gnn_trading.data_pipeline.ingest     --start 2025-06-01 --end 2025-06-30

# ② 피처 → 그래프 → 모델 학습
poetry run python -m gnn_trading.data_pipeline.feature_builder     --mkt_table market_intraday_202506 --macro_table macro_indicators
poetry run python -m gnn_trading.graphs.graph_builder     --feature_root feature_store/processed
poetry run python -m gnn_trading.models.trainer     --snapshot_dir graph_snapshots --epochs 10

# ③ 백테스트
poetry run python -m gnn_trading.backtest.engine     --config configs/backtest_config.yaml

# ④ 실시간 예측 API (선택)
poetry run uvicorn gnn_trading.api.main:app --port 8000
```
> 🔑 **API Key** 및 자산·지표 목록은 `configs/source_config.yaml`에 입력해야 합니다.

---

## 3. 폴더 구조
```
├─ src/gnn_trading/
│  ├─ data_pipeline/   # Phase 0·1  인제스트 & 피처
│  ├─ graphs/          # Phase 2    그래프 스냅샷
│  ├─ models/          # Phase 3    TGAT 모델 & 학습
│  ├─ backtest/        # Phase 4‑A  백테스트 엔진
│  └─ api/             # Phase 4‑B  FastAPI 예측 서버
├─ feature_store/      # Raw·Processed Parquet 저장소 (런타임 생성)
├─ graph_snapshots/    # 시점별 .pt 파일 (Phase 2 출력)
├─ checkpoints/        # 학습된 모델 (.ckpt)
├─ configs/            # YAML 설정 (API, 피처, 그래프, 학습, 백테스트)
└─ docs/               # 단계별 상세 설명
```

---

## 4. 주요 설정 파일
| 파일 | 설명 |
| ---- | ---- |
| `configs/source_config.yaml`   | API 엔드포인트·자산·거시 지표 목록 |
| `configs/feature_config.yaml`  | 리샘플 주기·FFill/MIDAS 방식 |
| `configs/graph_config.yaml`    | 스냅샷 주기·Correlation/Granger Edge 기준 |
| `configs/train_config.yaml`    | 에폭·배치·시퀀스 길이 등 TGAT 학습 하이퍼파라미터 |
| `configs/backtest_config.yaml` | 수수료·슬리피지·심볼 등 백테스트 파라미터 |

---

## 5. 결과물
* **백테스트** `backtest_result.csv`  → `cum_pnl`, `pos` 열 포함
* **리스크 리포트** CLI 출력 : Sharpe, Max Drawdown
* **API** `POST /predict` → { "predictions": [...node values] }

---

## 6. 커스텀/확장 Tips
1. 🔄 **거시 MIDAS 임베딩** : `feature_builder.fill_macro_daily()`에 `method="midas"` 로 확장
2. 🪄 **Edge 정의 변경** : `graph_builder.py` 내 `_edges_from_corr` / `_edges_from_granger` 수정
3. ⚙️ **수익률‑기반 Loss** : `models/tgat.py` 또는 `trainer.py` 에서 커스텀 손실 적용

---

## 7. 문의
*Issue / PR 환영*  
(예시 저장소 URL 자리)
