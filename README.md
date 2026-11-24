# 스마트팩토리 YOLOv8 + 온톨로지 웹서비스

이전 Streamlit 사이트의 프레임별 Temporary 그래프, 누적 그래프, LLM Prompt를 유지하면서 Roboflow 학습 모델 `models/best.pt`를 연결한 버전입니다.

## Windows VS Code 실행

1. 기존 실행 중인 터미널에서 `Ctrl+C`로 중지하세요.
2. 새 ZIP을 별도 폴더에 압축 해제하고 `factory_kg_web` 폴더를 VS Code에서 여세요.
3. 터미널에서 아래 명령을 **한 줄씩** 실행하세요.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. `http://localhost:8501`을 열고 영상을 업로드하세요.

## 바뀐 내용
- `models/best.pt`: 사용자 학습 모델
- `factory_core.py`: YOLO 탐지 및 라벨-온톨로지 매핑, SIFT 의존 없는 초기화
- `app.py`: YOLO 모델 로딩, 신뢰도 슬라이더, 그래프 연결
- `requirements.txt`: ultralytics 추가

## 한계와 주의
- 학습 모델의 실제 정확도는 검증 영상과 mAP/precision/recall로 확인해야 합니다.
- 2초 간격 추적은 간단한 위치 기반 추적이며, **ByteTrack 자체가 아닙니다**. 프레임 간 동일 ID가 항상 보장되지 않습니다.
- 공정 타임라인은 기존 펜팩토리 영상에 고정되어 있어 영상에 맞게 수정해야 합니다.
- `controls`, `handles` 같은 의미 관계는 실제 동작을 관찰한 확정 사실이 아니라 규칙 기반 후보입니다.
- LLM 응답은 기존 사이트처럼 Ollama가 실행 중이어야 가능합니다.
- 로컬 시연용이며 외부 공개 서비스가 아닙니다.
