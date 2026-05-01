# Presence — Frontend (Neural Network Visualization)

영화 *Her* 의 사만다 톤 — 화면 가득 펼쳐진 뉴런 클러스터가 AI의 내면을 표현한다.  
말을 들을 때·생각할 때 뉴런 사이를 달리는 섬광(spark)으로 살아있는 두뇌처럼 느껴진다.  
단일 페이지, 빌드 없음, 순수 ESM.

---

## 시각 디자인 개요

- **뉴런 ~420개**: 유기적 비대칭 blob 안에 분산. 각자 고유한 저주파 깜빡임.  
- **시냅스 엣지**: 뉴런 k=4 최근접 이웃을 연결하는 희미한 선 골격 (alpha ~0.05). 배경에 구조감을 줄 뿐 바쁘지 않다.  
- **스파크(Spark)**: 엣지를 따라 이동하는 밝은 구슬. 목적지 도착 시 뉴런 플래시 + 연쇄 반응(확률 35%, 최대 2단계). 이게 "wow factor".  
- **UnrealBloomPass**: 뉴런과 스파크의 빛을 증폭해 생물발광 느낌.  
- **클러스터 회전**: 30초 주기 느린 Y축 회전 + 미세한 X 진동 — ambient, not busy.

---

## 상태 ↔ 시각 대응

| 상태 | 스파크 속도 | 방향 / 패턴 | 색조 | 분위기 |
|---|---|---|---|---|
| `idle` | 0.8 /s | 랜덤 | 차분한 청남색 | 고요, 명상적 |
| `listening` | 3.0 /s | **주변 → 중심 수렴** | 쿨 블루 | 받아들이는 느낌 |
| `thinking` | 6.5 /s + 간헐적 버스트 | 클러스터 전체 혼돈, hue 라벤더↔틸 스윕 | 보라-청록 | 내부 활동, 복잡 |
| `speaking` | 4.0 /s, 리듬감 | **중심 → 바깥 방사** + `response_chunk`마다 추가 버스트 | 따뜻한 피치-앰버 | 발화하는 느낌 |
| `wake` | 10 /s 순간 + 방사 10개 | 중심 대형 플래시 → 파동 방사, ~600ms | 밝은 황금 | 깨어남 |
| `quiet` | 0.2 /s | 랜덤, 매우 희미 | 탈채도 회청 | 침묵, 억제 |
| `sleep` | 0.05 /s | 거의 정지 | 거의 무채색 | 어둠 속 희미한 깜빡임 |

`memory_recall` 이벤트: **금색 단일 스파크** 가 두 랜덤 뉴런 사이를 가로지름 — "방금 뭔가 기억했다"는 시각 단서.

상태 전환은 ~600ms lerp — 뚝뚝 튀지 않고 녹아든다.

---

## 자막 (Subtitle) 동작

- `transcript` — 사용자 발화 전사.
  - `final: true` 일 때만 자막에 표시 (`› ` 접두사, 약간 어둡게).
  - `final: false` (interim) 는 실시간으로 반영되지만 타이머를 재시작하지 않는다.
- `response_chunk` — 에이전트 응답을 청크 단위로 이어붙여 표시. 동시에 outward spark 버스트.
  - 마지막 청크로부터 1.5 s 이상 간격이 벌어지면 새 줄로 시작.
- `response_end` — 자막을 지우지 않고 5 s 후 자연스럽게 페이드.

---

## 수동 테스트

1. `make presence` 실행 — `localhost:8765` 에 서버가 뜬다.
2. 브라우저로 `http://127.0.0.1:8765/` 열기.
3. 같은 머신의 다른 터미널에서 텍스트 채널로 입력 → 뉴런 클러스터가 `listening → thinking → speaking` 으로 전환되는 걸 관찰.

접속 직후 우상단 점이 녹색으로 바뀌면 WS 연결 성공.  
서버를 끄면 2 초 후 자동 재연결을 시도한다.

상태 이벤트를 직접 쏘려면:
```bash
curl -s -X POST http://127.0.0.1:8765/publish \
  -H "Content-Type: application/json" \
  -d '{"type":"state","payload":{"value":"thinking","channel":"voice"},"ts":1}'
```

---

## 파일 구조

```
apps/presence/web/
├── index.html   HTML 뼈대 (three + three/addons/ importmap 포함)
├── styles.css   CSS custom properties 팔레트 + 레이아웃
├── network.js   NeuralNetwork 클래스 — 뉴런/엣지/스파크 + UnrealBloomPass
├── orb.js       기존 오브 셰이더 (보관용 — main.js 에서는 더 이상 import 안 함)
└── main.js      WS 클라이언트 + 자막 컨트롤러 + 상태 머신 → NeuralNetwork 연결
```

---

## 의존성 (CDN)

| 패키지 | 버전 | 용도 |
|---|---|---|
| `three` | 0.169.0 | WebGL 장면 렌더링 |
| `three/addons/` | 0.169.0 | EffectComposer + UnrealBloomPass |
| Pretendard | 1.3.9 | 한국어 자막 폰트 |

빌드 도구 없음. 인터넷 없는 환경에서는 Pretendard 대신 시스템 sans-serif 로 폴백한다.

---

## 브라우저 요구 사항

Chrome / Firefox / Safari 최신 버전 (WebGL 2.0 + ES modules + importmap 지원).  
importmap 은 Chromium 89+ / Firefox 108+ / Safari 16.4+ 에서 지원된다.  
Apple Silicon (M1/M2) 60fps — ~500 뉴런 + ~2000 엣지 + 최대 80개 활성 스파크.

---

## 알려진 제한

- WS 이벤트 없이 정적으로 열면 클러스터는 `idle` 상태로 스파크만 드문드문 튄다 (정상).
- `make presence` 없이 `file://` 로 열면 CORS 로 Three.js CDN 모듈이 막힐 수 있다 — 반드시 로컬 서버 경유.
