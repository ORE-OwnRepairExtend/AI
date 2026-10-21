# ORE AI Server

> “우리는 소비를 줄이자고 말하지 않는다. 대신, 제품을 더 오래, 더 제대로 사용할 수 있도록 돕는다.”

## 프로젝트 소개

**오래(ORE, Own Repair Extend)** AI 서버는 전자·가전제품을 오래, 더 제대로 사용할 수 있도록 제품 정보를 해석하고 필요한 관리 정보를 제공하는 서비스입니다.

사용자가 등록한 영수증, 문자, 제품 스티커, 매뉴얼 이미지를 분석해 브랜드·제품명·모델명을 추출합니다. 추출한 정보로 제조사 공식 매뉴얼을 찾고, 매뉴얼 본문을 바탕으로 사용자가 빠르게 이해할 수 있는 핵심 정보를 제공합니다. 또한 제품 정보, 매뉴얼, 이전 대화 맥락을 활용해 제품 관리와 서비스 이용을 돕는 AI 채팅 답변을 생성합니다.

이 서버는 ORE Spring Boot 백엔드와 연동되는 독립 FastAPI 서비스입니다. 제품의 구매 이후부터 관리, 수리, 재사용까지 이어지는 흐름에서 필요한 정보를 다시 찾는 시간을 줄이고, 사용자가 자신의 제품을 더 오래 사용할 수 있는 환경을 만드는 것을 목표로 합니다.

## 주요 기능

- OpenAI Vision 기반 이미지 OCR 및 제품 정보 추출
- 영수증, 문자, 제품 스티커, 매뉴얼 이미지 출처별 제품 정보 인식
- 브랜드·제품명·모델명을 기반으로 한 공식 매뉴얼 검색
- 공식 웹페이지 또는 PDF 매뉴얼의 텍스트 추출 및 핵심 내용 요약
- 제품 정보, 매뉴얼 내용, 이전 대화 이력을 반영한 AI 채팅
- 제품을 선택하지 않은 사용자를 위한 ORE 서비스 이용 안내 채팅
- 위험한 분해·임의 수리·전기 작업에는 공식 서비스센터 문의를 안내하는 안전 응답 규칙
- 헬스 체크 및 FastAPI 자동 API 문서 제공

## 기술 스택

- Python 3.11
- FastAPI / Uvicorn
- OpenAI Python SDK
- Pydantic
- Beautiful Soup 4, Requests
- PyMuPDF
- python-dotenv
- GitHub Actions / Amazon EC2

## 프로젝트 구조

```text
.
├── main.py             # FastAPI 앱, API 엔드포인트, OCR·매뉴얼·채팅 로직
├── requirements.txt    # Python 의존성
├── .github/
│   └── workflows/
│       └── deploy.yml  # EC2 배포 워크플로
└── README.md
```

## 로컬 실행 방법

### 1. 사전 준비

- Python 3.11 이상
- OpenAI API 키

### 2. 가상환경 및 의존성 설정

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. 환경 변수 설정

루트에 `.env` 파일을 만들고 OpenAI 설정을 추가합니다. `.env`는 Git에서 제외되므로 API 키를 커밋하지 마세요.

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

`OPENAI_MODEL`을 설정하지 않으면 기본값으로 `gpt-4.1-mini`를 사용합니다.

### 4. 애플리케이션 실행

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버는 기본적으로 `http://localhost:8000`에서 실행됩니다. ORE Backend의 기본 AI 서버 주소도 `http://localhost:8000`입니다.

## API 문서

로컬 실행 후 FastAPI가 제공하는 API 문서를 확인할 수 있습니다.

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API 개요

| Method | Endpoint | 설명 |
| --- | --- | --- |
| `GET` | `/health` | AI 서버 상태 확인 |
| `POST` | `/ai/ocr/model` | 이미지 URL과 출처 타입을 받아 OCR 및 제품 정보 추출 |
| `POST` | `/ai/product-manuals/process` | 제품 정보로 공식 매뉴얼을 검색하고 핵심 내용 요약 |
| `POST` | `/ai/chat` | 제품·매뉴얼·대화 이력을 바탕으로 AI 답변 생성 |

### 이미지 OCR

`POST /ai/ocr/model`은 `imageUrl`과 `sourceType`을 받습니다. `sourceType`은 `receipt`, `sms`, `sticker`, `manual` 중 하나를 사용합니다.

```json
{
  "imageUrl": "https://example.com/product-sticker.jpg",
  "sourceType": "sticker"
}
```

응답에는 OCR 원문과 추출된 `brandName`, `productName`, `modelNumber`가 포함됩니다.

### 매뉴얼 검색 및 요약

`POST /ai/product-manuals/process`는 모델명을 필수로 받으며, 브랜드명과 제품명은 선택 사항입니다. 공식 매뉴얼 URL, 핵심 요약, 고객센터 정보를 반환합니다.

```json
{
  "brandName": "Samsung",
  "productName": "Example Product",
  "modelNumber": "ABC-1234"
}
```

### AI 채팅

`POST /ai/chat`은 질문과 세션 ID를 필수로 받습니다. 제품 정보, 저장된 매뉴얼 내용, 매뉴얼 URL, 이전 대화 이력을 함께 전달하면 해당 제품에 맞춘 답변을 생성합니다. 이전 대화는 최근 10개 메시지만 반영합니다.

```json
{
  "sessionId": "session-id",
  "message": "필터는 언제 청소해야 하나요?",
  "product": {
    "productId": "product-id",
    "name": "Example Product",
    "nickname": "우리 집 제품",
    "category": "AIR_PURIFIER"
  },
  "manualContent": "매뉴얼 본문 또는 요약 내용",
  "manualUrl": "https://example.com/manual",
  "history": []
}
```

## 배포

GitHub Actions를 통해 Amazon EC2에 배포합니다. [deploy.yml](.github/workflows/deploy.yml)은 `main` 브랜치에 푸시되면 소스 코드를 EC2로 전달하고, 의존성 설치 및 `ore-ai` 시스템 서비스를 재시작합니다.

배포를 위해 다음 GitHub Actions Secrets가 필요합니다.

- `EC2_HOST`, `EC2_USERNAME`, `EC2_SSH_KEY`
- `OPENAI_API_KEY`, `OPENAI_MODEL`
