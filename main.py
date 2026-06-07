import json
import os
import re
import tempfile
from typing import Literal, Optional
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import fitz
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(title="ORE AI Server")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class ModelOcrRequest(BaseModel):
    imageUrl: str = Field(..., description="분석할 이미지 URL")
    sourceType: Literal["receipt", "sms", "sticker", "manual"] = Field(
        ...,
        description="이미지 출처 타입"
    )


class ModelOcrResponse(BaseModel):
    ocrText: str
    brandName: str
    productName: str
    modelNumber: str


class ManualProcessRequest(BaseModel):
    brandName: Optional[str] = ""
    productName: Optional[str] = ""
    modelNumber: str


class ManualProcessResponse(BaseModel):
    manualUrl: str
    manualSummary: str
    customerCenter: str


class ChatProductInfo(BaseModel):
    productId: str
    name: Optional[str] = ""
    nickname: Optional[str] = ""
    category: Optional[str] = ""


class ChatHistoryItem(BaseModel):
    role: Literal["USER", "ASSISTANT"]
    content: str


class AiChatRequest(BaseModel):
    sessionId: str
    message: str = Field(..., description="사용자 질문")
    product: Optional[ChatProductInfo] = None
    manualContent: Optional[str] = None
    manualUrl: Optional[str] = None
    history: list[ChatHistoryItem] = []


class AiChatResponse(BaseModel):
    answer: str


# 서버 상태 확인용 health check API
@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "AI server is running"
    }


# 이미지 URL과 출처 타입을 받아 OCR 기반 제품 정보를 추출하는 API
@app.post("/ai/ocr/model", response_model=ModelOcrResponse)
def analyze_model_from_image(request: ModelOcrRequest):
    try:
        result = extract_product_info(
            image_url=request.imageUrl,
            source_type=request.sourceType
        )

        return ModelOcrResponse(
            ocrText=result.get("ocrText", ""),
            brandName=result.get("brandName", ""),
            productName=result.get("productName", ""),
            modelNumber=result.get("modelNumber", "")
        )

    except Exception as e:
        print("[ERROR] 이미지 분석 실패:", e)
        return JSONResponse(
            status_code=400,
            content={"message": "이미지 분석에 실패했습니다."}
        )


# 제품 정보로 공식 매뉴얼을 검색하고 요약 결과를 반환하는 API
@app.post("/ai/product-manuals/process", response_model=ManualProcessResponse)
def process_product_manual(request: ManualProcessRequest):
    # 매뉴얼 검색에는 모델명이 필수이므로 먼저 검증
    if not request.modelNumber or not request.modelNumber.strip():
        return JSONResponse(
            status_code=400,
            content={"message": "모델명이 필요합니다."}
        )

    try:
        # 제품 정보를 기반으로 공식 매뉴얼 검색부터 요약까지 수행
        result = search_and_summarize_manual(
            brand_name=request.brandName or "",
            product_name=request.productName or "",
            model_number=request.modelNumber
        )

        # 응답에 포함할 URL, 요약문, 고객센터 정보를 정리
        manual_url = clean_url(result.get("manualUrl", ""))
        manual_summary = clean_summary_text(result.get("manualSummary", ""))
        customer_center = clean_text(result.get("customerCenter", ""))

        # 필수 정보가 없거나 요약 섹션이 부족하면 실패 응답 반환
        if (
            not manual_url
            or not has_required_summary_sections(manual_summary)
        ):
            return JSONResponse(
                status_code=404,
                content={"message": "매뉴얼 정보를 찾을 수 없습니다."}
            )

        # 정상적으로 찾은 매뉴얼 정보를 반환
        return ManualProcessResponse(
            manualUrl=manual_url,
            manualSummary=manual_summary,
            customerCenter=customer_center
        )

    except Exception as e:
        print("[ERROR] 매뉴얼 검색 실패:", e)
        return JSONResponse(
            status_code=500,
            content={"message": "매뉴얼 검색 중 오류가 발생했습니다."}
        )


# 사용자 질문과 제품/매뉴얼 문맥을 기반으로 AI 답변을 생성하는 API
@app.post("/ai/chat", response_model=AiChatResponse)
def generate_ai_chat_answer(request: AiChatRequest):
    # 질문 내용 검증
    if not request.message or not request.message.strip():
        return JSONResponse(
            status_code=400,
            content={"message": "질문을 이해할 수 없습니다."}
        )

    try:
        answer = generate_chat_answer(request)

        return AiChatResponse(
            answer=answer
        )

    except Exception as e:
        print("[ERROR] AI 채팅 응답 생성 실패:", e)
        return JSONResponse(
            status_code=500,
            content={"message": "AI 응답 생성 중 오류가 발생했습니다."}
        )


# OpenAI Vision을 사용해 이미지에서 OCR 텍스트와 제품 정보 추출
def extract_product_info(image_url: str, source_type: str) -> dict:
    # 이미지 유형에 맞춰 OCR 및 제품 정보 추출 프롬프트 구성
    prompt = f"""
        너는 제품 영수증, 결제 문자, 제품 스티커, 설명서 이미지에서 텍스트를 읽고 제품 정보를 추출하는 AI야.

        이미지 유형: {source_type}

        작업:
        1. 이미지에 보이는 텍스트를 최대한 정확하게 추출해.
        2. 제품 브랜드명을 추출해.
        3. 제품명을 추출해.
        4. 제품 모델명 또는 모델 코드를 추출해.
        5. 모델명을 찾지 못하면 modelNumber는 빈 문자열("")로 반환해.
        6. 브랜드명이나 제품명을 확실히 알 수 없으면 빈 문자열("")로 반환해.
        7. 날짜, 가격, 전화번호, 사업자번호, 주문번호, 카드번호, 승인번호는 모델명으로 선택하지 마.
        8. 모델명은 보통 영문 대문자, 숫자, 하이픈, 슬래시 조합일 가능성이 높아.
        9. 영수증의 경우 상품명 영역에 있는 값에서 제품명과 모델명을 우선적으로 추출해.
        10. 브랜드명이 영수증에 직접 없더라도, 제품명/모델명으로 강하게 추론 가능한 경우에만 브랜드명을 채워.
        11. 확실하지 않은 값은 억지로 만들지 말고 빈 문자열("")로 반환해.

        반드시 아래 JSON 형식으로만 답해.
        마크다운, 코드블록, 설명 문장은 절대 넣지 마.

        {{
            "ocrText": "이미지에서 추출한 전체 텍스트",
            "brandName": "브랜드명",
            "productName": "제품명",
            "modelNumber": "제품 모델명 또는 모델 코드"
        }}
    """

    # OpenAI Vision 모델에 이미지 URL과 프롬프트를 함께 전달
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url
                    }
                ]
            }
        ]
    )

    output_text = response.output_text.strip()

    try:
        return json.loads(output_text)

    except json.JSONDecodeError:
        print("[ERROR] OCR JSON 파싱 실패. 원본 응답:", output_text)
        raise ValueError("AI 응답을 JSON으로 파싱할 수 없습니다.")


# 일반 텍스트에서 불필요한 링크/공백을 정리
def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\([^)]*utm_source[^)]*\)", "", text)
    text = re.sub(r"\([^)]*openai[^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# URL에서 쿼리 파라미터와 불필요한 값을 제거하고 유효 길이를 제한
def clean_url(url: str) -> str:
    if not url:
        return ""

    parsed = urlparse(url)

    # 추적/세션성 파라미터만 제거하고, manualId 같은 실제 식별자는 유지
    blocked_query_keys = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "msockid",
        "fbclid",
        "gclid"
    }

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in blocked_query_keys
    ]

    cleaned = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        "",
        urlencode(filtered_query),
        ""
    )).strip()

    # 추적 파라미터 제거 후에도 너무 길면 비정상 URL로 처리
    if len(cleaned) > 500:
        return ""

    return cleaned


# 매뉴얼 요약문은 섹션별 줄바꿈을 유지하면서 텍스트를 정리
def clean_summary_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\([^)]*utm_source[^)]*\)", "", text)
    text = re.sub(r"\([^)]*openai[^)]*\)", "", text)

    # 운영체제별 줄바꿈 차이를 통일
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 세 섹션이 한 줄로 붙어 내려와도 라벨 기준으로 다시 문단 분리
    text = re.sub(r"\s*(사용 방법:)", r"\n\1", text)
    text = re.sub(r"\s*(관리 방법:)", r"\n\n\1", text)
    text = re.sub(r"\s*(고장/수리 안내:)", r"\n\n\1", text)

    # 줄바꿈은 유지하고 각 줄 안의 불필요한 공백만 정리
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    return text.strip()


# 요약문에 필수 섹션 3개가 모두 포함되어 있는지 확인
def has_required_summary_sections(summary: str) -> bool:
    if not summary:
        return False

    required_sections = [
        "사용 방법:",
        "관리 방법:",
        "고장/수리 안내:"
    ]

    return all(section in summary for section in required_sections)


# 공식 매뉴얼 URL로 사용할 수 있는지 도메인, 길이, 비정상 패턴 검증
def is_valid_manual_url(url: str) -> bool:
    # URL이 없거나 너무 길면 정상적인 매뉴얼 주소로 보지 않음
    if not url:
        return False

    if len(url) > 500:
        return False

    lowered = url.lower()

    # 세션/추적 파라미터 또는 비정상적으로 긴 manualId URL 차단
    invalid_patterns = [
        "manualid=kor-000000",
        "manualid=k000000",
        "000000000000000000000000",
        "msockid=",
        "utm_source=",
        "utm_medium=",
        "utm_campaign="
    ]

    if any(pattern in lowered for pattern in invalid_patterns):
        return False

    if re.search(r"manualid=[a-z0-9]{100,}", lowered):
        return False

    # 현재 서비스에서 허용하는 제조사 공식 도메인 목록
    official_domains = [
        "lge.co.kr",
        "lg.com",
        "samsung.com",
        "apple.com"
    ]

    # 공식 매뉴얼/지원 페이지로 판단할 수 있는 키워드 목록
    manual_keywords = [
        "support",
        "manual",
        "download",
        "service",
        "manuals",
        "user-guide",
        "userguide",
        "사용설명서",
        "매뉴얼",
        "고객지원",
        "다운로드"
    ]

    # 공식 도메인이거나 매뉴얼 관련 키워드가 포함된 경우만 허용
    return (
        any(domain in lowered for domain in official_domains)
        or any(keyword in lowered for keyword in manual_keywords)
    )


# OpenAI 검색 응답에서 매뉴얼 URL과 고객센터 정보를 추출하고 검증
def parse_manual_search_response(output_text: str, error_label: str) -> dict:
    try:
        # OpenAI 응답을 JSON으로 파싱
        result = json.loads(output_text)

        # URL과 고객센터 정보를 후처리
        manual_url = clean_url(result.get("manualUrl", ""))
        customer_center = clean_text(result.get("customerCenter", ""))

        # 비정상 URL이면 URL만 비워서 검색 실패로 처리
        if not is_valid_manual_url(manual_url):
            return {
                "manualUrl": "",
                "customerCenter": customer_center
            }

        return {
            "manualUrl": manual_url,
            "customerCenter": customer_center
        }

    except json.JSONDecodeError:
        print(f"[ERROR] {error_label} JSON 파싱 실패. 원본 응답:", output_text)
        return {
            "manualUrl": "",
            "customerCenter": ""
        }


# OpenAI 웹 검색을 통해 제조사 공식 매뉴얼 또는 지원 페이지 검색
def find_official_manual_page(
    brand_name: str,
    product_name: str,
    model_number: str
) -> dict:
    # 제품 정보로 제조사 공식 매뉴얼/지원 페이지를 찾도록 프롬프트 구성
    prompt = f"""
        너는 제조사 공식 매뉴얼 또는 공식 제품 지원 페이지를 찾는 검색 보조 AI야.

        입력 제품 정보:
        - 브랜드명: {brand_name}
        - 제품명: {product_name}
        - 모델명: {model_number}

        작업:
        1. 제조사 공식 사용설명서 PDF, 공식 매뉴얼 다운로드 페이지, 공식 제품 지원 페이지를 찾아.
        2. 가장 우선순위는 공식 PDF 사용설명서 URL이야.
        3. PDF를 찾지 못하면 공식 support, download, manual, service 페이지를 선택해.
        4. 그것도 찾지 못하면 제조사 공식 제품 페이지 또는 공식 제품 지원 페이지를 선택해.
        5. 블로그, 쇼핑몰, 커뮤니티, 중고거래, 광고성 페이지는 제외해.
        6. 모델명이 정확히 일치하거나 매우 강하게 관련된 공식 URL만 선택해.
        7. 제조사 공식 사이트의 관련 페이지가 있으면 반드시 manualUrl에 반환해.
        8. 정말 아무 공식 페이지도 찾을 수 없을 때만 manualUrl을 빈 문자열("")로 반환해.
        9. URL에 utm_source, msockid 등 추적 파라미터가 있으면 제거해.
        10. manualUrl은 500자 이하의 정상적인 URL만 반환해.
        11. manualId, sessionId, tracking parameter가 비정상적으로 길게 붙은 URL은 반환하지 마.
        12. customerCenter는 공식 고객센터 정보를 작성해. 확실하지 않으면 빈 문자열("")로 반환해.

        반드시 아래 JSON 형식으로만 답해.
        마크다운, 코드블록, 설명 문장, 출처 표기, citation은 절대 넣지 마.

        {{
        "manualUrl": "공식 매뉴얼 PDF, 공식 매뉴얼 다운로드 페이지, 공식 제품 지원 페이지, 또는 제조사 공식 제품 페이지 URL",
        "customerCenter": "고객센터 정보"
        }}
    """

    # OpenAI 웹 검색 도구를 사용해 공식 URL 후보 검색
    response = client.responses.create(
        model=OPENAI_MODEL,
        tools=[
            {
                "type": "web_search_preview"
            }
        ],
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output_text = response.output_text.strip()

    return parse_manual_search_response(
        output_text=output_text,
        error_label="공식 매뉴얼 페이지"
    )


# 1차 매뉴얼 URL이 실패했을 때 대체 공식 페이지 다시 검색
def find_alternative_official_page(
    brand_name: str,
    product_name: str,
    model_number: str,
    failed_url: str
) -> dict:
    prompt = f"""
        너는 제조사 공식 매뉴얼 또는 공식 지원 페이지를 다시 찾는 검색 보조 AI야.

        입력 제품 정보:
        - 브랜드명: {brand_name}
        - 제품명: {product_name}
        - 모델명: {model_number}

        이전 후보 URL은 접근 실패 또는 404였어:
        - 실패 URL: {failed_url}

        작업:
        1. 실패 URL은 절대 다시 반환하지 마.
        2. 제조사 공식 사용설명서 PDF, 공식 매뉴얼 다운로드 페이지, 공식 제품 지원 페이지를 다시 찾아.
        3. 공식 PDF가 없으면 공식 support, manual, guide, service 페이지를 선택해.
        4. Apple 제품의 경우 support.apple.com의 사용 설명서, user guide, manual 페이지를 우선 찾아.
        5. 블로그, 쇼핑몰, 커뮤니티, 광고성 페이지는 제외해.
        6. 모델명이 정확히 일치하지 않아도 제품명과 강하게 관련된 공식 사용 안내 페이지면 선택할 수 있어.
        7. 정말 아무 공식 페이지도 찾을 수 없을 때만 manualUrl을 빈 문자열("")로 반환해.
        8. URL은 500자 이하의 정상적인 URL만 반환해.
        9. customerCenter는 공식 고객센터 정보를 작성해. 확실하지 않으면 빈 문자열("")로 반환해.

        반드시 아래 JSON 형식으로만 답해.
        마크다운, 코드블록, 설명 문장, 출처 표기, citation은 절대 넣지 마.

        {{
        "manualUrl": "대체 공식 매뉴얼 또는 공식 지원 페이지 URL",
        "customerCenter": "고객센터 정보"
        }}
    """

    response = client.responses.create(
        model=OPENAI_MODEL,
        tools=[
            {
                "type": "web_search_preview"
            }
        ],
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output_text = response.output_text.strip()
    print("[DEBUG] alternative page raw:", output_text)

    return parse_manual_search_response(
        output_text=output_text,
        error_label="대체 공식 페이지"
    )


# 공식 지원 페이지 내부에서 PDF 매뉴얼 링크 후보 탐색
def find_pdf_url_from_page(page_url: str) -> str:
    if not page_url:
        return ""

    try:
        # 공식 지원 페이지 HTML 추출
        response = requests.get(
            page_url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        candidates = []

        # 페이지 내부 링크 중 매뉴얼/PDF/다운로드 관련 링크를 후보로 수집
        for link in soup.find_all("a"):
            href = link.get("href")
            text = link.get_text(" ", strip=True)

            if not href:
                continue

            full_url = urljoin(page_url, href)
            lower_url = full_url.lower()
            lower_text = text.lower()

            if (
                ".pdf" in lower_url
                or "manual" in lower_url
                or "userguide" in lower_url
                or "user-guide" in lower_url
                or "download" in lower_url
                or "사용설명서" in text
                or "매뉴얼" in text
                or "다운로드" in text
                or "manual" in lower_text
            ):
                candidates.append(clean_url(full_url))

        # 후보 중 실제 PDF 링크를 가장 우선해서 반환
        for candidate in candidates:
            if ".pdf" in candidate.lower():
                return candidate

        return ""

    except Exception as e:
        print("[ERROR] PDF 링크 탐색 실패:", e)
        return ""


# PDF 파일을 다운로드한 뒤 텍스트 추출
def extract_text_from_pdf(pdf_url: str) -> str:
    if not pdf_url:
        return ""

    try:
        # PDF 파일 다운로드
        response = requests.get(
            pdf_url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        # PDF가 아닌 응답이면 바로 포기
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            print("[ERROR] PDF가 아닌 응답입니다:", pdf_url, content_type)
            return ""

        # fitz가 읽을 수 있도록 임시 PDF 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        try:
            doc = fitz.open(temp_file_path)

            text_parts = []
            max_pages = min(len(doc), 12)

            # 너무 긴 PDF는 앞부분 일부 페이지만 추출해 요약 비용과 시간을 줄이기
            for page_index in range(max_pages):
                page = doc[page_index]
                text_parts.append(page.get_text())

            doc.close()

            return "\n".join(text_parts).strip()

        finally:
            # 임시 파일은 사용 후 삭제
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        print("[ERROR] PDF 텍스트 추출 실패:", e)
        return ""


# PDF가 없을 경우 공식 페이지의 본문 텍스트 추출
def extract_text_from_page(page_url: str) -> str:
    if not page_url:
        return ""

    try:
        # PDF가 없을 경우 공식 지원 페이지의 HTML 추출
        response = requests.get(
            page_url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # 요약에 불필요한 스크립트/스타일 영역은 제거
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # 페이지 전체 텍스트를 추출하되 모델 입력 길이를 고려해 일부만 사용
        text = soup.get_text(" ", strip=True)
        return text[:12000]

    except Exception as e:
        print("[ERROR] 페이지 텍스트 추출 실패:", e)
        return ""


# OpenAI 요약 응답을 API 응답 형식에 맞게 정리
def parse_manual_summary_response(
    output_text: str,
    manual_url: str,
    customer_center: str,
    error_label: str
) -> dict:
    try:
        # 요약 결과를 JSON으로 파싱
        result = json.loads(output_text)

        # 매뉴얼 URL, 요약문, 고객센터 정보를 각각 정리
        return {
            "manualUrl": clean_url(result.get("manualUrl", manual_url)),
            "manualSummary": clean_summary_text(result.get("manualSummary", "")),
            "customerCenter": clean_text(result.get("customerCenter", customer_center))
        }

    except json.JSONDecodeError:
        print(f"[ERROR] {error_label} JSON 파싱 실패. 원본 응답:", output_text)
        return {
            "manualUrl": manual_url,
            "manualSummary": "",
            "customerCenter": customer_center
        }


# 추출한 매뉴얼/지원 페이지 텍스트를 3개 섹션으로 요약
def summarize_manual_from_text(
    brand_name: str,
    product_name: str,
    model_number: str,
    manual_url: str,
    manual_text: str,
    customer_center: str
) -> dict:
    prompt = f"""
        너는 공식 제품 매뉴얼 내용을 사용자가 이해하기 쉽게 요약하는 AI야.

        제품 정보:
        - 브랜드명: {brand_name}
        - 제품명: {product_name}
        - 모델명: {model_number}
        - 매뉴얼 URL: {manual_url}
        - 고객센터 정보: {customer_center}

        아래는 공식 매뉴얼 또는 공식 지원 페이지에서 추출한 텍스트야.

        매뉴얼/지원 텍스트:
        {manual_text[:12000]}

        요약 규칙:
        1. 제품 스펙 소개만 쓰지 마.
        2. 페이지나 설명서 자체를 소개하지 마.
        3. "설명서입니다", "안내합니다", "제공됩니다", "확인할 수 있습니다" 같은 표현을 쓰지 마.
        4. manualSummary는 반드시 아래 3개 라벨을 포함해 작성해:
        - 사용 방법:
        - 관리 방법:
        - 고장/수리 안내:

        5. 사용 방법은 6~8문장으로 작성해.
        단순한 기능 소개가 아니라, 사용자가 제품을 실제로 사용할 때 따라 할 수 있는 절차 중심으로 작성해.
        전원 연결/전원 켜기, 초기 설정, 기본 조작, 주요 기능 선택, 사용 전 확인 사항, 사용 후 마무리 과정을 제품 유형에 맞게 포함해.
        제품에 버튼, 리모컨, 앱, 디스플레이, 다이얼, 터치패널, 연결 포트 등이 있다면 해당 조작 요소를 활용한 사용 흐름을 설명해.
        "지원한다", "탑재되어 있다", "사용할 수 있다"처럼 기능을 나열하는 문장만 반복하지 마.
        가능하면 "...을 누른다", "...을 선택한다", "...을 연결한다", "...을 설정한다", "...을 확인한다"처럼 사용자의 행동이 드러나는 문장으로 작성해.
        공식 텍스트에서 확인할 수 없는 세부 기능명, 메뉴명, 수치, 단계는 만들어내지 마.

        6. 관리 방법은 5~7문장으로 작성해.
        제품 유형에 맞는 관리 항목만 작성해.
        공통적으로 청소, 보관, 전원/케이블 상태, 외관 관리, 정기 점검, 안전한 사용 환경을 포함할 수 있어.
        제품에 필터, 먼지통, 물통, 배수구, 세제함, 냉각부, 흡입구, 통풍구, 배터리, 화면, 렌즈, 날개, 브러시, 소모품 등이 있는 경우 해당 부품의 관리 방법을 중심으로 작성해.
        해당 제품에 존재하지 않는 부품이나 관리 항목은 억지로 넣지 마.
        물 세척 가능 여부, 세제 사용 가능 여부, 분해 가능 여부처럼 안전과 관련된 내용은 공식 텍스트에서 확인되는 범위에서만 작성해.

        7. 고장/수리 안내는 3~5문장으로 작성해.
        제품이 정상 작동하지 않을 때 사용자가 먼저 확인할 수 있는 기본 항목을 설명해.
        전원 연결 상태, 설정 상태, 연결 상태, 부품 장착 상태, 소모품 상태, 이물질 여부, 과열 여부, 오류 표시 여부 등을 제품 유형에 맞게 포함해.
        사용자가 임의로 분해하거나 비공식 수리를 시도하도록 안내하지 마.

        8. 고객센터 전화번호나 연락처는 manualSummary에 쓰지 말고, 반드시 customerCenter 필드에만 작성해.
        9. 문장 안에 URL, 출처 표기, 마크다운 링크, citation을 넣지 마.
        10. 한국어로 작성해.
        11. manualSummary는 반드시 아래 형식처럼 섹션 사이에 줄바꿈 문자 \n\n을 넣어 작성해.
            "사용 방법: ...\n\n관리 방법: ...\n\n고장/수리 안내: ..."
        12. 세 섹션은 한 줄로 이어 쓰지 말고 반드시 분리해.
        13. 매뉴얼 텍스트에 제품별 구체적인 사용/관리 정보가 부족한 경우에도, 제품군에 맞는 일반적인 안전 사용법 수준에서만 요약하고 구체적인 기능을 지어내지 마.

        customerCenter 작성 규칙:
        1. customerCenter에는 브랜드의 공식 고객센터 또는 공식 지원 정보를 작성해.
        2. 확실한 전화번호를 찾은 경우 전화번호를 포함해.
        3. customerCenter에는 고객센터 정보만 작성하고, 요약 문장을 넣지 마.
        4. customerCenter에도 마크다운 링크를 넣지 마.

        중요:
        manualSummary에는 반드시 "사용 방법:", "관리 방법:", "고장/수리 안내:" 세 라벨이 모두 정확히 포함되어야 해.
        매뉴얼 텍스트에 특정 섹션 정보가 부족하더라도 라벨을 생략하지 마.
        정보가 부족한 섹션은 제품군에 맞는 일반적인 안전 사용/관리 수준에서 작성하되, 확인되지 않은 구체적인 기능명이나 수치는 만들지 마.
        절대 한 섹션만 작성하지 마.

        반드시 아래 JSON 형식으로만 답해.
        마크다운, 코드블록, 설명 문장, 출처 표기, citation은 절대 넣지 마.

        {{
            "manualUrl": "{manual_url}",
            "manualSummary": "사용 방법: ...\\n\\n관리 방법: ...\\n\\n고장/수리 안내: ...",
            "customerCenter": "{customer_center}"
        }}
    """

    # 매뉴얼 텍스트를 OpenAI 모델에 전달해 요약 결과를 생성
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output_text = response.output_text.strip()

    result = parse_manual_summary_response(
        output_text=output_text,
        manual_url=manual_url,
        customer_center=customer_center,
        error_label="매뉴얼 요약"
    )

    if not has_required_summary_sections(result["manualSummary"]):
        print("[WARN] 요약 섹션 누락. 재요약 시도")
        return retry_summarize_with_required_sections(
            brand_name=brand_name,
            product_name=product_name,
            model_number=model_number,
            manual_url=manual_url,
            manual_text=manual_text,
            customer_center=customer_center
        )

    return result


# PDF 매뉴얼이 있으면 PDF 텍스트를 우선 추출해 요약
def summarize_from_pdf_if_available(
    brand_name: str,
    product_name: str,
    model_number: str,
    page_url: str,
    customer_center: str
) -> Optional[dict]:
    # URL 자체가 PDF면 바로 사용하고, 일반 페이지면 내부 PDF 링크 탐색
    if ".pdf" in page_url.lower():
        pdf_url = page_url
    else:
        pdf_url = find_pdf_url_from_page(page_url)

    # PDF 링크가 없으면 페이지 텍스트 요약 흐름으로 넘김
    if not pdf_url:
        return None

    # PDF에서 텍스트 추출
    pdf_text = extract_text_from_pdf(pdf_url)

    # PDF 텍스트를 읽지 못하면 페이지 텍스트 요약 흐름으로 넘김
    if not pdf_text:
        return None

    # PDF 텍스트를 기준으로 매뉴얼 요약 생성
    return summarize_manual_from_text(
        brand_name=brand_name,
        product_name=product_name,
        model_number=model_number,
        manual_url=pdf_url,
        manual_text=pdf_text,
        customer_center=customer_center
    )


# 공식 페이지 검색부터 PDF/페이지 텍스트 추출, 요약까지 전체 매뉴얼 흐름 처리
def search_and_summarize_manual(
    brand_name: str,
    product_name: str,
    model_number: str
) -> dict:
    # 1차로 제조사 공식 매뉴얼 또는 지원 페이지 검색
    page_result = find_official_manual_page(
        brand_name=brand_name,
        product_name=product_name,
        model_number=model_number
    )

    page_url = page_result.get("manualUrl", "")
    customer_center = page_result.get("customerCenter", "")

    # 1차 검색에 실패하면 대체 공식 페이지를 한 번 더 검색
    if not page_url:
        print("[WARN] 공식 매뉴얼 1차 검색 실패. 대체 검색 시도")
        alternative_result = find_alternative_official_page(
            brand_name=brand_name,
            product_name=product_name,
            model_number=model_number,
            failed_url=""
        )

        page_url = alternative_result.get("manualUrl", "")
        customer_center = alternative_result.get("customerCenter", customer_center)

        if not page_url:
            return {
                "manualUrl": "",
                "manualSummary": "",
                "customerCenter": customer_center
            }

    pdf_summary = summarize_from_pdf_if_available(
        brand_name=brand_name,
        product_name=product_name,
        model_number=model_number,
        page_url=page_url,
        customer_center=customer_center
    )

    if pdf_summary:
        return pdf_summary

    # PDF가 없거나 추출에 실패하면 공식 페이지 본문 텍스트를 사용
    page_text = extract_text_from_page(page_url)

    # 페이지 텍스트도 가져오지 못하면 대체 페이지를 다시 검색
    if not page_text:
        alternative_result = find_alternative_official_page(
            brand_name=brand_name,
            product_name=product_name,
            model_number=model_number,
            failed_url=page_url
        )

        alternative_url = alternative_result.get("manualUrl", "")
        alternative_customer_center = alternative_result.get("customerCenter", "")

        if alternative_url:
            page_url = alternative_url
            customer_center = alternative_customer_center or customer_center

            pdf_summary = summarize_from_pdf_if_available(
                brand_name=brand_name,
                product_name=product_name,
                model_number=model_number,
                page_url=page_url,
                customer_center=customer_center
            )

            if pdf_summary:
                return pdf_summary

            # 대체 페이지에 PDF가 없으면 페이지 텍스트를 다시 추출
            page_text = extract_text_from_page(page_url)

    # 모든 텍스트 추출에 실패하면 URL만 반환하고 요약은 비움
    if not page_text:
        return {
            "manualUrl": page_url,
            "manualSummary": "",
            "customerCenter": customer_center
        }

    # 최종적으로 확보한 페이지 텍스트를 기준으로 요약
    return summarize_manual_from_text(
        brand_name=brand_name,
        product_name=product_name,
        model_number=model_number,
        manual_url=page_url,
        manual_text=page_text,
        customer_center=customer_center
    )


# 결과에 필수 섹션이 누락된 경우 같은 텍스트로 다시 요약 시도
def retry_summarize_with_required_sections(
    brand_name: str,
    product_name: str,
    model_number: str,
    manual_url: str,
    manual_text: str,
    customer_center: str
) -> dict:
    # 첫 요약에서 필수 섹션이 빠진 경우 더 강한 조건으로 재요약
    prompt = f"""
        너는 제품 매뉴얼 요약 결과를 정해진 형식으로 다시 작성하는 AI야.

        제품 정보:
        - 브랜드명: {brand_name}
        - 제품명: {product_name}
        - 모델명: {model_number}
        - 매뉴얼 URL: {manual_url}

        아래 텍스트를 바탕으로 manualSummary를 다시 작성해.

        매뉴얼/지원 텍스트:
        {manual_text[:12000]}

        반드시 지켜야 할 조건:
        1. manualSummary에는 아래 3개 라벨이 모두 정확히 포함되어야 해.
        - 사용 방법:
        - 관리 방법:
        - 고장/수리 안내:
        2. 절대 한 섹션만 작성하지 마.
        3. 사용 방법은 사용자가 실제로 따라 할 수 있는 절차 중심으로 6~8문장 작성해.
        4. 관리 방법은 제품 유형에 맞는 청소, 보관, 부품 관리, 전원/연결 상태 점검 중심으로 5~7문장 작성해.
        5. 고장/수리 안내는 사용자가 먼저 확인할 항목과 공식 서비스센터 안내 중심으로 3~5문장 작성해.
        6. 제품에 없는 부품이나 기능은 지어내지 마.
        7. 고객센터 전화번호는 manualSummary에 쓰지 마.
        8. 한국어로 작성해.

        반드시 아래 JSON 형식으로만 답해.

        {{
            "manualUrl": "{manual_url}",
            "manualSummary": "사용 방법: ...\\n\\n관리 방법: ...\\n\\n고장/수리 안내: ...",
            "customerCenter": "{customer_center}"
        }}
    """

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output_text = response.output_text.strip()

    return parse_manual_summary_response(
        output_text=output_text,
        manual_url=manual_url,
        customer_center=customer_center,
        error_label="재요약"
    )


# 제품/매뉴얼/대화 이력을 기반으로 AI 채팅 답변 생성
def generate_chat_answer(request: AiChatRequest) -> str:
    prompt = build_chat_prompt(request)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.output_text.strip()


# AI 채팅 답변 생성을 위한 프롬프트 구성
def build_chat_prompt(request: AiChatRequest) -> str:
    history_text = format_chat_history(request.history)

    if request.product is None:
        context = build_general_chat_context()
    else:
        context = build_product_chat_context(
            product=request.product,
            manual_content=request.manualContent or "",
            manual_url=request.manualUrl or ""
        )

    prompt = f"""
너는 ORE 서비스의 제품 관리 AI 챗봇이야.

답변 규칙:
1. 한국어로 답변해.
2. 사용자가 이해하기 쉽게 짧고 명확하게 답변해.
3. 확실하지 않은 내용은 단정하지 마.
4. 제품 질문에서 manualContent가 제공된 경우, manualContent를 가장 우선하여 답변해.
5. manualContent에 없는 내용은 "제공된 매뉴얼 내용 기준으로는 확인되지 않지만"처럼 표현하고 일반적인 안내로 답변해.
6. manualUrl은 manualContent만으로 답변이 부족할 때 참고 가능한 원본 매뉴얼 위치로만 취급해.
7. 사용자가 위험한 분해, 임의 수리, 전기 작업을 요구하면 직접 수리 방법을 자세히 안내하지 말고 공식 서비스센터 문의를 권장해.
8. 답변에는 마크다운 표를 쓰지 마.
9. 너무 길게 설명하지 말고 3~6문장 정도로 답변해.

{context}

이전 대화:
{history_text}

사용자 질문:
{request.message}
    """

    return prompt.strip()


# 제품 질문용 문맥 구성
def build_product_chat_context(
    product: ChatProductInfo,
    manual_content: str,
    manual_url: str
) -> str:
    product_display_name = product.nickname or product.name or "선택한 제품"

    context = f"""
현재 사용자는 특정 제품에 대해 질문하고 있어.

제품 정보:
- 제품 ID: {product.productId}
- 제품명: {product.name}
- 제품 별명: {product.nickname}
- 카테고리: {product.category}
- 화면 표시용 제품명: {product_display_name}
    """

    if manual_content.strip():
        context += f"""

아래 내용은 사용자가 확인/수정/확정한 최종 매뉴얼 내용이야.
AI 답변은 반드시 이 내용을 가장 우선 근거로 삼아야 해.

manualContent:
{manual_content[:12000]}
        """
    else:
        context += """

manualContent가 제공되지 않았어.
이 경우 제품 정보와 일반적인 제품 관리 지식만 바탕으로 답변하되, 단정하지 마.
        """

    if manual_url.strip():
        context += f"""

manualUrl:
{manual_url}

manualUrl은 원본 매뉴얼 위치야.
현재 답변에서는 manualContent를 우선하고, manualContent만으로 부족한 경우 manualUrl이 참고 자료로 존재한다고 판단해.
단, 실제로 manualUrl 내용을 읽지 못했다면 manualUrl에 있는 내용을 본 것처럼 말하지 마.
        """

    return context.strip()


# 기타 질문용 문맥 구성
def build_general_chat_context() -> str:
    return """
현재 사용자는 특정 제품을 선택하지 않은 기타 질문을 하고 있어.

답변 범위:
- ORE 서비스 사용 방법
- 제품 등록, 매뉴얼 확인, 수리 이력 관리, 수리 예정 알림 등 서비스 관련 안내
- 일반적인 제품 관리 질문

특정 제품 매뉴얼이 필요한 질문이면, 제품을 선택한 뒤 다시 질문하도록 자연스럽게 안내해.
    """.strip()


# 이전 대화 이력을 프롬프트에 넣기 좋은 형태로 변환
def format_chat_history(history: list[ChatHistoryItem]) -> str:
    if not history:
        return "이전 대화 없음"

    formatted_messages = []

    # 너무 긴 대화는 최근 10개만 사용
    recent_history = history[-10:]

    for item in recent_history:
        role = "사용자" if item.role == "USER" else "AI"
        content = item.content.strip()

        if content:
            formatted_messages.append(f"{role}: {content}")

    if not formatted_messages:
        return "이전 대화 없음"

    return "\n".join(formatted_messages)