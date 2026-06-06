import json
import os
from typing import Literal

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
    sourceType: Literal["receipt", "product", "warranty", "manual"] = Field(
        ...,
        description="이미지 출처 타입"
    )


class ModelOcrResponse(BaseModel):
    ocrText: str
    brandName: str
    productName: str
    modelNumber: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "AI server is running"
    }


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
            content={
                "message": "이미지 분석에 실패했습니다."
            }
        )


def extract_product_info(image_url: str, source_type: str) -> dict:
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
        print("[ERROR] JSON 파싱 실패. 원본 응답:", output_text)
        raise ValueError("AI 응답을 JSON으로 파싱할 수 없습니다.")