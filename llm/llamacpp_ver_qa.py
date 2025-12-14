from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import json
from typing import Optional, Any

import re
import random
import torch
import pandas as pd
from difflib import SequenceMatcher

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from utils.text_pp import bandae_to_jondae
from region_config import REGION_MAP  

from llama_cpp import Llama
import time
app = FastAPI()

BANNED_KEYWORDS = [
    "빨갱이", "좌빨", "토착왜구", "일베", "메갈", "워마드", "한남", "한녀", "남혐", "여혐",
    "맘충", "김치녀", "된장녀", "보슬아치", "틀딱", "홍어", "절라", "짱깨", "쪽바리", "조센징",
    "드럼통", "탄핵", "부정선거", "내로남불", "종북", "친일파", "독재", "사이비",
    "좆", "씹", "창녀", "걸레", "병신", "미친놈", "개새끼", "죽여", "살인", "자살",
    "테러", "강간", "성폭행", "매춘", "음란", "변태", "쓰레기", "양아치", "꼽추", "절름발이",
    "5.18", "세월호", "천안함", "연평해전", "김일성", "김정일", "김정은",
    "섹스", "오르가즘", "포르노", "성교", "삽입", "오입", "보지", "자지",
    "대선", "총선", "선거", "공천", "검찰", "국정원", "사법부", "특검", "청문회",
    "입법", "행정", "사법", "헌법", "대통령실", "비리", "부패", "뇌물", "접대",
    "간첩", "북한", "주사파", "좌파", "우파", "극우", "극좌", "보수", "진보",
    "이념", "숙청", "정치공작", "정경유착", "매국노", "민주당", "국민의힘", "진보당",
    "투표조작", "게이트", "책임론", "사퇴", "논란", "여론조작", "언론탄압", "친북", "친중","존나","개심심","개짜증",
    "씨발", "씹새", "개새", "개좆", "좆같", "좆나", "존나", "존맛", "존멋",
    "병맛", "썅", "개색", "개같", "미친년", "미친놈", "개년", "등신",
    "돌아이", "도라이", "닥쳐", "입닥쳐", "입다물어",
    "시발", "존나", "좆같", "병신", "븅신",
    "뻐큐", "뻐큐머겅", "퍼큐", "퍽유",
    "똥꼬", "방구쟁이", "지랄", "지랄해",
    "발발이", "호구", "멍청이", "멍청하네",
    "썅년", "썅놈",
    "거지같", "노가다꾼", "잡것", "개같", "개지랄", "개빡", "개좆", "개년", "개놈",
]

def check_invalid_question(text: str) -> bool:
    return any(word in text for word in BANNED_KEYWORDS)

TAG_RE   = re.compile(r"<[^>]*>")
ALLOW_RE = re.compile(r'[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\s\.\,\!\?\:\"\'\(\)\-\_\/&%:;]')
DUMMY_PATTERNS = [
    re.compile(r"<dummy\d+>", re.I),
    re.compile(r"\[DUMMY\]", re.I),
]

def sanitize_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = TAG_RE.sub(" ", s)
    s = s.replace("<", " ").replace(">", " ")
    s = ALLOW_RE.sub(" ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def clean_dummy_tokens(text: Any) -> str:
    s = "" if text is None else str(text)
    for pat in DUMMY_PATTERNS:
        s = pat.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def looks_garbage(txt: str) -> bool:
    if not txt:
        return True
    if "://://://:" in txt:
        return True
    if len(set(txt.strip())) <= 3:
        return True
    return False

INFO_TRIGGERS = ["운영시간", "휴관일", "입장료", "관람", "전시", "프로그램", "체험",
                 "예약", "층", "어디에 있어", "위치", "주차", "수유실", "유모차",
                 "단체", "해설", "교육", "사진", "촬영"]

from rapidfuzz import fuzz

DOMAIN_RULES = [
    {
        "canonical": "문순득",
        "category": "person",
        "prefixes": [
            "문순득", "문순드", "문슌드", "문슨드", "문쑨드", "문쓘드", "문숭드",
            "문순딕", "문슌딕", "문순듁", "문슌듁",
            "문순드기", "문순드긱", "문슌드기", "문슌드긱",
            "문순득씨", "문순둑", "문순덛", "문순덕",
        ],
    },
    {
        "canonical": "국립인천해양박물관",
        "category": "museum",
        "prefixes": [
            "국립인천해양박물관", "국립인천해양박믈관",
            "국립인천박물관",
            "인천해양박물관",
            "해양박물관",
            "박물관", "박믈관", "박물괸", "박믈괸", "박믈꽌",
            "방물관", "방믈관", "방물괸", "방믈괸", "방믈꽌",
            "밥물관", "밥믈관", "밥물괸", "밥믈괸",
            "빡물관", "빡믈관", "박불관", "박무관", "박몰관",
        ],
    },
    {
        "canonical": "지명",
        "category": "place",
        "prefixes": [
            "류큐", "유큐", "뉴큐", "류크", "류쿠", "루큐", "루쿠", "리큐",
            "유구", "유국", "류구",
            "오키나와", "오끼나와", "오키나왁", "오키나왈",
            "여송", "여숑",
            "필리핀", "피리핀", "퓔리핀",
            "루손", "루송", "루순", "루썬",
            "비간", "비깐",
            "복건성", "복껀성", "복겐성",
            "마카오", "마까오", "마카우",
            "북경", "복경", "부경",
            "의주", "의쥬",
            "청나라", "청라라",
            "인천", "인쳔", "인췬", "인춘", "인츈", "인첸", "인쳰",
            "섬사람", "섬생활", "섬거주민", "섬",
        ],
    },
    {
        "canonical": "개념",
        "category": "concept",
        "prefixes": [
            "조선후기", "조선 후기",
            "표류", "펴류", "표유",
            "풍랑", "풍낭",
            "항해", "항행",
            "출항", "귀환", "표착",
            "표해시말", "漂海始末",
            "정약전", "정약용", "실학",
            "구술기록", "민중기록", "필사본",
            "언어표", "언어기록", "언문", "언문 기록",
            "류큐어", "여송어", "필리핀어", "조선어",
            "해양문화", "섬문화", "문화교류",
            "해양사", "해양교류사", "표류사",
            "해상무역", "교역선", "선박", "선박 구조",
            "해양유물", "표류기록", "항해기록",
            "세계 인식", "세계사적 의의",
            "바다와 인간", "생존과 기록",
            "조선인의 세계화",
        ],
    },
]

DOMAIN_CANDIDATES = []
for rule in DOMAIN_RULES:
    if rule["canonical"] in ("지명", "개념"):
        continue 
    for p in rule["prefixes"]:
        DOMAIN_CANDIDATES.append((p, rule["canonical"]))

def is_english_query(text: str) -> bool:
    """
    - 한글이 하나도 없고 라틴 알파벳이 있으면 영어로 간주
    - 또는 'answer in english', '영어로' 같은 표현이 있으면 영어 요청으로 간주
    """
    if not text:
        return False

    t = text.strip()
    has_hangul = bool(re.search(r"[가-힣]", t))
    has_latin = bool(re.search(r"[A-Za-z]", t))
    lower_t = t.lower()

    # 명시적으로 영어 요청
    if "answer in english" in lower_t or "reply in english" in lower_t or "영어로" in t:
        return True

    # 한글 없이 알파벳만 있으면 영어로 간주
    if has_latin and not has_hangul:
        return True

    return False

def normalize_domain_terms(text: str, threshold: int = 85) -> str:
    if not text:
        return text

    tokens = text.split()
    new_tokens = []

    for tok in tokens:
        if len(tok) <= 2:
            new_tokens.append(tok)
            continue

        if not re.search(r"[가-힣A-Za-z]", tok):
            new_tokens.append(tok)
            continue

        best_score = 0
        best_canonical = None
        best_prefix = None

        for p, canonical in DOMAIN_CANDIDATES:
            if abs(len(tok) - len(p)) > 4:
                continue

            score = fuzz.partial_ratio(tok, p)
            if score > best_score:
                best_score = score
                best_canonical = canonical
                best_prefix = p

        if best_score >= threshold and best_prefix:
            if len(tok) < len(best_prefix):
                new_tokens.append(tok)
                continue

            new_tokens.append(best_canonical)
        else:
            new_tokens.append(tok)

    return " ".join(new_tokens)


def classify_query(query: str) -> str:
    q = query.lower()

    if any(k in q for k in INFO_TRIGGERS):
        return "info"

    return "casual"












def detect_lang(text: str) -> str:
    """
    질문에 한글이 하나라도 있으면 'ko', 아니면 'en'으로 간주
    """
    if re.search(r"[가-힣]", text):
        return "ko"
    return "en"





BANNED_FALLBACKS_EN = [
    "I cannot quite grasp what you mean.",
    "I’m sorry, but that is beyond the scope of the guidance I can provide. Please ask another question.",
    "I apologize for not understanding your intention. Could you please ask again?",
    "I wish to give you a better answer, but it seems my understanding is lacking. Could you please ask again?",
    "It seems that what you've asked is outside the range of what I know. If it is related to the museum, I will gladly answer."
]



BANNED_FALLBACKS = [
    "제가 그 뜻을 잘 헤아리지 못하겠소",
    "송구하오나, 안내 범위를 벗어나오. 다른 문의사항을 부탁드리오",
    "뜻을 파악하지 못하여 미안하오. 다시 여쭈어 주실 수 있겠소?",
    "더 나은 답변을 해 드리고 싶은데, 저의 이해가 부족한 듯싶소. 다시 여쭈어 주실 수 있겠소?",
    "여쭈어 주신 내용은 제가 아는 범위 밖에 있는 듯하오. 박물관 관련 질문이라면, 성심껏 답해 드리겠소."
]

model_name = "./local_model/BGE-m3-ko"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Embedding model using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": device,
                  "local_files_only": True},
    
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local(
    "faiss_munsunduk_db_pdf/",
    embeddings,
    allow_dangerous_deserialization=True,
)
vectorstore_pre = FAISS.load_local(
    "faiss_munsunduk_qna_db",
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# retriever_pre = vectorstore_pre.as_retriever(search_kwargs={"k": 1})











# # =========================
# # 한/영 임베딩 & FAISS 로드
# # =========================
# device = "cuda"
# print(f"Embedding model using device: {device}")

# # TODO: 실제 경로 맞게 수정
# KO_EMBED_MODEL      = "/home/user/hayang/llm/local_model/BGE-m3-ko"
# EN_EMBED_MODEL      = "/home/user/hayang/llm/local_model/bge-m3"   # 영어용 임베딩 모델
# KO_FAISS_PATH       = "faiss_munsunduk_db/"        # 기존 한글 인덱스
# KO_FAISS_QNA_PATH   = "faiss_munsunduk_qna_db"     # 기존 QnA 인덱스 (엑셀 기반)
# EN_FAISS_PATH       = "faiss_munsunduk_db_en/"     # 영어 인덱스 (네가 만든 경로로 수정)

# # --- Korean embeddings & index ---
# embeddings_ko = HuggingFaceEmbeddings(
#     model_name=KO_EMBED_MODEL,
#     model_kwargs={
#         "device": device,
#         "local_files_only": True,
#     },
#     encode_kwargs={"normalize_embeddings": True},
# )

# vectorstore_ko = FAISS.load_local(
#     KO_FAISS_PATH,
#     embeddings_ko,
#     allow_dangerous_deserialization=True,
# )
# vectorstore_pre_ko = FAISS.load_local(
#     KO_FAISS_QNA_PATH,
#     embeddings_ko,
#     allow_dangerous_deserialization=True,
# )

# retriever_ko = vectorstore_ko.as_retriever(search_kwargs={"k": 3})
# retriever_pre_ko = vectorstore_pre_ko.as_retriever(search_kwargs={"k": 1})

# # --- English embeddings & index ---
# #  * EN_FAISS_PATH 에는 영어로 만든 인덱스가 있어야 함 (예: 영문 QnA / 영문 도메인 문서)
# embeddings_en = HuggingFaceEmbeddings(
#     model_name=EN_EMBED_MODEL,
#     model_kwargs={
#         "device": device,
#         "local_files_only": True,
#     },
#     encode_kwargs={"normalize_embeddings": True},
# )

# vectorstore_en = FAISS.load_local(
#     EN_FAISS_PATH,
#     embeddings_en,
#     allow_dangerous_deserialization=True,
# )
# retriever_en = vectorstore_en.as_retriever(search_kwargs={"k": 3})

# print("[FAISS] ko/en vectorstores loaded.")


XLSX_PATH = "./data/haeyang_qna_v2_4.xlsx"

df_raw = pd.read_excel(XLSX_PATH)
df_raw = df_raw.iloc[:, 0:3]
df_raw.columns = ["region", "question", "answer"]
df_raw.ffill(inplace=True)

qa_dict = {}
for _, row in df_raw.iterrows():
    key = (str(row["region"]).strip(), str(row["question"]).strip())
    qa_dict.setdefault(key, []).append(str(row["answer"]).strip())

df_qna = pd.DataFrame(
    [{"region": k[0], "question": k[1], "answers": v} for k, v in qa_dict.items()]
)

def check_predefined_answer_from_xlsx(query: str, threshold: float = 0.50) -> Optional[dict]:
    results = retriever.get_relevant_documents(query)
    print(f"[PREDEFINED SEARCH] {results}")

    if not results:
        return None

    matched_question = results[0].page_content.strip()
    sim = SequenceMatcher(None, query.strip(), matched_question).ratio()
    print(f"[PREDEFINED] matched='{matched_question}', sim={sim:.3f}")

    if sim < threshold:
        return None

    matched_rows = df_qna[df_qna["question"] == matched_question]
    if matched_rows.empty:
        return None

    row = matched_rows.sample(n=1).iloc[0]
    return {
        "region": row["region"],
        "answer": random.choice(row["answers"]),
    }






# def check_predefined_answer_from_xlsx(
#     query: str,
#     lang: str = "ko",
#     threshold: float = 0.50,
# ) -> Optional[dict]:
#     # 현재 엑셀/프리디파인 QnA는 한글 기준이므로 영어면 사용 안 함
#     if lang != "ko":
#         return None

#     results = retriever_pre_ko.get_relevant_documents(query)
#     print(f"[PREDEFINED SEARCH] {results}")

#     if not results:
#         return None

#     matched_question = results[0].page_content.strip()
#     sim = SequenceMatcher(None, query.strip(), matched_question).ratio()
#     print(f"[PREDEFINED] matched='{matched_question}', sim={sim:.3f}")

#     if sim < threshold:
#         return None

#     matched_rows = df_qna[df_qna["question"] == matched_question]
#     if matched_rows.empty:
#         return None

#     row = matched_rows.sample(n=1).iloc[0]
#     return {
#         "region": row["region"],
#         "answer": random.choice(row["answers"]),
#     }










llm = Llama(
    model_path="./local_model/Hermes-2-Pro-Llama-3-8B-Q5_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    n_batch=512,
    verbose=True,
)

SYSTEM_MSG = (
"""
당신은 인천해양국립박물관의 공식 AI 휴먼 안내원 ‘문순득’입니다.
당신은 조선 후기 실존 인물 문순득의 생애와 표류 기록을 기반으로
박물관 관람객에게 정확하고 친절한 안내를 제공하는 역할을 맡고 있습니다.

[문순득 기본 정보]
- 1777년 전라남도 신안군 우이도에서 태어난 조선 후기 상인입니다.
- 흑산도 홍어를 육지로 실어 나르고, 다시 쌀과 생활 물자를 섬으로 공급하는
  해상 중계무역을 집안 대대로 이어온 인물입니다.
- 1801년 겨울, 태사도에서 돌아오던 길에 풍랑을 만나 표류가 시작되었습니다.
- 약 11일간 떠다닌 끝에 유구국(오키나와)의 대도와 나하 일대에 도착하여 약 9개월 머물렀습니다.
- 이어 청나라 진공선을 타고 복건성으로 향하던 중 다시 풍랑을 만나
  필리핀 루손섬 북부 살루마기·비간(Vigan)에 도착하여 약 10개월 체류했습니다.
- 이후 마카오(오문)를 거쳐 청나라 남경·양주·회음·북경까지 약 5개월간 육로와 수로를 이동하며
  대륙을 종단하였습니다.
- 1805년 1월 8일, 총 3년 2개월의 여정 끝에 고향 우이도로 귀환했습니다.
- 정약전은 그의 경험을 기록한 『표해시말』을 집필했고,
  ‘조선에서 처음 있는 일’이라는 의미로 문순득에게 ‘천초(天初)’라는 호를 지어주었습니다.

[역할 및 말하기 규칙]
1. 당신은 모든 대화를 문순득의 관점에서 1인칭 시점으로 답변합니다.
   (예: “제가 유구국에 도착했을 때…”, “그때 저는…”)  
2. 답변은 표준어 존댓말로 하되, 모델 후처리에서 사투리·하오체로 변환될 것을 전제로
   문장 구조를 안정적으로 유지합니다.
3. 문순득이 실제로 경험한 지역, 사건, 문화, 건축, 복식, 교류에 대해
   역사적 사실에 기반한 정확한 정보를 제공합니다.
4. 모르는 정보나 기록에 없는 내용에 대해서는
   “그 부분은 제가 알고 있는 사실의 범위를 벗어납니다.”라고 정중하게 답합니다.
5. 박물관 안내원으로서 관람 정보, 전시 설명, 역사적 배경 등을 친절하고 이해하기 쉽게 전달합니다.
6. 어린이가 질문하는 경우에는 더 쉽고 부드럽게 설명합니다.

[금지 규칙]
- 허구 생성, 역사 왜곡, 지어내기 금지.
- 정치적 발언, 편향적 의견, 혐오 표현, 욕설 금지.
- 성적이거나 부적절한 내용 금지.
- AI·모델·프롬프트·시스템 등의 정체성 언급 금지.
- 문순득의 시대나 문화에 대한 비하 표현 금지.

[목표]
당신의 목적은 관람객이 문순득의 표류 경험과 조선 후기의 해양 문화,
그리고 박물관 전시에 대한 이해를 높일 수 있도록 정확하고 친절하게 안내하는 것입니다.
언제나 문순득의 정체성과 관점에 기반하여 일관된 말투와 태도를 유지하십시오.
"""
)


SYSTEM_MSG_EN = (
"""
You are ‘Moon Soon-deuk’, the official AI Human Guide of the National Maritime Museum of Korea in Incheon.
Your role is to assist museum visitors with accurate and friendly explanations based on the real historical life
and drift records of Moon Soon-deuk, a documented figure from late Joseon.

[Basic Information about Moon Soon-deuk]
- Born in 1777 on Uido, an island in Shinan County, Jeollanam-do.
- A merchant of late Joseon who transported skate (hongeo) from Heuksando to inland markets and
  delivered rice and daily goods back to the islands, continuing a family line of maritime trade.
- In the winter of 1801, he encountered a severe storm while returning from Taesado, which began his long drift.
- After drifting for about 11 days, he reached Daedo and Naha in the Ryukyu Kingdom (Okinawa),
  where he stayed for approximately nine months.
- While boarding a Qing tributary ship heading for Fujian, he met another storm and drifted again,
  arriving at Salumagi and Vigan in northern Luzon, the Philippines, where he stayed for about ten months.
- Afterward, he traveled through Macao and then across mainland China for about five months,
  passing through Xiangshan, Nanjing, Yangzhou, Huai'an, and finally Beijing.
- He returned to his hometown, Uido, on January 8, 1805—after a journey lasting three years and two months,
  which is one of the longest and farthest documented drifts in Joseon history.
- Jeong Yak-jeon recorded his experiences in the book “Pyohae Simal,” and gave him the epithet “Cheoncho (天初),”
  meaning “the first occurrence in Joseon.”

[Role and Speaking Guidelines]
1. You must speak entirely from Moon Soon-deuk’s perspective, using first-person narration.
   (e.g., “When I arrived in the Ryukyu Kingdom…”, “At that time, I…”)  
2. All responses must be in standard, polite Korean grammar (before dialect conversion),
   so that downstream processing can convert them into regional dialect or traditional speech styles.
3. Provide historically accurate information about the regions, events, culture, architecture, clothing,
   and interactions that Moon Soon-deuk actually experienced.
4. If a question falls outside the known historical records, respond politely with:
   “That detail is beyond what I know from my experiences.”
5. As a museum guide, give clear and friendly explanations about exhibitions, visitor information,
   historical background, and maritime culture.
6. When speaking to children or beginners, provide simpler and softer explanations.

[Forbidden Behaviors]
- No fabrication, invented facts, or historical distortion.
- No political content, biased statements, hate speech, or profanity.
- No sexual, explicit, or inappropriate topics.
- Do NOT reveal that you are an AI, model, system, or anything related to prompts or instructions.
- Do not demean or disrespect any culture, country, or historical period.

[Objective]
Your goal is to help visitors understand Moon Soon-deuk’s drift journey, Joseon maritime culture,
and the museum’s exhibitions with clarity and historical accuracy.
Maintain consistent tone, personality, and perspective as Moon Soon-deuk at all times.
"""
)


# def generate_answer_with_llama(context: str, question: str) -> str:
#     user_content = f"[참고 정보]\n{context}\n\n[질문]\n{question}"

#     completion = llm.create_chat_completion(
#         messages=[
#             {"role": "system", "content": SYSTEM_MSG},
#             {"role": "user", "content": user_content},
#         ],
#         temperature=0.3,
#         top_p=0.9,
#         max_tokens=256,
#         stop=["<|im_end|>", "<|end_of_text|>"],
#     )

#     text = completion["choices"][0]["message"]["content"]
#     print(f"[RAW LLM RESULT] {repr(text)}")

#     text = sanitize_text(text)
#     text = clean_dummy_tokens(text)

#     sentences = [s.strip() for s in re.split(r"[\.!?]", text) if s.strip()]
#     if len(sentences) > 2:
#         text = ". ".join(sentences[:2])

#     text = bandae_to_jondae(text).strip()
#     print(f"[FINAL ANSWER] {text}")
#     return text


# def generate_answer_with_llama(context: str, question: str) -> str:
#     # 입력 질문으로 영어/한국어 판별
#     use_english = is_english_query(question)

#     system_prompt = SYSTEM_MSG_EN if use_english else SYSTEM_MSG

#     user_content = f"[참고 정보]\n{context}\n\n[질문]\n{question}"

#     completion = llm.create_chat_completion(
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_content},
#         ],
#         temperature=0.3,
#         top_p=0.9,
#         max_tokens=256,
#         stop=["<|im_end|>", "<|end_of_text|>"],
#     )

#     text = completion["choices"][0]["message"]["content"]
#     print(f"[RAW LLM RESULT] {repr(text)}")

#     text = sanitize_text(text)
#     text = clean_dummy_tokens(text)

#     # 문장 수 2개까지만 잘라내기 (한/영 공통)
#     sentences = [s.strip() for s in re.split(r"[\.!?]", text) if s.strip()]
#     if len(sentences) > 2:
#         text = ". ".join(sentences[:2])

#     # 한국어일 때만 반말→존댓말 후처리
#     if not use_english:
#         text = bandae_to_jondae(text).strip()
#     else:
#         text = text.strip()

#     print(f"[FINAL ANSWER] {text}")
#     return text


def generate_answer_with_llama(context: str, question: str) -> str:
    # 입력 질문으로 영어/한국어 판별
    use_english = is_english_query(question)

    system_prompt = SYSTEM_MSG_EN if use_english else SYSTEM_MSG

    user_content = f"[참고 정보]\n{context}\n\n[질문]\n{question}"

    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=256,
        stop=["<|im_end|>", "<|end_of_text|>"],
    )

    text = completion["choices"][0]["message"]["content"]
    print(f"[RAW LLM RESULT] {repr(text)}")

    text = sanitize_text(text)
    text = clean_dummy_tokens(text)

    # 문장 수 2개까지만 잘라내기 (한/영 공통)
    sentences = [s.strip() for s in re.split(r"[\.!?]", text) if s.strip()]
    if len(sentences) > 2:
        text = ". ".join(sentences[:2])+"."

    # 한국어일 때만 반말→존댓말 후처리
    if not use_english:
        text = bandae_to_jondae(text).strip()
    else:
        text = text.strip()

    print(f"[FINAL ANSWER] {text}")
    return text


# def llm_predictor(query: str) -> str:
#     docs = retriever.get_relevant_documents(query)
#     context = "\n".join(d.page_content for d in docs)
#     return generate_answer_with_llama(context, query)





# def llm_predictor(query: str, lang: str = "ko") -> str:
#     if lang == "en":
#         docs = retriever_en.get_relevant_documents(query)
#     else:
#         docs = retriever_ko.get_relevant_documents(query)

#     context = "\n".join(d.page_content for d in docs)
#     return generate_answer_with_llama(context, query)

def llm_predictor(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    context = "\n".join(d.page_content for d in docs)
    return generate_answer_with_llama(context, query)


















def warmup_model():
    print("서버 시작 전 모델 워밍업 중...")
    try:
        _ = llm_predictor("국립인천해양박물관은 어디에 있나요?", lang="ko")
        print("워밍업 완료.")
    except Exception as e:
        print(f"워밍업 중 오류: {e}")


warmup_model()
print("서버가 정상 실행되었습니다.")

@app.post("/llm")
async def chat_with_llm(
    meta: Optional[str] = Form(None),
    text: str = Form(...),
):
    try:
        meta_data = json.loads(meta) if meta else {}
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid meta JSON"},
        )
    
    text = normalize_domain_terms(text)

    query_lang = detect_lang(text)
    # print(f"[LANG] detected = {query_lang}")
    
    print("질문:", text)
    print("메타:", meta_data)

    if check_invalid_question(text):
        if query_lang == "en":
            bad_fallback = random.choice(BANNED_FALLBACKS_EN)
        else:
            bad_fallback = random.choice(BANNED_FALLBACKS)
        return {
            "user": meta_data.get("user", "unknown"),
            "input": text,
            "response": bad_fallback,
            "region_index": "-1",
        }

    qtype = classify_query(text)
    print(f"[QTYPE] {qtype}")

    if qtype == "info":
        # predefined = check_predefined_answer_from_xlsx(text, lang=query_lang)
        predefined = check_predefined_answer_from_xlsx(text)

        print(f"predefined: {predefined}")

        if predefined:
            return {
                "user": meta_data.get("user", "unknown"),
                "input": text,
                "response": predefined["answer"],
                "region_index": str(predefined["region"]),
            }
        if query_lang == "en":
            fallback = random.choice(BANNED_FALLBACKS)
        else:
            fallback = random.choice(BANNED_FALLBACKS)
        return {
            "user": meta_data.get("user", "unknown"),
            "input": text,
            "response": fallback,
            "region_index": "-1",
        }
    
    start_predifined = time.time()
    # predefined = check_predefined_answer_from_xlsx(text, lang=query_lang)
    predefined = check_predefined_answer_from_xlsx(text)

    print(f"predefined: {predefined}")
    print(f"end predifined : {time.time() - start_predifined}")
    if predefined:
        return {
            "user": meta_data.get("user", "unknown"),
            "input": text,
            "response": bandae_to_jondae(predefined["answer"]).strip(),
            "region_index": str(predefined["region"]),
        }

    try:
    #     if query_lang == "en":
    #         docs = retriever_en.get_relevant_documents(text)
    #     else:
        docs = retriever.get_relevant_documents(text)
    except Exception as e:
        print(f"[RAG ERROR - CASUAL] {e}")
        docs = []

    if docs:
        context = "\n".join(d.page_content for d in docs)
        answer = generate_answer_with_llama(context, text)
    else:
        answer = llm_predictor(text)

    return {
        "user": meta_data.get("user", "unknown"),
        "input": text,
        "response": answer,
        "region_index": "-1",
    }
