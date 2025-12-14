import time
import json
import os
import tempfile
import pathlib
import re
import logging
import uuid
from typing import Optional, Dict, Any, Tuple, List
from difflib import SequenceMatcher
import gc

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from anyio import to_thread

from faster_whisper import WhisperModel
from transformers import pipeline, MarianMTModel, MarianTokenizer
from pydub import AudioSegment
import torch
import soundfile as sf
import librosa
import noisereduce as nr

from apiGateway import APIGateway
from config import TTS_URL, LLM_URL
import shutil




app = FastAPI()

qa_logger = logging.getLogger("QALogger")
qa_logger.setLevel(logging.INFO)
qa_file_handler = logging.FileHandler("qa_list.log", encoding="utf-8")
qa_formatter = logging.Formatter(
    "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
qa_file_handler.setFormatter(qa_formatter)
qa_logger.addHandler(qa_file_handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

AUDIO_OUTPUT_DIR = "/var/lib/tts/"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

INPUT_AUDIO_DIR = "/var/lib/stt_inputs"
os.makedirs(INPUT_AUDIO_DIR, exist_ok=True)

gateway = APIGateway(TTS_URL, LLM_URL)

stt_model = WhisperModel("small", device="cpu", compute_type="int8")

# action_classifier = pipeline(
#     "zero-shot-classification", model="facebook/bart-large-mnli"
# )


# ===============================
# Log rotate 
# ===============================

# -------------------------------
# 디렉토리 용량 관리 (오디오 100MB)
# -------------------------------
def cleanup_directory(directory: str, max_bytes: int):
    if not os.path.exists(directory):
        return

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    total_size = sum(os.path.getsize(f) for f in files)

    if total_size <= max_bytes:
        return

    # 오래된 파일 순으로 삭제
    files_sorted = sorted(files, key=lambda x: os.path.getmtime(x))
    for f in files_sorted:
        if total_size <= max_bytes:
            break
        try:
            size = os.path.getsize(f)
            os.remove(f)
            total_size -= size
            print(f"[CLEANUP] Deleted: {f}")
        except Exception as e:
            print(f"[CLEANUP ERROR] {f}: {e}")


# -------------------------------
# qa_list.log 파일 전용 rotate (30MB)
# -------------------------------
def rotate_qa_log(logger, handler, max_bytes):
    log_path = handler.baseFilename

    if os.path.exists(log_path) and os.path.getsize(log_path) >= max_bytes:
        logger.removeHandler(handler)
        handler.close()

        # 기존 파일 → 백업
        backup_path = log_path + ".1"
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(log_path, backup_path)

        # 새 핸들러 만들기
        new_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        new_handler.setFormatter(formatter)
        logger.addHandler(new_handler)

        return new_handler

    return handler

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path

# 로컬 모델 경로 (snapshot 폴더 안)
local_model_dir = Path(
    "./models/bart-large-mnli"
)

# 토크나이저와 모델을 직접 로드 (여기서만 local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(str(local_model_dir), local_files_only=True)

# pipeline 에 이미 로드된 객체를 직접 전달
action_classifier = pipeline(
    task="zero-shot-classification",
    model=model,
    tokenizer=tokenizer
)

EMOTION_ID_MAP: Dict[str, int] = {
    "happy": 1,
    "positive": 1,
    "happiness": 1,
    "neutral": 2,
    "negative": 2,
    "none": 2,
    "surprise": 3,
    "joy": 4,
    "sad": 5,
    "sadness": 5,
    "fear": 2,
    "anger": 6,
    "angry": 6,
    "disgust": 6,
}
EMOTION_LIST = list(EMOTION_ID_MAP.keys())

ACTION_ID_MAP: Dict[str, int] = {
    "Idle": 1,
    "Nod_Yes": 2,
    "Shake_No": 3,
    "Greeting": 4,
    "Explaining": 5,
    "Thinking": 6,
}
ACTION_LIST = list(ACTION_ID_MAP.keys())

DOMAIN_TERMS = [
    "문순득",
    "국립인천해양박물관",
    "국립 인천해양박물관",
    "인천해양박물관",
    "해양박물관",
    "인천항",
    "박물관",
    "전시관",
    "유구",
    "여송",
    "홍어"
]

COMMON_MISSPELLS = {
    "방물관": "박물관",
    "빵물관": "박물관",
    "팍물관": "박물관",
    "문수득": "문순득",
    "뭉슨득": "문순득",
    "문순덕": "문순득",
    "문숭득": "문순득",
    "류뀨":"류큐",
    "오끼나와":"오키나와",
}

def normalize_stt_text(text: str) -> str:
    if not text:
        return text

    for wrong, right in COMMON_MISSPELLS.items():
        if wrong in text:
            text = text.replace(wrong, right)

    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
    unique_tokens = set(tokens)

    for tok in unique_tokens:
        if len(tok) < 2:
            continue

        best_term = None
        best_score = 0.0

        for term in DOMAIN_TERMS:
            if abs(len(tok) - len(term)) > 3:
                continue

            score = SequenceMatcher(None, tok, term).ratio()
            if score > best_score:
                best_term, best_score = term, score

        if best_term and best_score >= 0.8 and tok != best_term:
            pattern = rf"\b{re.escape(tok)}\b"
            text = re.sub(pattern, best_term, text)

    return text

REASK_TEMPLATES = [
    "죄송하오, 주변이 조금 시끄러워 잘 들리지 않았소. 다시 한 번 또렷하게 말씀해 주시겠소?",
    "방금 말씀을 정확히 듣지 못했소. 한 번만 더 말씀해 주시겠소?",
    "죄송하오, 내용을 잘 알아듣지 못했소. 천천히 다시 말씀해 주시면 안내해 드리겠소.",
]

REPEAT_PATTERNS = [
    "다시 말해줘",
    "다시 한번 말해줘",
    "다시 한 번 말해줘",
    "다시 얘기해줘",
    "다시 이야기해줘",
    "다시 들려줘",
    "다시 들려 주라",
    "방금 뭐라고 했어",
    "방금 뭐라 했어",
    "방금 뭐라 그랬어",
    "repeat",
    "say again",
    "can you repeat",
    '다시 설명해줘',
    '한번 더 말해줘',
    '반복해줘',
    '다시 알려줘',
    '다시 이야기해줘',
    '다시 말해봐',
    '다시 말해줄래',
    '재설명해줘',
    '한 번만 더',
    '다시 정리해줘',
    '다시 표현해줘',
    '다시 얘기해줘',
    '리마인드 해줘',
    '요약해줘',
    '정리해서 말해줘',
    '다시 말해',
    '말 다시 해줘',
    '다시 말해바',
    '다시 말해봐봐',
    '다시 얘기해봐',
    '다시 알려봐',
    '다시 알려바',
    '다시 해줘',
    '다시말해조',
    '다시말해쥬',
    '다시말해죠',
    '다시말해쫭',
    '다시말해쥬세여',
    '다시말해줘염',
    '다시말해죠요',
    '다시말해죵',
    '다시말해죠앙',
    '다시말해죠오',
    'repeat please',
    'say again',
    'can you say again',
    'say once more',
    'repeat that'
]

FILLER_TOKENS = {"음", "어", "에", "아", "어어", "음음", "어음", "흠"}
LAUGH_RE = re.compile(r"^[ㅋㅎ]+$")

# 직전 답변 (키오스크 1대 기준 글로벌로 관리)
LAST_ANSWER: Dict[str, Any] = {"text": "", "lang": "ko"}

def is_repeat_request(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in REPEAT_PATTERNS)

def is_unclear_input(text: str) -> bool:
    if not text or not text.strip():
        return True

    t = text.strip()

    if len(t) <= 1:
        return True

    if LAUGH_RE.fullmatch(t):
        return True

    words = t.split()
    if len(words) <= 3 and all(w in FILLER_TOKENS for w in words):
        return True

    valid_chars = re.findall(r"[가-힣A-Za-z0-9]", t)
    if len(valid_chars) / max(len(t), 1) < 0.3:
        return True

    return False

def _label_to_emotion_id(label: str) -> str:
    key = label.lower().strip()
    if key in EMOTION_ID_MAP:
        return str(EMOTION_ID_MAP[key])
    for k in EMOTION_ID_MAP:
        if k in key:
            return str(EMOTION_ID_MAP[k])
    return str(EMOTION_ID_MAP["neutral"])


def check_emotion(text: str) -> str:
    if not text:
        return str(EMOTION_ID_MAP["neutral"])
    try:
        res = action_classifier(text, EMOTION_LIST)        
        if isinstance(res, dict):
            label = res["labels"][0]
        else:
            label = res[0]["labels"][0]
        return _label_to_emotion_id(label)
    except Exception as e:
        print(f"[check_emotion 에러] {e}")
        return str(EMOTION_ID_MAP["neutral"])


def check_action(text: str) -> str:
    if not text:
        return str(ACTION_ID_MAP["Idle"])

    try:
        greetings = ("안녕", "안녕하세요", "안녕하시오", "하이", "헬로", "헬로우", "헤이", "안뇽")
        if any(g in text for g in greetings):
            return str(ACTION_ID_MAP["Greeting"])

        res = action_classifier(text, ACTION_LIST)
        if isinstance(res, dict):
            label = res["labels"][0]
        else:
            label = res[0]["labels"][0]

        return str(ACTION_ID_MAP.get(label, ACTION_ID_MAP["Idle"]))
    except Exception as e:
        print(f"[check_action 에러] {e}")
        return str(ACTION_ID_MAP["Idle"])

def safe_filename_from_text(text: str, max_len: int = 40) -> str:
    base = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text or "")
    base = re.sub(r"\s+", "_", base).strip("_")

    if not base:
        base = "stt_audio"

    if len(base) > max_len:
        base = base[:max_len]

    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{base}.wav"


def save_input_audio(src_path: str, text: str) -> str:
    try:
        filename = safe_filename_from_text(text)
        dst_path = os.path.join(INPUT_AUDIO_DIR, filename)
        shutil.copy(src_path, dst_path)
        print(f"[STT SAVE] {dst_path}")
        return dst_path
    except Exception as e:
        print(f"[STT SAVE ERROR] {e}")
        return ""

def save_upload_to_tmp(upload_file: UploadFile) -> str:
    suffix = pathlib.Path(upload_file.filename or "").suffix or ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload_file.file.read())
    return tmp_path

def denoise_file(in_path: str) -> str:
    y, sr = librosa.load(in_path, sr=16000, mono=True)

    noise_len = min(int(sr * 0.3), len(y))
    noise_clip = y[:noise_len] if noise_len > 0 else y

    reduced = nr.reduce_noise(
        y=y,
        y_noise=noise_clip,
        sr=sr,
        prop_decrease=0.7,
    )

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(out_path, reduced, sr)
    return out_path





from typing import Dict, Tuple
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# MADLAD-400 3B MT 로컬 경로 (huggingface-cli로 받아둔 폴더)
MADLAD_DIR = "/home/user/hayang/main/models/madlad400-3b-mt"

# MADLAD 타겟 언어 태그
LANG_TAG = {"en": "<2en>", "ko": "<2ko>"}

# 캐시: 경로별로 토크나이저/모델 1회만 로드
# value: (tokenizer, model, device)
_loaded_translation_models: Dict[str, Tuple[T5Tokenizer, T5ForConditionalGeneration, str]] = {}


def _get_madlad() -> Tuple[T5Tokenizer, T5ForConditionalGeneration, str]:
    """
    MADLAD-400 3B MT 로컬 모델을 한 번만 로드해서 캐시.
    """
    if MADLAD_DIR not in _loaded_translation_models:
        device = "cuda"

        tok = T5Tokenizer.from_pretrained(MADLAD_DIR, local_files_only=True)

        # GPU면 bfloat16으로 살짝 메모리 아끼기 (원하면 None으로 바꿔도 됨)
        torch_dtype = torch.bfloat16 if device == "cuda" else None

        mdl = T5ForConditionalGeneration.from_pretrained(
            MADLAD_DIR,
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(device)
        mdl.eval()

        _loaded_translation_models[MADLAD_DIR] = (tok, mdl, device)

    return _loaded_translation_models[MADLAD_DIR]


def _translate_madlad(text: str, src: str, tgt: str) -> str:
    """
    MADLAD-400 사용 en <-> ko 번역.
    src/tgt는 "en", "ko"만 지원한다고 가정.
    """
    src = src.lower()
    tgt = tgt.lower()

    if tgt not in LANG_TAG:
        # 지원 안 하는 타겟이면 그냥 원문 반환
        return text

    tok, mdl, device = _get_madlad()

    # MADLAD는 "목표 언어" 태그를 앞에 붙이는 방식
    tagged = f"{LANG_TAG[tgt]} {text}"
    inputs = tok(tagged, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,  # 필요하면 빔 서치 늘려도 됨
        )

    return tok.decode(out[0], skip_special_tokens=True)


def local_translate(text: str, src: str = "ko", tgt: str = "en") -> str:
    """
    - en ↔ ko 모두 /home/user/hayang/main/models/madlad400-3b-mt (MADLAD-400 3B MT) 하나로 처리
    - 완전 로컬 (local_files_only=True)
    """
    text = (text or "").strip()
    if not text:
        return text

    src = src.lower()
    tgt = tgt.lower()

    if src == tgt:
        return text

    try:
        return _translate_madlad(text, src, tgt)
    except Exception as e:
        print(f"[local_translate ERROR] {e} — fallback to identity")
        return text
# =====================================









COMMON_INITIAL_PROMPT = "국립인천해양박물관, 해양박물관, 문순득, 전시관, 인천항"

def transcribe_file(path: str) -> Tuple[str, str]:
    segments_ko, _ = stt_model.transcribe(
        path,
        task="transcribe",
        language=None,
        beam_size=1,
        temperature=0.0,
        vad_filter=True,
        no_speech_threshold=0.2,
        condition_on_previous_text=False,
        word_timestamps=False,
    )

    text_ko = "".join(seg.text for seg in segments_ko).strip()
    if not text_ko:
        return "", "ko"

    print(f"[STT-KO] {text_ko}")

    ko_chars = len(re.findall(r"[가-힣]", text_ko))
    en_chars = len(re.findall(r"[A-Za-z]", text_ko))
    total = max(len(text_ko), 1)
    ko_ratio = ko_chars / total
    en_ratio = en_chars / total

    DOMAIN_HINTS = ["문순득", "박물관", "전시관", "인천", "유구", "류큐", "오키나와"]
    if any(h in text_ko for h in DOMAIN_HINTS):
        return text_ko, "ko"

    if ko_chars >= 2 and ko_ratio >= 0.1 and en_ratio < 0.7:
        return text_ko, "ko"

    if en_ratio > 0.7 and ko_chars == 0:
        print("[STT] 영어 가능성 높음, en으로 재시도")
        segments_en, _ = stt_model.transcribe(
            path,
            task="transcribe",
            language="en",
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
            no_speech_threshold=0.2,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        text_en = "".join(seg.text for seg in segments_en).strip()
        print(f"[STT-EN] {text_en}")

        en2 = len(re.findall(r"[A-Za-z]", text_en))
        ko2 = len(re.findall(r"[가-힣]", text_en))
        total2 = max(len(text_en), 1)
        en_ratio2 = en2 / total2 if total2 else 0.0

        # 진짜 영어 문장 같으면 en으로 확정
        if en2 >= 3 and en_ratio2 > 0.7 and ko2 == 0:
            return text_en, "en"

    # 4) 그 외 애매한 건 전부 ko로 본다 (영어 질문 소수라고 가정)
    return text_ko, "ko"

def is_repeated_stt(text: str) -> bool:
    if not text:
        return False

    t = text.strip().lower()

    # -----------------------
    # 1) 동일 문자 반복 (아아아아, ㅎㅎㅎㅎ, aaaaa)
    # -----------------------
    # 같은 문자 4번 이상 연속
    if re.search(r"(.)\1{3,}", t):
        return True

    # -----------------------
    # 2) 동일 단어 반복 (hello hello hello)
    # -----------------------
    words = re.findall(r"[가-힣a-zA-Z]+", t)
    if len(words) >= 3:
        unique = set(words)
        # 단어 종류는 적고 반복 횟수만 많은 경우
        if len(unique) <= 2 and len(words) / max(len(unique), 1) >= 3:
            return True

    # -----------------------
    # 3) 붙어 있는 반복 (hellohellohello)
    # -----------------------
    collapsed = re.sub(r"\s+", "", t)
    for w in set(words):
        if len(w) >= 2 and collapsed.count(w) >= 3:
            return True


    return False

# =====================================
# Warmup
# =====================================
@app.on_event("startup")
async def warmup():
    print("warmup 시작...")

    # LLM warmup
    try:
        text = "테스트 문장입니다."
        meta = {"user": "해양박물관 챗봇", "lang": "ko"}
        _ = await gateway.send_to_llm(text, meta)
        print("LLM warmup 완료")
    except Exception as e:
        print(f"LLM warmup 실패: {e}")

    # TTS warmup
    try:
        sentence = "테스트 문장입니다."
        meta = {"user": "문순득", "lang": "ko"}
        emotion_id = check_emotion(sentence)
        action_id = check_action(sentence)
        _ = await gateway.send_to_tts(sentence, meta, emotion_id, action_id, "ko")
        print("TTS warmup 완료")
    except Exception as e:
        print(f"TTS warmup 실패: {e}")

@app.post("/server")
async def start_chat(
    file: UploadFile = File(None),
    question_text: Optional[str] = Form(None),
):
    """
    - file: 음성 업로드 시 STT + LLM + TTS
    - question_text: 텍스트 바로 질의 (STT 생략)
    """
    global LAST_ANSWER
    global qa_file_handler
    start_time = time.time()
    text = ""
    lang = "ko" 
    language_style = "ko" 
    ts_stt_start = time.time()
    if file is not None:
        tmp_path = await to_thread.run_sync(save_upload_to_tmp, file)
        clean_path = None
        try:
            raw_text, lang = await to_thread.run_sync(transcribe_file, tmp_path)

            text = normalize_stt_text(raw_text)

            await to_thread.run_sync(
                save_input_audio,
                tmp_path,
                text or raw_text or "stt_audio",
            )

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if clean_path and os.path.exists(clean_path):
                os.remove(clean_path)

    elif question_text is not None and question_text.strip():
        text = question_text.strip()
        contains_korean = bool(re.search(r"[가-힣]", text))
        lang = "ko" if contains_korean else "en"

    else:
        raise HTTPException(
            status_code=400, detail="음성 파일(file) 또는 question_text 중 하나는 필요합니다."
        )
    
    text = normalize_stt_text(text)
    ts_stt_end = time.time()
    dur_stt = ts_stt_end - ts_stt_start
    print(f"[PERF] STT: {dur_stt:.3f} sec")
    print(f"[STT 결과] ({lang}) {text}")

    if is_repeated_stt(text):
        msg = REASK_TEMPLATES[int(time.time()) % len(REASK_TEMPLATES)]
        emotion_id = check_emotion(msg)
        action_id = check_action(msg)
        meta_for_tts = {"user": "문순득", "lang": "ko"}

        tts_response = await gateway.send_to_tts(
            msg, meta_for_tts, emotion_id, action_id, "ko"
        )

        audio_uri = ""
        if getattr(tts_response, "is_success", False):
            tts_data = tts_response.json()
            audio_uri = (
                tts_data.get("audio_file")
                or tts_data.get("file")
                or tts_data.get("url")
                or ""
            )

        return JSONResponse(
            {
                "user": "문순득",
                "input": text,
                "response": msg,
                "region_index": "-1",
                "scriptList": [
                    {"action": [emotion_id, action_id], "text": msg}
                ],
                "audioURIList": [audio_uri] if audio_uri else [],
                "combinedAudioPath": None,
            }
        )

    if is_repeat_request(text):
        if LAST_ANSWER.get("text"):
            repeat_text = LAST_ANSWER["text"]
            repeat_lang = LAST_ANSWER.get("lang", "ko")
            emotion_id = check_emotion(repeat_text)
            action_id = check_action(repeat_text)
            meta_for_tts = {"user": "문순득", "lang": repeat_lang}

            tts_response = await gateway.send_to_tts(
                repeat_text, meta_for_tts, emotion_id, action_id, repeat_lang
            )
            audio_uri = ""
            if getattr(tts_response, "is_success", False):
                tts_data = tts_response.json()
                audio_uri = (
                    tts_data.get("audio_file")
                    or tts_data.get("file")
                    or tts_data.get("url")
                    or ""
                )

            return JSONResponse(
                {
                    "user": "문순득",
                    "input": text,
                    "response": repeat_text,
                    "region_index": -1,
                    "scriptList": [
                        {"action": [emotion_id, action_id], "text": repeat_text}
                    ],
                    "audioURIList": [audio_uri] if audio_uri else [],
                    "combinedAudioPath": None,
                }
            )
        else:
            msg = (
                "아직 드린 답변이 없어 다시 들려드릴 내용이 없소. "
                "궁금하신 내용을 다시 말씀해 주시겠소?"
            )
            emotion_id = check_emotion(msg)
            action_id = check_action(msg)
            meta_for_tts = {"user": "문순득", "lang": "ko"}
            tts_response = await gateway.send_to_tts(
                msg, meta_for_tts, emotion_id, action_id, "ko"
            )
            audio_uri = ""
            if getattr(tts_response, "is_success", False):
                tts_data = tts_response.json()
                audio_uri = (
                    tts_data.get("audio_file")
                    or tts_data.get("file")
                    or tts_data.get("url")
                    or ""
                )
            return JSONResponse(
                {
                    "user": "문순득",
                    "input": text,
                    "response": msg,
                    "region_index": -1,
                    "scriptList": [
                        {"action": [emotion_id, action_id], "text": msg}
                    ],
                    "audioURIList": [audio_uri] if audio_uri else [],
                    "combinedAudioPath": None,
                }
            )

    if is_unclear_input(text):
        msg = REASK_TEMPLATES[int(time.time()) % len(REASK_TEMPLATES)]
        emotion_id = check_emotion(msg)
        action_id = check_action(msg)
        meta_for_tts = {"user": "문순득", "lang": "ko"}
        tts_response = await gateway.send_to_tts(
            msg, meta_for_tts, emotion_id, action_id, "ko"
        )
        audio_uri = ""
        if getattr(tts_response, "is_success", False):
            tts_data = tts_response.json()
            audio_uri = (
                tts_data.get("audio_file")
                or tts_data.get("file")
                or tts_data.get("url")
                or ""
            )

        return JSONResponse(
            {
                "user": "문순득",
                "input": text,
                "response": msg,
                "region_index": -1,
                "scriptList": [
                    {"action": [emotion_id, action_id], "text": msg}
                ],
                "audioURIList": [audio_uri] if audio_uri else [],
                "combinedAudioPath": None,
            }
        )








    print(f"language:{lang}")

    ts_llm_start = time.time()
    if lang == "en":
        language_style = "en"
        query_for_llm = local_translate(text, src="en", tgt="ko")
    else:
        language_style = "ko"
        query_for_llm = text








    meta_for_llm = {"user": "해양박물관 챗봇", "lang": "ko"}
    llm_response = await gateway.send_to_llm(query_for_llm, meta_for_llm)

    if not getattr(llm_response, "is_success", False):
        raise HTTPException(
            status_code=getattr(llm_response, "status_code", 500),
            detail="LLM 서버 에러 발생",
        )

    llm_data = llm_response.json()
    llm_ko_text = llm_data.get("response", "").strip()

    del llm_response
    gc.collect()
    torch.cuda.empty_cache()
    

    if not llm_ko_text:
        raise HTTPException(status_code=500, detail="LLM 응답이 비어 있습니다.")
    
    if language_style == "en":
        speak_text = local_translate(llm_ko_text, src="ko", tgt="en")
    else:
        speak_text = llm_ko_text

    LAST_ANSWER = {"text": speak_text, "lang": language_style}

    ts_llm_end = time.time()
    dur_llm = ts_llm_end - ts_llm_start
    print(f"[PERF] LLM: {dur_llm:.3f} sec")

    ts_tts_start = time.time()

    emotion_id = check_emotion(speak_text)
    action_id = check_action(speak_text)

    meta_for_tts = {"user": "문순득", "lang": language_style}
    tts_response = await gateway.send_to_tts(
        speak_text, meta_for_tts, emotion_id, action_id, language_style
    )

    ts_tts_end = time.time()
    dur_tts = ts_tts_end - ts_tts_start
    print(f"[PERF] TTS: {dur_tts:.3f} sec")
    if not getattr(tts_response, "is_success", False):
        raise HTTPException(
            status_code=getattr(tts_response, "status_code", 500),
            detail="TTS 서버 에러 발생",
        )

    tts_data = tts_response.json()
    audio_uri = (
        tts_data.get("audio_file")
        or tts_data.get("file")
        or tts_data.get("url")
        or ""
    )

    qa_logger.info(f"Q({lang}): {text}\nA({language_style}): {speak_text}")

    # -------------------------
    # 오디오 디렉토리 정리 (100MB)
    # -------------------------
    cleanup_directory(AUDIO_OUTPUT_DIR, 100 * 1024 * 1024)

    # -------------------------
    # qa_list.log 파일 rotate (30MB)
    # -------------------------
    new_handler = rotate_qa_log(qa_logger, qa_file_handler, 30 * 1024 * 1024)
    if new_handler is not None:
        qa_file_handler = new_handler

    print(f"total elapsed time : {time.time() - start_time:.3f}s")

    return JSONResponse(
        {
            "user": llm_data.get("user", "unknown"),
            "input": text,
            "response": speak_text,
            "region_index": llm_data.get("region_index", "-1"),
            "scriptList": [
                {"action": [emotion_id, action_id], "text": speak_text}
            ],
            "audioURIList": [audio_uri] if audio_uri else [],
            "combinedAudioPath": None,
        }
    )
