#!/bin/bash
set -e
set -o pipefail

# ────────── Step 0: 가상환경 초기화 ──────────
echo "🔹 Step 0: 가상환경 초기화..."
deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true


# ────────── Step 0.1: 기존 서버 종료 ──────────
echo "🔹 Step 0.1: 기존 서버 종료..."
pkill -f infer-web.py 2>/dev/null || true
pkill -f tts_server_ref 2>/dev/null || true
pkill -f llamacpp_ver_qa 2>/dev/null || true
pkill -f server:app 2>/dev/null || true


# ────────── Step 1: Conda 초기화 ──────────
echo "🔹 Step 1: Conda 초기화..."
eval "$(conda shell.bash hook)"

# ────────── Step 2: RVC 서버 실행 ──────────
RVC_PORT=7865
echo "🔹 Step 2: RVC 서버 실행 (포트 $RVC_PORT)..."
if lsof -i:$RVC_PORT >/dev/null; then
    echo "⚠ RVC 포트 $RVC_PORT 사용 중, 프로세스 종료"
    lsof -t -i:$RVC_PORT | xargs kill -9 || true
fi
conda activate rvc
cd /home/user/hayang/rvc/Retrieval-based-Voice-Conversion-WebUI
python infer-web.py --port $RVC_PORT &
RVC_PID=$!
cd -
# RVC 준비 대기
until nc -z localhost $RVC_PORT; do
    echo "RVC 서버 준비 대기..."
    sleep 1
done

export HF_HUB_OFFLINE=1
# ────────── Step 3: TTS 서버 실행 ──────────
TTS_PORT=8003
echo "🔹 Step 3: TTS 서버 실행 (포트 $TTS_PORT)..."
if lsof -i:$TTS_PORT >/dev/null; then
    echo "⚠ TTS 포트 $TTS_PORT 사용 중, 프로세스 종료"
    lsof -t -i:$TTS_PORT | xargs kill -9 || true
fi
conda activate melo2
cd /home/user/hayang/tts
redis-server --daemonize yes
python -m uvicorn tts_server_ref:app --host 0.0.0.0 --port $TTS_PORT &
TTS_PID=$!
cd -
# TTS 준비 대기
until nc -z localhost $TTS_PORT; do
    echo "TTS 서버 준비 대기..."
    sleep 1
done

# ────────── Step 4: LLM 서버 실행 ──────────
LLM_PORT=8004
echo "🔹 Step 4: LLM 서버 실행 (venv)..."
# source /home/user/hayang/munja_3d/bin/activate
conda activate llama_env

# export CUDA_VISIBLE_DEVICES=1

cd /home/user/hayang/llm
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python -m uvicorn llamacpp_ver_qa:app --host 0.0.0.0 --port $LLM_PORT &
LLM_PID=$!
cd -
# LLM 준비 대기
until nc -z localhost $LLM_PORT; do
    echo "LLM 서버 준비 대기..."
    sleep 1
done

# ────────── Step 5: Main(STT) 서버 실행 ──────────
MAIN_PORT=8002
echo "🔹 Step 5: Main(STT) 서버 실행 (venv)..."
cd /home/user/hayang/main
python -m uvicorn server:app --host 0.0.0.0 --port $MAIN_PORT &
MAIN_PID=$!
cd -

echo "✅ 모든 서버가 실행되었습니다! warmup start"
echo "RVC PID: $RVC_PID, TTS PID: $TTS_PID, LLM PID: $LLM_PID, MAIN PID: $MAIN_PID"

# ────────── 스크립트 종료 시 서버 정리 ──────────
cleanup() {
    echo "🔹 서버 종료 중..."
    kill $RVC_PID $TTS_PID $LLM_PID $MAIN_PID 2>/dev/null || true

    echo "✅ 종료 완료"
}
trap cleanup EXIT

# 서버 유지
wait






# #!/bin/bash
# set -o pipefail

# echo "🔁 FastAPI + Celery 환경 Reload 시작..."

# echo "🔹 Step 1: 기존 가상환경 비활성화..."
# deactivate 2>/dev/null || true
# conda deactivate 2>/dev/null || true

# echo "🔹 Step 2: web 디렉토리로 이동 후 가상환경 활성화..."
# cd "$(dirname "$0")/web" || exit 1
# source .venv/bin/activate

# echo "🔹 Step 3: 기존 Celery 프로세스 종료..."
# sudo pkill -f 'celery' 2>/dev/null || true

# echo "🔹 Step 4: Redis 컨테이너 확인/실행..."
# REDIS_CONTAINER=$(sudo docker ps -q -f name=redis-broker)
# if [ -n "$REDIS_CONTAINER" ]; then
#     echo "🧹 기존 Redis 컨테이너 발견, 재시작..."
#     sudo docker restart redis-broker
# else
#     echo "🚀 Redis 컨테이너 실행..."
#     sudo docker run -d --name redis-broker -p 6379:6379 -v redis-data:/data --restart unless-stopped redis:7-alpine
# fi

# echo "⏳ Redis 응답 대기 중..."
# for i in {1..15}; do
#     if redis-cli -h 127.0.0.1 -p 6379 ping | grep -q PONG; then
#         echo "✅ Redis 연결 성공"
#         break
#     fi
#     echo "⏳ Redis 준비 중... ($i/15)"
#     sleep 1
# done

# echo "🔹 Step 5: Celery 워커 실행 (백그라운드)..."
# uv run celery -A app.tasks:app worker --loglevel=INFO -P solo --concurrency=1 > ../celery.log 2>&1 &
# CELERY_PID=$!
# echo "✅ Celery 워커 실행됨 (PID: $CELERY_PID, 로그: ../celery.log)"

# echo "🔹 Step 6: FastAPI 서버 Reload..."
# # 기존 FastAPI 서버 종료
# sudo pkill -f 'uvicorn app.main:app' 2>/dev/null || true
# sleep 1

# # --reload 옵션으로 자동재시작 기능 활성화
# uv run -- uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &

# FASTAPI_PID=$!
# echo "✅ FastAPI 서버 재시작됨 (PID: $FASTAPI_PID, 포트: 8000)"

# echo "✨ 모든 서비스가 재시작되었습니다."
