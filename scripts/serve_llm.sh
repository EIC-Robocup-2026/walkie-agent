#!/bin/bash
# scripts/serve_llm.sh

MODE=${1:-vllm}
echo "🧠 Starting Model Host with Tool Support: Qwen/Qwen3.5-9B"

if [ "$MODE" == "vllm" ]; then
    # เพิ่ม --enable-auto-tool-choice และ --tool-call-parser ให้กับ vLLM
    # สำหรับ Qwen แนะนำให้ใช้ parser 'hermes' หรือ 'tool_calling' (ขึ้นอยู่กับเวอร์ชัน vLLM)
    uv run python -m vllm.entrypoints.openai.api_server \
        --model "Qwen/Qwen3.5-9B" \
        --served-model-name "qwen3.5-9b" \
        --port 8000 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.7 \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code

elif [ "$MODE" == "ollama" ]; then
    export OLLAMA_HOST="0.0.0.0:8000"
    ollama run qwen3.5:9b
fi
