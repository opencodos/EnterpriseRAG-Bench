#!/usr/bin/env bash
# Boot script for the baseline host: NVIDIA driver, Docker, vLLM, OpenSearch.
#
# Runs as the instance's startup-script, so it must be idempotent — a Spot instance is
# preempted and restarted, and every restart re-runs this from the top. Each step
# therefore checks for its own result before doing anything.
#
# The reader and the embedding model are served by two vLLM containers. The reader takes
# both GPUs with tensor parallelism, because a 27B at bf16 is ~54GB of weights and no
# single 40GB card holds it; the embedding model is 0.6B and takes a sliver of what the
# reader leaves. Neither reproduced arm queries the embedding model — BM25 is lexical and
# the File-System Agent greps — but it is part of the reader stack the study serves, so
# the host that stands in for that stack serves it too.
set -euo pipefail

READER_MODEL="${READER_MODEL:-Qwen/Qwen3.6-27B}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
# The agent conversation grows to ~200k characters before the harness prunes it, which
# is ~50k tokens, and compaction is reactive — it fires when the server rejects the
# request. A window materially below this makes the agent compact constantly and stops
# measuring the study's agent.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# What the reader may take, leaving room for the embedding server beside it.
READER_GPU_FRACTION="${READER_GPU_FRACTION:-0.86}"
# What the embedding model may take of card 0, out of the ~14% the reader leaves. Its
# weights are only ~1.2GB, but vLLM still sizes a KV cache from this fraction and
# refuses to start when what remains is too small to hold one.
EMBEDDER_GPU_FRACTION="${EMBEDDER_GPU_FRACTION:-0.08}"
# The embedding model's own context window. Its default is 32k, and vLLM refuses to
# start unless the KV cache can hold one full-length sequence -- which the sliver left
# beside the reader cannot. Embeddings are taken over chunks far shorter than this.
EMBEDDER_MAX_MODEL_LEN="${EMBEDDER_MAX_MODEL_LEN:-8192}"
# Pinned, not :latest. The four rungs are measured over days and the ladder claims to be
# re-measurable later; a serving stack that changes underneath would put a version
# difference into a curve read as a property of corpus size. This is also the version the
# tool-call parser name below was chosen against -- the valid set moves between releases.
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.28.0}"
# How the served model writes a tool call, which is a property of its chat template and
# not a preference. This model's template emits XML -- <function=name><parameter=x> --
# and not the JSON-inside-<tool_call> that the `hermes` parser expects, so `hermes`
# yields no tool calls at all and the agent arm explores nothing. The agent arm's
# preflight probes this before a run rather than discovering it 500 questions later.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
# This model is a hybrid Mamba/attention architecture, and every decode sequence needs
# its own Mamba cache block. vLLM's default max_num_seqs of 256 exceeds the blocks that
# fit at this memory budget (242 measured here), and it refuses to capture CUDA graphs
# rather than degrading -- so the engine does not start at all. The arms ask one
# question at a time, so this is far more concurrency than they use; it is set only to
# stay under the cache the model's hybrid layers can hold.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
HF_CACHE="/opt/models"

log() { echo "[startup $(date -u +%H:%M:%S)] $*"; }

# --- NVIDIA driver ---------------------------------------------------------
# Not installed here, and deliberately. Building the driver from NVIDIA's bundled
# installer fails against recent GCP kernels -- 6.17.0-1022-gcp cannot compile it
# ("os-interface.h: No such file or directory") -- and a box whose driver builds or
# does not depending on which kernel the image last shipped is not a box a ladder can
# be measured on twice. The image carries the driver instead; this only checks it.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "no nvidia-smi: this image does not carry a driver. Use a Deep Learning VM"
  log "image (deeplearning-platform-release) rather than a stock Ubuntu one."
  exit 1
fi
nvidia-smi || { log "driver present but no GPU visible; refusing to start"; exit 1; }
log "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPU(s) visible"

# --- Docker + NVIDIA container toolkit -------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "installing Docker"
  curl -fsSL https://get.docker.com | sh
fi
# Installing the toolkit and wiring it into Docker are separate steps, because on a
# Deep Learning VM image the first is already done and the second is not: the image
# ships nvidia-container-toolkit, and installing Docker over it above leaves a daemon
# that does not know about the runtime. Running the install anyway fails -- gpg will
# not overwrite the keyring the image already placed -- so it is conditioned on the
# toolkit actually being absent rather than on Docker not yet knowing about it.
if ! docker info 2>/dev/null | grep -q nvidia; then
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    log "installing NVIDIA container toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
  fi
  log "wiring the NVIDIA runtime into Docker"
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

mkdir -p "${HF_CACHE}"

# --- The reader ------------------------------------------------------------
# Temperature, top-p and thinking are set per request by the client, not here: the
# study's settings belong with the arms that claim them, so a reader restarted by hand
# cannot silently serve a different sampler than the one the results name.
if ! docker ps --format '{{.Names}}' | grep -qx reader; then
  log "starting reader ${READER_MODEL} (tp=${TENSOR_PARALLEL}, len=${MAX_MODEL_LEN})"
  docker rm -f reader >/dev/null 2>&1 || true
  docker run -d --name reader --restart unless-stopped \
    --gpus all --ipc=host -p 8000:8000 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "${VLLM_IMAGE}" \
    --model "${READER_MODEL}" \
    --served-model-name "${READER_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${READER_GPU_FRACTION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}" \
    --port 8000
fi

# --- The embedding model ---------------------------------------------------
if ! docker ps --format '{{.Names}}' | grep -qx embedder; then
  log "starting embedder ${EMBEDDING_MODEL}"
  docker rm -f embedder >/dev/null 2>&1 || true
  docker run -d --name embedder --restart unless-stopped \
    --gpus '"device=0"' --ipc=host -p 8001:8001 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "${VLLM_IMAGE}" \
    --model "${EMBEDDING_MODEL}" \
    --served-model-name "${EMBEDDING_MODEL}" \
    --runner pooling \
    --max-model-len "${EMBEDDER_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${EMBEDDER_GPU_FRACTION}" \
    --port 8001
fi

# --- OpenSearch, for the BM25 arm's chunk index ----------------------------
if ! docker ps --format '{{.Names}}' | grep -qx opensearch; then
  log "starting OpenSearch"
  docker rm -f opensearch >/dev/null 2>&1 || true
  sysctl -w vm.max_map_count=262144
  docker run -d --name opensearch --restart unless-stopped \
    -p 9200:9200 -e discovery.type=single-node \
    -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true \
    -e "OPENSEARCH_JAVA_OPTS=-Xms8g -Xmx8g" \
    -v opensearch-data:/usr/share/opensearch/data \
    opensearchproject/opensearch:2.18.0
fi

log "done; the reader needs several minutes to pull and load weights"
