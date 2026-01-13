# vLLM Inference Lab

A learning-focused exploration of **LLM inference systems** using **vLLM**, with an emphasis on
**concurrency, KV cache behavior, batching, streaming, and real-world GPU constraints**.

This repository does **not** aim to replicate ChatGPT or build a production chatbot.
Instead, it focuses on understanding **how large language models are actually served and optimized**
in modern inference engines.

---

## Motivation

Most LLM projects focus on:
- Prompting
- Model selection
- UI features

This project focuses on what happens **after** a prompt is sent:

- How requests are scheduled
- How KV cache affects memory and throughput
- Why latency does not scale linearly
- How inference engines behave under GPU constraints
- Why stability often matters more than peak throughput

The goal is to develop **systems-level intuition** for LLM inference.

---

## What This Project Covers

### Core Topics
- Local GPU-backed LLM serving using **vLLM**
- OpenAI-compatible inference API
- Token-level **streaming responses**
- **Concurrent inference benchmarking**
- KV cache behavior and batching trade-offs
- Latency, throughput, and time-to-first-token (TTFT)
- Prompt prefill vs decode cost analysis
- Inference behavior on **memory-constrained GPUs**

### Supporting Components
- Lightweight Streamlit UI for streaming inference
- RAG (PDF-based) pipeline for document-grounded generation
- Clean **client / server environment separation**
- Dependency isolation to avoid Pydantic/OpenAI conflicts

---

## Architecture Overview

This project follows a **realistic inference architecture**:

Client / UI / Benchmarks
↓
OpenAI-compatible HTTP API
↓
vLLM Inference Server
↓
GPU (KV Cache, Scheduler, Decode Engine)
### Key Design Decisions
- vLLM runs in a **dedicated server environment**
- Clients never import vLLM directly
- Communication happens purely over HTTP
- GPU memory and KV cache are owned exclusively by the server

This mirrors real-world deployment patterns used in production inference stacks.

---

## Hardware & Model Configuration

- **GPU:** RTX 4050 Laptop (6GB VRAM)
- **Model:** Qwen2-1.5B-Instruct
- **Precision:** FP16
- **OS:** Linux (WSL2)
- **Inference Engine:** vLLM

The hardware choice is intentional:  
**small GPUs expose inference bottlenecks clearly**, making trade-offs easier to observe.

---

## Benchmarking Methodology

### Concurrency Test
- Multiple identical prompts sent concurrently
- Fixed output length
- Requests issued using a thread pool
- Metrics captured:
  - Per-request latency
  - Total wall time
  - Average latency

### Example Result (4 concurrent requests)

Request completed in ~18–23s
Total wall time: ~23s
Average latency: ~19s

---

## Observations & Analysis

### Key Findings

- Requests **overlap execution**, confirming concurrent scheduling
- Latency does **not scale linearly** with concurrency on small GPUs
- **Prompt prefill dominates latency** for short outputs
- KV cache pressure limits aggressive batching on 6GB VRAM
- vLLM prioritizes **stability and memory safety** over peak throughput
- Throughput improves under concurrency while avoiding OOM

These behaviors are expected and reflect **production-grade inference design**.

---

## Screenshots

### Concurrent Inference Benchmark
![Concurrency benchmark](screenshots/concurrency_benchmark.png)

### vLLM Server Logs (KV Cache & Scheduling)
![vLLM server logs](screenshots/vllm_server_logs.png)

---

## Project Structure

vllm-inference-lab/
│
├── concurrency_test.py # Core concurrency benchmark
├── vllm_ui.py # Streaming UI (optional)
├── rag_chatbot.py # RAG CLI (optional)
│
├── screenshots/ # Proof & benchmarks
│ ├── concurrency_benchmark.png
│ └── vllm_server_logs.png
│
├── requirements-server.txt # vLLM server dependencies
├── requirements-client.txt # Client / UI dependencies
└── README.md

---

## Environment Separation

This project intentionally uses **two isolated environments**:

### vLLM Server Environment (`vllm-env`)
- Runs GPU-backed inference
- Owns KV cache and scheduling
- Uses modern Pydantic / vLLM dependencies

### Client Environment (`client-env`)
- Runs UI, RAG, and benchmarks
- Communicates via HTTP only
- Uses a stable OpenAI client for concurrency testing

This separation avoids dependency conflicts and mirrors production practice.

---

## How to Run (High-Level)

1. Start the vLLM server in `vllm-env`
2. Run benchmarks or UI from `client-env`
3. Observe:
   - Server logs (KV cache usage, throughput)
   - Client-side latency and wall time

Detailed setup instructions are intentionally omitted to keep the focus on **concepts and behavior**, not boilerplate.

---

## Key Learnings

- LLM inference performance is often **memory-bound**, not compute-bound
- KV cache size directly affects batching efficiency
- Throughput optimization depends on **workload shape**
- More concurrency does not always mean better latency
- Stability and predictability matter more than peak numbers
- Measuring real behavior teaches more than synthetic benchmarks

---

## Current Limitations

- Benchmarks are single-node only
- No request prioritization or fairness policy
- No persistent vector store for RAG
- No autoscaling or multi-GPU support
- No authentication or rate limiting

These limitations are intentional to keep the focus on **core inference behavior**.

---

## Future Work (Planned)

The following extensions are planned or under consideration:

### Inference & Performance
- Prefix caching comparison (on/off)
- Latency vs throughput curves across concurrency levels
- Model size comparison (0.5B vs 1.5B vs larger)
- vLLM vs Transformers inference comparison

### System Design
- FastAPI gateway with request validation
- Rate limiting and basic auth
- Async client load testing
- Structured logging and metrics export

### Deployment
- Dockerized vLLM server and client
- GPU-aware Docker Compose setup
- Multi-GPU experiments (if hardware allows)

### RAG Enhancements
- Persistent vector store
- Chunking strategy comparisons
- Retrieval quality vs latency trade-offs

---

## Why This Project Exists

This repository exists to build **intuition**, not just demos.

Understanding **how inference behaves under constraints** is critical for:
- ML Engineers
- GenAI Engineers
- Infra / Platform Engineers

The focus is on **learning deeply, measuring honestly, and documenting trade-offs**.

---

## Disclaimer

This project is a **learning-focused exploration** and is not intended to be a
production-ready system.

Performance numbers depend heavily on hardware, model choice, and workload shape.
 