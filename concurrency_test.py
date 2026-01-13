import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# OpenAI-compatible client pointing to vLLM server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)

PROMPT = "Explain vLLM batching and KV cache behavior in simple terms."

def send_request(i):
    start = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen2-1.5B-Instruct",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=50,
        temperature=0.2
    )
    latency = time.time() - start
    print(f"Request {i} completed in {latency:.2f}s")
    return latency

if __name__ == "__main__":
    N = 4  # number of concurrent requests

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=N) as executor:
        latencies = list(executor.map(send_request, range(N)))

    total_time = time.time() - start_all

    print("\n--- SUMMARY ---")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Average latency: {sum(latencies)/len(latencies):.2f}s")
