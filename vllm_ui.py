import streamlit as st
import time
from openai import OpenAI

# vLLM OpenAI-compatible client
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)

st.set_page_config(page_title="vLLM Streaming UI", page_icon="⚡")
st.title("⚡ vLLM Streaming Inference UI")

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7)
max_tokens = st.sidebar.slider("Max tokens", 20, 200, 60)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.chat_message("user").write(user_input)

    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()

    full_response = ""
    start_time = time.time()
    first_token_time = None
    token_count = 0

    stream = client.chat.completions.create(
        model="Qwen/Qwen2-1.5B-Instruct",
        messages=[{"role": "user", "content": user_input}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += len(delta.split())
            full_response += delta
            placeholder.markdown(full_response + "▌")

    end_time = time.time()

    latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else latency
    tps = token_count / latency if latency > 0 else 0

    placeholder.markdown(full_response)

    st.sidebar.subheader("📊 Metrics")
    st.sidebar.metric("Latency (s)", f"{latency:.2f}")
    st.sidebar.metric("TTFT (s)", f"{ttft:.2f}")
    st.sidebar.metric("Tokens/sec", f"{tps:.2f}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
