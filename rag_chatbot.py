from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import os

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)

DATA_DIR = "data"

documents = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

print("\nRAG chatbot ready (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(d.page_content for d in docs)
    pages = sorted({d.metadata.get("page", 0) for d in docs})

    prompt = f"""
Answer using ONLY the context below.
If the answer is not present, say "Not found in the document".

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="Qwen/Qwen2-1.5B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    print("\nAssistant:", response.choices[0].message.content)
    print(f"Sources: Page(s) {', '.join(map(str, pages))}\n")
