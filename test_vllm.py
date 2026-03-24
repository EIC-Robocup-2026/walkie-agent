from langchain_openai import ChatOpenAI
import time

inference_server_url = "http://localhost:8000/v1"

llm = ChatOpenAI(
    model="qwen3.5-9b",
    api_key="your api key goes here",
    base_url=inference_server_url,
    temperature=0,
)

start_time = time.time()
response = llm.invoke("What is the capital of France?")
print(response)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")