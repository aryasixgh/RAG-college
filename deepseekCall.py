# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()

api_key_depseek = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= api_key_depseek,
)

start_time = time.time()

completion = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[
        {"role": "user", "content": "What is the meaning of life?"}
    ],
    max_tokens = 500
)

end_time = time.time()

response_text = completion.choices[0].message.content
elapsed_time = end_time - start_time
num_tokens = len(response_text.split())  # Approximate (you can use tiktoken for precise count)
tps = num_tokens / elapsed_time

print(f"Response time: {elapsed_time:.2f} seconds")
print(f"Estimated tokens: {num_tokens}")
print(f"Estimated TPS: {tps:.2f} tokens/second\n")

print(completion.choices[0].message.content)