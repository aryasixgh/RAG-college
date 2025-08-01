import time
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

prompt = "Explain the theory of relativity in 200 words."

start = time.time()
response = llm.invoke(prompt)
end = time.time()

print("Time taken: {:.2f} seconds".format(end - start))
print("\n---\nResponse:\n", response)
