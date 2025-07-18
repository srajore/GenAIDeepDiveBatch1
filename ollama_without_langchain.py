import ollama

response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {"role": "user", "content": "what is the capital of France?"},
    ])

print(response["message"]["content"])