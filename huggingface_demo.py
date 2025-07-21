import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv(override=True)

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM3-3B",
    messages=[
        {
            "role": "user",
            "content": "What is AgenticAI?"
        }
    ],
)

print(completion.choices[0].message)