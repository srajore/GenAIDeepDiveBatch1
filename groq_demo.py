from groq import Groq

from dotenv import load_dotenv

import os

#import httpx

load_dotenv(override=True)

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
    #http_client=httpx.Client(verify=False)  # Disable SSL verification for local testing
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)
