from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    aws_access_key_id="",
    aws_secret_access_key="",
    # aws_session_token=...,
    # temperature=...,
    # max_tokens=...,
    # other params...
)

messages = [
    ("human", "what is agenticAI?"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)

