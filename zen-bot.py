# imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr


# Set up llama3.2:latest with Ollama
llm = ChatOllama(model="llama3.2:latest")

# Create a simple chat prompt

prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant.Answer the user's questions to the best of your ability."),
        ("human", "{question}"),
    ]
)

chain = prompt | llm

# Function to handle user input and generate a response
def chatbot(question):
    response = chain.invoke({"question": question})
    return response.content


# Set up the Gradio interface

with gr.Blocks() as demo:
    gr.Markdown("## Zensar Chatbot")
    input_box = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    output_box= gr.Textbox(label="Answer",interactive=False)
    submit_button=gr.Button("Submit")


    submit_button.click(
        fn=chatbot,
        inputs=input_box,
        outputs=output_box
    )


# Launch the Gradio app
demo.launch()