# imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import gradio as gr
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Set up the language model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    #api_key=''  # Replace with your actual API key
)

# Create a simple chat prompt
prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a friendly and helpful assistant named ZenBot. Start with the greeting 'Hello, I am ZenBot. How can I assist you today?'. Answer the user's questions to the best of your ability."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Initialize chat history store
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Set up the chain with history
chain = RunnableWithMessageHistory(
    runnable=prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Function to handle user input and generate a response
def chatbot(user_input, history_state, session_id=str(uuid.uuid4())):
    # Check for empty or whitespace-only input
    if not user_input or user_input.strip() == "":
        error_message = "Please enter a valid question!"
        if history_state is None:
            history_state = []
        history_state.append((user_input, error_message))
        return error_message, history_state, session_id

    # Process valid input
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    formatted_response = response.content if hasattr(response, 'content') else str(response)
    
    if history_state is None:
        history_state = []
    history_state.append((user_input, formatted_response))

    return formatted_response, history_state, session_id

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Zensar Chatbot")
    history_state = gr.State(value=None)
    session_id = gr.State(value=str(uuid.uuid4()))
    input_box = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    output_box = gr.Textbox(label="Answer", interactive=False)
    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=chatbot,
        inputs=[input_box, history_state, session_id],
        outputs=[output_box, history_state, session_id]
    )

# Launch the Gradio app
demo.launch()