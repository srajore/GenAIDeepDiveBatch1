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

# Initialize chat history store
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Function to handle user input and generate a response
def chatbot(user_input, history_state, session_id, temperature):
    # Check for empty or whitespace-only input (Assignment 1)
    if not user_input or user_input.strip() == "":
        error_message = "Please enter a valid question!"
        if history_state is None:
            history_state = []
        history_state.append({"role": "user", "content": user_input})
        history_state.append({"role": "assistant", "content": error_message})
        # Display full history in reverse chronological order (latest to oldest)
        return history_state[::-1], history_state, session_id, "", f"Current temperature: {temperature}"

    # Set up the language model with the selected temperature (Assignment 3)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
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

    # Set up the chain with history
    chain = RunnableWithMessageHistory(
        runnable=prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    formatted_response = response.content if hasattr(response, 'content') else str(response)
    
    if history_state is None:
        history_state = []
    history_state.append({"role": "user", "content": user_input})
    history_state.append({"role": "assistant", "content": formatted_response})

    # Display full history in reverse chronological order (latest to oldest)
    return history_state[::-1], history_state, session_id, "", f"Current temperature: {temperature}"

# Function to clear conversation history and input box (Assignment 2)
def clear_history(session_id):
    # Clear the session history from the store
    if session_id in store:
        store[session_id].clear()
    # Reset history_state, session_id, input_box, chatbot display, and temperature display
    return [], None, str(uuid.uuid4()), "", "Current temperature: 0.7"

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Zensar Chatbot")
    history_state = gr.State(value=None)
    session_id = gr.State(value=str(uuid.uuid4()))
    temperature = gr.State(value=0.7)
    
    input_box = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    temperature_display = gr.Textbox(label="Temperature Status", interactive=False)
    temperature_slider = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.7,
        label="Temperature (controls response creativity)",
        step=0.1
    )
    with gr.Row():
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear History")

    chatbot_display = gr.Chatbot(label="Conversation History", height=400, type="messages")

    submit_button.click(
        fn=chatbot,
        inputs=[input_box, history_state, session_id, temperature_slider],
        outputs=[chatbot_display, history_state, session_id, input_box, temperature_display]
    )
    clear_button.click(
        fn=clear_history,
        inputs=[session_id],
        outputs=[chatbot_display, history_state, session_id, input_box, temperature_display]
    )

# Launch the Gradio app
demo.launch()