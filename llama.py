import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  

# Load API key from environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session state for chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores full chat history
if "user_name" not in st.session_state:
    st.session_state.user_name = None  # Stores the user's name

# Define prompt template
prompt_template = """System: You are a helpful assistant.
User: Question: {question}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question: {question}")
])

def generate_response(question, api_key, engine, temperature, max_tokens):
    """Generate response from Groq API."""
    try:
        llm = ChatGroq(model=engine, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("🧠 Smart Chatbot with Memory")

# Sidebar
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Groq API Key:", type="password", value=API_KEY)

# Model selection
engine = st.sidebar.selectbox("🤖 Select Model", [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "qwen-qwq-32b",
    "qwen-2.5-coder-32b",
    "qwen-2.5-32b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-70b"
])

# Adjust response parameters
temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("✍️ Max Tokens", 50, 300, 150)

# Clear chat history button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.user_name = None
    st.success("Chat history cleared!")

# Display chat messages
st.write("💬 **Chat with me!**")

# Display full chat history correctly
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(message["content"])

# User input field
user_input = st.chat_input("Type your message...")

if user_input:
    # Store user question
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user's question
    with st.chat_message("user"):
        st.write(user_input)

    # Check if user is introducing their name
    if "my name is" in user_input.lower():
        st.session_state.user_name = user_input.split("my name is")[-1].strip().capitalize()
        response = f"Hello {st.session_state.user_name}, nice to meet you! 😊"
    elif "what's my name" in user_input.lower():
        response = f"Your name is {st.session_state.user_name}!" if st.session_state.user_name else "I don't remember your name yet. Tell me again! 🤔"
    else:
        # Maintain full chat history for context
        conversation_history = "\n".join([msg["content"] for msg in st.session_state.messages])
        full_question = f"{conversation_history}\nUser: {user_input}"
        
        response = generate_response(full_question, api_key, engine, temperature, max_tokens)

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display AI response
    with st.chat_message("assistant"):
        st.write(response)
