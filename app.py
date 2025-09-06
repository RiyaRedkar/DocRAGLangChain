import streamlit as st
import worker

st.set_page_config(page_title="Simple RAG Chatbot", layout="centered")
st.title("📚 Simple RAG Chatbot")

# Sidebar for file upload
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

# Process uploaded PDF
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Processing document..."):
        worker.process_document("temp.pdf")
    st.sidebar.success("✅ PDF uploaded and processed!")

# Chat container
messages = st.container()
user_input = st.chat_input("Ask something from your PDF...")

if user_input:
    with st.spinner("Thinking..."):
        response = worker.process_prompt(user_input)
    with messages:
        # User message
        st.markdown(f"""
        <div style='text-align: right; margin-bottom: 1rem; padding: 0.8rem 1rem; border-radius: 0.5rem; background-color: #d3d3d3; display: inline-block; max-width: 80%; float: right; clear: both;'>
            {user_input}
        </div>
        """, unsafe_allow_html=True)

        # Bot response
        st.write(f"**Bot:** {response}")
