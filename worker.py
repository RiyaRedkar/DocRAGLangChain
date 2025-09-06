import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm = None
llm_embeddings = None

# Initialize LLM and embeddings
def init_llm():
    global llm, llm_embeddings
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Set it in your .env file.")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    llm_embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain, llm, llm_embeddings
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, llm_embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

# Process user input
def process_prompt(prompt):
    global conversation_retrieval_chain, chat_history
    if not conversation_retrieval_chain:
        return "Please upload a PDF document first."
    result = conversation_retrieval_chain.invoke(
        {"question": prompt, "chat_history": chat_history}
    )
    chat_history.append((prompt, result["answer"]))
    return result['answer']

# Initialize LLM when module loads
init_llm()
