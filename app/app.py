import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import requests
from dotenv import load_dotenv
load_dotenv()
# Load and split documents
docs = []
for i in range(1, 4):
    loader = TextLoader(f"docs/doc{i}.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs.extend(text_splitter.split_documents(documents))

openai_api_key = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = openai_api_key

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = FAISS.from_documents(docs, embedding)

# QA Chain
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Define tools
def simple_calculator(query):
    try:
        return str(eval(query))
    except:
        return "Invalid expression."

def dictionary_lookup(query):
    word = query.lower().split("define ")[-1]
    resp = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
    if resp.status_code == 200:
        return resp.json()[0]['meanings'][0]['definitions'][0]['definition']
    return "Word not found."

def retrieve_chunks(query, k=3):
    return vectorstore.similarity_search(query, k=k)

def generate_answer(query):
    return qa_chain.run(query)

def route_query(query):
    log = ""
    if "calculate" in query.lower():
        log = "Routed to Calculator"
        result = simple_calculator(query.replace("calculate", "").strip())
    elif "define" in query.lower():
        log = "Routed to Dictionary"
        result = dictionary_lookup(query)
    else:
        log = "Routed to RAG Pipeline"
        chunks = retrieve_chunks(query)
        result = generate_answer(query)
    return log, result

# Streamlit UI
st.title("📚 Multi-Agent Knowledge Assistant")

query = st.text_input("Ask your question:")

if query:
    log, answer = route_query(query)
    st.markdown(f"**Decision:** {log}")
    st.markdown(f"**Answer:** {answer}")