from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing_extensions import List, TypedDict
import bs4
from os import read
from PyPDF2 import PdfReader
import getpass
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
api_key = os.getenv("GOOGLE_API_KEY")

reader = PdfReader("UG_RULE_BOOK.pdf")
pdf_text = ""
for page in reader.pages:
    pdf_text += " " + page.extract_text()

if not api_key:
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai")

if not api_key:
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_text(pdf_text)

# Index chunks
_ = vector_store.add_texts(texts=all_splits)

#prompt
prompt = PromptTemplate.from_template("""
INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn’t contain the facts to answer the QUESTION return I don't know
Explain each query in detail while sticking to the facts and knowledge of the DOCUMENT.
Don't use phrases like 'Based on the provided document'
Use HTML for formatting your response, such as <b> for bold, <i> for italics, and <ul> for lists,<br> for line break, <hr> for horizontal line.

DOCUMENT: {context}

QUESTION: {question}
""")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

if __name__ == "__main__":
    # Test the graph with a sample question
    test_state = {"question": "What is the grading system for UG students?"}
    response = graph.invoke(test_state)
    print(response["answer"])
