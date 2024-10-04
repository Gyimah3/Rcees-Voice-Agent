from langchain_core.tools import tool

# from langchain_community.tools import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import os
from dotenv import load_dotenv
load_dotenv()
# tavily=os.getenv('TAVAILY_API_KEY')
# os.environ['TAVAILY_API_KEY']=tavily

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

@tool
def add(a: int, b: int):
    """Add two numbers. Please let the user know that you're adding the numbers BEFORE you call the tool"""
    return a + b

@tool
async def setup_rag():
  """
  retrieve from RCEES internal documents and information on RCEES that cannot be found on the internet
  """
  page_url1 = "https://rcees.uenr.edu.gh/about-us/"
  page_url2='https://rcees.uenr.edu.gh/'
  page_url3='https://rcees.uenr.edu.gh/research/'
  page_url4= 'https://rcees.uenr.edu.gh/publication/'
  page_url5='https://rcees.uenr.edu.gh/msc-seem/'
  page_url6='https://rcees.uenr.edu.gh/msc-eema/'
  page_url7='https://rcees.uenr.edu.gh/phd-seem/'
  page_url8='https://rcees.uenr.edu.gh/phd-eema/'
  page_url9 = 'https://rcees.uenr.edu.gh/short-courses/'
  loader = WebBaseLoader(web_paths=[page_url1,page_url2,page_url3,page_url4,page_url5,page_url6,page_url7,page_url8,page_url9])
  docs = []
  async for doc in loader.alazy_load():
    docs.append(doc)

  assert len(docs) == 9
  doc = docs[0]
  # Load documents
  # loader = TextLoader("university.txt")  # Changed to TextLoader for single file
  # documents = loader.load()
  
  # Split documents
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_documents(docs)
  
  # Create embeddings
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  
  # Create vector store
  vectorstore = Chroma.from_documents(texts, embeddings)
  
  # Create retriever tool
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  rag_tool = create_retriever_tool(
      retriever,
      "rcees_rag_tool",
      "Searches the RCEES(Regional Center for Energy and Environmental Sustainability) internal documents for information that is found on the web.\n\n"
      "Let the user know you're asking your friend RCEES for help before you call the tool."
  )
  
  return rag_tool

# rag_tool = setup_rag()

tavily_tool = TavilySearchResults(
    max_results=5,
    include_answer=True,
    description=(
        "This is a search tool for accessing the internet.\n\n"
        "Let the user know you're asking your friend Tavily for help before you call the tool."
    ),
)

TOOLS = [setup_rag, tavily_tool]
