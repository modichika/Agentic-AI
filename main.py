# #The Brain - selecting a chat model
# LOAD -> SPLIT -> EMBED -> RETRIEVE.
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")



# Injection Engine data->document with same format
# Loading Documents. 
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.tools import tool

def load_web_documents(urls: list):
    """
    Takes a list of URLs and returns a flat list of LangChain documents.
    """
    print(f"Fetching data from {len(urls)} URLs...")

    nested_docs = [WebBaseLoader(url).load() for url in urls] # becomes a list of lists -- [[Doc1], [Doc2], [Doc3]]

    flat_docs = [doc for sublist in nested_docs for doc in sublist] # flatten the list of lists -- ([Doc1, Doc2, Doc3])

    return flat_docs


# Tool for the langgraph
@tool  # The interface between the custom logic and AI's thinking.
def retrieve_blog_posts(query: str) -> str:
    """
    Search and return information about Lilian Weng blog posts.
    """
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# The execution

if __name__ == "__main__": # python standard only run the code inside here if i run python main.py
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = load_web_documents(urls)
    print(f"Successfully loaded {len(docs)} documents.")
    # if docs:
    #    print(f"Sample Metadata: {docs[0].metadata}") # metadata = source url, title, language.

    # print("RAW DATA CHECK:")
    # print(docs[0].page_content.strip()[:1000])


#Splitting docs

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder( # counts tokens like cat, ing, tion
            chunk_size=100,
            chunk_overlap=50,
        )
    doc_splits = text_splitter.split_documents(docs) # Is Storing chunks of the docs
    print(f"Split into {len(doc_splits)} chunks.")
    # print(doc_splits[0].page_content.strip()) # strips the whole page content we might set a limit like 1000 or even leave like this.
    # print(f"Total chunks created: {len(doc_splits)}")

# Select an embeddigns model, Select a vector store the memory:

    URI = "./milvus_example.db"

    vector_store = Milvus.from_documents(
        documents=doc_splits,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
        connection_args={"uri": URI},
        drop_old=True,
    )
    # Convert it into the Retriever tool for LangGraph
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("Testing Tool...")
    result = retrieve_blog_posts.invoke({"query": "types of reward hacking"})
    print(f"Reward Hacking Result: {result[:1000]}... {len(result)}")





