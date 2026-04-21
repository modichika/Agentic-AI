# PART-1
# #The Brain - selecting a chat model
# LOAD -> SPLIT -> EMBED -> RETRIEVE. AI is asleep in these phases. These phases are handled by my script.
# The LLM does not knows or runs my python code.


import os
# Injection Engine data->document with same format
# Loading Documents. 
import asyncio
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient 
from langchain_milvus import Milvus
from langchain.tools import tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableLambda

# api_key = os.getenv("OPENAI_API_KEY")

def check_api_keys():
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
check_api_keys()



def load_web_documents(urls: list):
    """
    Takes a list of URLs and returns a flat list of LangChain documents.
    """
    print(f"Fetching data from {len(urls)} URLs...")
    
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    nested_docs = [] # becomes a list of lists -- [[Doc1], [Doc2], [Doc3]]
    for url in urls:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs={"parse_only": bs4_strainer}
        )
        nested_docs.append(loader.load())

    flat_docs = [doc for sublist in nested_docs for doc in sublist] # flatten the list of lists -- ([Doc1, Doc2, Doc3])

    return flat_docs

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to store documents in vector_store
milvus_client = MilvusClient(uri="http://localhost:19530")
collection_name = "rag_docs_collection"
def milvus_store(doc_splits):
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=768,
    )
    
    # Preparing the data
    data = []
    for i, doc in enumerate(doc_splits):
        vector = embedding_model.embed_query(doc.page_content)
        data.append({
        "id": i,
        "vector": vector,
         "text": doc.page_content,
         "source": doc.metadata.get("source", "unknown"), # coz if any doc doesn't have metadata then it will return unknown    
        })
    
    milvus_client.insert(collection_name=collection_name, data=data)
    print(f"Successfully inserted {len(data)} chunks into Milvus Standalone.")
    



# Tool for the langgraph
@tool  # The interface between the custom logic and AI's thinking.
def retrieve_blog_posts(query: str) -> str:  # query is an args as a JSON object.
    """
    Search and return information about Lilian Weng blog posts.
    """
    query_vector = embedding_model.embed_query(query)
    
    search_milvus = milvus_client.search(collection_name=collection_name, data=[query_vector], limit=3, output_fields=["text"])
    
    # Format the output for the LLM
    retrieved_texts = [res["entity"] ["text"] for res in search_milvus[0]]
    return "\n\n".join(retrieved_texts)
    
retriever_tool = retrieve_blog_posts


# THINKING OF AI MODEL about whether to answer or skip HAPPENS The Node - 1. The assistant 
# Generating Query. Graph node. - # PART-2 - Autonomous Agent - that thinks before acting. The LANGGRAPH LOGIC
# Includes MessagesState makes it easy to use messages

response_model = init_chat_model("gpt-5-nano", temperature=0) # Initialize the model

def generate_query_or_respond(state: MessagesState):
    """
    Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
     
     # We bind the tool with the response_model so that it can decide whether to proceed or use the tool.

    response = (
        response_model.bind_tools([retriever_tool]).invoke(state["messages"][-2:]) # .invoke the messages from the langgraph state, we sent a hidden instruction to the OpenAI by using the .bind_tools([retriever_tool]). Returns an AIMessage Object.
    )
    return {"messages": [response]} # Return the new message to be added to the state. User's history


# GRADING DOCUMENTS

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import convert_to_messages

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n"
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

# Pytdantic is like zod for data validation, BaseModel is a class that serves as a foundation for defining data models using python type annotations. This goes to the LLM, so that the AI returns a JSON object {"binary_score": "yes/no"} to indicate whether the docs are relevant to user's question.
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model("gpt-5-nano", temperature=0)

# The node 2, it returns a string that acts as a router for the system it tells the system "Go to the generate_answer step or rewrite_question"
def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content # 1st message is almost always the user's question
    context = state["messages"][-1].content  # latest message -1 that is the message tool has returned, the search result
    
    # # ADD THIS PRINT:
    # print(f"--- GRADER IS COMPARING ---")
    # print(f"QUESTION: {question}")
    # print(f"CONTEXT: {context}")
    
    # This is where AI is the judge something like this : "ou are a grader... Here is the retrieved document: meow. Here is the user question: What does Lilian Weng say about types of reward hacking?... Give a binary score"
    prompt = GRADE_PROMPT.format(question=question, context=context) # Takes GRADE_PROMPT and injects the result of question(user's question), context(tool's search result)
    response = (
        # Telling the AI "I don't want the converstaion but the JSON object that matches the GradeDocuments class which is binary_score: 'yes' or 'no'. "
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    
    score = response.binary_score
    # The Edge of the graph
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question" # Fix the question and search again
# The grader_model and the assistant are the same AI model but doing different things. the grader_model:

# Node - 3 - Rewrite question - giving the response_model node retriever tool's irrelevant docs which indicates the need to improve the original user question.

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # langchain_core.messages imports a class HumanMessage used to represent a message sent by a human user to a chat model.

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning. \n"
    "Here is the initial question:"
    "\n ------ \n"
    "{question}"
    "\n ------ \n"
    "Formulate only one improved question:"
)


async def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content # coz messages[0] represents the user's question
    prompt = REWRITE_PROMPT.format(question=question)
    response = await response_model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]} # content attribute holds the main payload, it is usually a string, but can also be a list of dictionaries for multimodal data.

# Node - 4 generate_answer The llm decides to generate answer if the retriver tool gave the content relevant to the data

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks."
    "Use the foloowing pieces of retrieved context to answer the question."
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

async def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    print("--- DEBUG: WHAT THE AI IS ACTUALLY READING ---")
    print(context)
    prompt =  GENERATE_PROMPT.format(question=question, context=context)
    response = await response_model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}



# Assembling the graph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


workflow = StateGraph(MessagesState)

#Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")


# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call 'retriever_tool' tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the action node is called
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile(checkpointer=MemorySaver())

#Visualizing the Graph

from IPython.display import Image, display
from PIL import Image as PILImage
import io

#display(Image(graph.get_graph().draw_mermaid_png())) -- for jupyter notebook
graph_bytes = graph.get_graph().draw_mermaid_png()
    
with open("graph.png", "wb") as f:
  f.write(graph_bytes)



# The execution


from uuid import uuid4
from dotenv import load_dotenv
    
    
load_dotenv()
    
    
urls = [
    "https://www.kubeflow.org/docs/started/architecture/",
]

docs = load_web_documents(urls)
print(f"Successfully loaded {len(docs)} documents.")
# if docs:
#    print(f"Sample Metadata: {docs[0].metadata}") # metadata = source url, title, language.

# print("RAW DATA CHECK:")
# print(docs[0].page_content.strip()[:1000])


#Splitting docs

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder( # counts tokens like cat, ing, tion
        chunk_size=300,
        chunk_overlap=100,
     )
doc_splits = text_splitter.split_documents(docs) # Is Storing chunks of the docs
print(f"Split into {len(doc_splits)} chunks.")
# print(doc_splits[0].page_content.strip()) # strips the whole page content we might set a limit like 1000 or even leave like this.
# print(f"Total chunks created: {len(doc_splits)}")

milvus_store(doc_splits)

    
# Test of the complete graph
        
async def main():
        inputs = {"messages": [HumanMessage(content = "What does Lilian Weng say about types of reward hacking?",)]}
        async for chunk in graph.astream(inputs, stream_node="updates", config=RunnableConfig(configurable={"thread_id": uuid4()}),):
            for node, update in chunk.items():
                print(f"Update from node", node)
                print("----")
                print(update["messages"][-1].pretty_print())
                print("\n\n")
    

asyncio.run(main())




if __name__ == "__main__": # python standard only run the code inside here if i run python main.py
    print("Agent is ready for local testing!")
