import os
from typing import Annotated, Any, Sequence, TypedDict
os.environ["USER_AGENT"] = "local"

from hashlib import sha256
from pathlib import Path
import sys

import chromadb
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} [model]")

# TODO: improve debugging options
debug = False

# TODO: add dedicated embedding model?
model = sys.argv[1]
embedding_model = model

print(f'Using Ollama model "{model}"')

llm = Ollama(model=model)


### Construct retriever ###
def check_datahash(model: str, datafile: Path, datahashfile: Path) -> tuple[bool, str]:
    old_hash = ""

    try:
        old_hash = datahashfile.read_text().strip()
    except FileNotFoundError:
        pass

    text = model + datafile.read_text()
    new_hash = sha256(text.encode("utf-8")).hexdigest()

    return old_hash != new_hash, new_hash

def update_datahash(datahashfile: Path, new_hash: str):
    datahashfile.write_text(new_hash)

def init_chroma(collection_name: str, datafile: Path, data_updated: bool, embeddings: OllamaEmbeddings) -> Chroma:
    client = chromadb.PersistentClient()
    chroma = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)

    if not data_updated:
        return chroma

    chroma.reset_collection()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(datafile.read_text())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(md_header_splits)

    show_progress = embeddings.show_progress
    embeddings.show_progress = True

    chroma.add_documents(
        ids=[str(i) for i in range(len(splits))],
        documents=splits
    )

    embeddings.show_progress = show_progress

    return chroma



print("Initalizing...")

datafile = Path(__file__).parent.joinpath("data/content.md")
datahashfile = Path(__file__).parent.joinpath("data/contenthash.txt")
collection_name = "data"

data_updated, new_hash = check_datahash(model, datafile, datahashfile)

if data_updated:
    print("Model or data has changed, updating...")
else:
    print("Model and data has not changed, no update needed")

embeddings = OllamaEmbeddings(model=embedding_model, show_progress=debug)
vectorstore = init_chroma(collection_name, datafile, data_updated, embeddings)
retriever = vectorstore.as_retriever()

if data_updated:
    update_datahash(datahashfile, new_hash)

print("Initialization and data loading successful")


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
prompt_file = Path(__file__).parent.joinpath("data/prompt.txt")
system_prompt = prompt_file.read_text()
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def chat(session_id: str) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    try:
        while True:
            prompt = input("> ")

            print("...")
            result = app.invoke({"input": prompt}, config)

            answer = result["answer"].strip()

            if debug:
                docs_content = [doc.page_content for doc in result["context"]]

                print(f"\nCONTEXT: {"\n\n".join(docs_content).strip()}")
                print(f"\nANSWER: {answer}\n")
            else:
                # TODO: add output streaming
                print(answer)



    except KeyboardInterrupt:
        pass

print("Chatbot ready, ask a question:")
chat("42")
