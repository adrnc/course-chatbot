import os
os.environ["USER_AGENT"] = "local"

from hashlib import sha256
from pathlib import Path
import sys

import chromadb
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} [model]")

# TODO: add dedicated embedding model?
model = sys.argv[1]
embedding_model = model
print(f'Using Ollama model "{model}" and embedding model "{embedding_model}"')

llm = Ollama(model=model)


### Construct retriever ###
def update_datahash(datafile: Path, datahashfile: Path) -> bool:
    old_hash = ""

    try:
        old_hash = datahashfile.read_text().strip()
    except FileNotFoundError:
        pass

    new_hash = sha256(datafile.read_text().encode("utf-8")).hexdigest()

    if old_hash == new_hash:
        return False

    datahashfile.write_text(new_hash)
    return True

def init_chroma(collection_name: str, datafile: Path, data_updated: bool, embeddings: OllamaEmbeddings) -> Chroma:
    client = chromadb.PersistentClient()

    client.get_or_create_collection(collection_name)

    if data_updated:
        client.delete_collection(collection_name)

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(datafile.read_text())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(md_header_splits)

        collection = client.get_or_create_collection(collection_name)
        collection.add(
            ids=[str(i) for i in range(len(splits))],
            documents=splits # type: ignore
        )

    return Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)



print("Initalizing...")

datafile = Path(__file__).parent.joinpath("data/content.md")
datahashfile = Path(__file__).parent.joinpath("data/contenthash.txt")
collection_name = "data"

data_updated = update_datahash(datafile, datahashfile)

if data_updated:
    print("Data has changed, updating...")
else:
    print("Data has not changed, no update needed")

embeddings = OllamaEmbeddings(model=embedding_model)
vectorstore = init_chroma(collection_name, datafile, data_updated, embeddings)
retriever = vectorstore.as_retriever()

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
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def chat(session_id: str) -> None:
    try:
        while True:
            prompt = input("> ");

            conversational_rag_chain.invoke(
                {"input": prompt},
                config={
                    "configurable": {"session_id": session_id},
                },
            )

            ai_message = store[session_id].messages[-1]
            print(str(ai_message.content).strip())
    except KeyboardInterrupt:
        pass

print("Chatbot ready, ask a question:")
chat("42")
