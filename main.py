#Implementing AWS Bedrock
import os
import glob
from dotenv import load_dotenv
import gradio as gr
from prompts.prompts import EE_SYSTEM_PROMPT

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_aws import BedrockLLM  # Use the updated Bedrock LLM wrapper
from config import DB_NAME, OPENAI_MODEL_ID, BEDROCK_MODEL_ID, BEDROCK_REGION, BEDROCK_MODEL_KWARGS

def initialize():
    """Load environment variables."""
    print("****Initializing... ")
    load_dotenv(override=True)
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')


def add_metadata(doc, doc_type):
    """Add metadata to a document."""
    doc.metadata["doc_type"] = doc_type
    return doc


def create_knowledge_base():
    """Create the knowledge base and return chunks from clean_data only."""
    print("****Creating the knowledge base from clean_data only")
    # Only use the clean_data directory and its subfolders
    base_folder = "knowledge-base/clean_data"
    folders = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    text_loader_kwargs = {'encoding': 'utf-8'}
    # If any Windows issue --> uncomment the next line instead
    # text_loader_kwargs={'autodetect_encoding': True}

    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        folder_docs = []
        # Load .md and .MD files
        loader_md = DirectoryLoader(
            folder,
            glob="**/*.[mM][dD]",
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs
        )
        folder_docs.extend(loader_md.load())
        # Load .txt and .TXT files
        loader_txt = DirectoryLoader(
            folder,
            glob="**/*.[tT][xX][tT]",
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs
        )
        folder_docs.extend(loader_txt.load())
        print(f"Found {len(folder_docs)} documents in subfolder '{doc_type}'")
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

    # Use RecursiveCharacterTextSplitter for better chunking
    # OpenAI embedding models have a max token limit per request (e.g., 8191 for text-embedding-ada-002, 8192 for text-embedding-3-small/large)
    # To be safe, use chunk_size=1000 and chunk_overlap=100 (in characters)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Print max chunk length for debugging
    max_len = max(len(chunk.page_content) for chunk in chunks) if chunks else 0
    print(f"Total number of chunks: {len(chunks)} (max chunk length: {max_len} chars)")
    print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")

    return chunks


def create_vectorstore(chunks=None, load_only=False):
    """Create or load the vector store using embeddings."""
    print("****Creating or loading the vector store")
    embeddings = OpenAIEmbeddings()
    if os.path.exists(DB_NAME):
        print(f"Vectorstore found at {DB_NAME}, loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    else:
        if load_only:
            raise FileNotFoundError(f"Vectorstore not found at {DB_NAME} and load_only=True.")
        print(f"Vectorstore not found at {DB_NAME}, creating new vectorstore...")
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
        print(f"Vectorstore created with {vectorstore._collection.count()} documents")
        collection = vectorstore._collection
        count = collection.count()
        sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)
        print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


def create_conversation_chain(vectorstore, llm_provider):
    """Create a conversation chain using the vectorstore with a system prompt."""
    print(f"****Creating a conversation chain using the vectorstore and provider: {llm_provider}")
    if llm_provider == "openai-gpt-4o-mini":
        llm = ChatOpenAI(temperature=0.7, model_name=OPENAI_MODEL_ID)
    elif llm_provider == "aws_br_mistral_small":
        llm = BedrockLLM(
            model_id=BEDROCK_MODEL_ID,
            region_name=BEDROCK_REGION,
            model_kwargs=BEDROCK_MODEL_KWARGS,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {llm_provider}")

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    system_prompt = EE_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{question}\n\nContext:\n{context}")
    ])

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        callbacks=[StdOutCallbackHandler()],
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain


def make_chat(vectorstore):
    """Gradio accepts only fn signature with question, history, and model provider."""
    def chat(question, history, llm_provider):
        conversation_chain = create_conversation_chain(vectorstore, llm_provider)
        result = conversation_chain.invoke({"question": question})
        return result["answer"]
    return chat


def main():
    print("Hello from rag-electric!")
    initialize()

    if os.path.exists(DB_NAME):
        print(f"Vectorstore already exists at {DB_NAME}. Skipping knowledge base creation and using existing vectorstore.")
        vectorstore = create_vectorstore(load_only=True)
    else:
        chunks = create_knowledge_base()
        vectorstore = create_vectorstore(chunks)

    chat_fn = make_chat(vectorstore)

    with gr.Blocks() as demo:
        gr.Markdown("# RAG Electrical Projects")
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=["openai-gpt-4o-mini", "aws_br_mistral_small"],
                value="openai-gpt-4o-mini",
                label="Select LLM Provider"
            )
        chatbot = gr.Chatbot()
        with gr.Row():
            user_input = gr.Textbox(label="Your question", scale=4)
            send_btn = gr.Button("Send", scale=1)
        state = gr.State([])  # To keep chat history

        def respond(message, history, llm_provider):
            # Defensive: ensure history is a list of [user, bot] pairs
            if not isinstance(history, list):
                history = []
            answer = chat_fn(message, history, llm_provider)
            history = history + [[message, answer]]
            return history, history

        send_btn.click(
            respond,
            inputs=[user_input, state, model_selector],
            outputs=[chatbot, state]
        )
        user_input.submit(
            respond,
            inputs=[user_input, state, model_selector],
            outputs=[chatbot, state]
        )

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
