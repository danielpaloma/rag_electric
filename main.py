import os
import glob
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks import StdOutCallbackHandler

def initialize_constants():
    """Initialize constants and load environment variables."""
    print("****Initializing model and db_name")
    global MODEL, db_name
    
    MODEL = "gpt-4o-mini"
    db_name = "vector_db"
    
    load_dotenv(override=True)
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')


def add_metadata(doc, doc_type):
    """Add metadata to a document."""
    doc.metadata["doc_type"] = doc_type
    return doc


def create_knowledge_base():
    """Create the knowledge base and return chunks."""
    print("****Creating the knowledge base")
    # Read in documents using LangChain's loaders
    # Take everything in all the sub-folders of our knowledgebase
    folders = glob.glob("knowledge-base/*")
    
    text_loader_kwargs = {'encoding': 'utf-8'}
    # If that doesn't work (Windows users) --> uncomment the next line instead
    # text_loader_kwargs={'autodetect_encoding': True}
    
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total number of chunks: {len(chunks)}")
    print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")
    
    return chunks


def create_vectorstore(chunks):
    """Create the vector store using embeddings."""
    print("****Creating the vector store")
    #embeddings instance
    embeddings = OpenAIEmbeddings()
    
    #remove if db_name exists
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    
    # investigate the vectors
    collection = vectorstore._collection
    count = collection.count()
    
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    
    return vectorstore


def create_conversation_chain(vectorstore):
    """Create a conversation chain using the vectorstore."""
    print("****Creating a conversation chain using the vectorstore")
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        callbacks=[StdOutCallbackHandler()]
    )
    return conversation_chain

def make_chat(conversation_chain):
    """"Gradio accepts only fn signature with question and history
        This is a wrapper function"""
    
    def chat(question, history):
        result = conversation_chain.invoke({"question": question})
        return result["answer"]
    
    return chat


def main():
    print("Hello from rag-electric!")
    initialize_constants()
    chunks = create_knowledge_base()
    vectorstore = create_vectorstore(chunks)
    conversation_chain = create_conversation_chain(vectorstore)
    
    print("****Launching Gradio UI chat")
    chat_fn = make_chat(conversation_chain)
    view = gr.ChatInterface(chat_fn, type="messages").launch(inbrowser=True)


if __name__ == "__main__":
    main()
