# Building a Personalized RAG Chatbot with Session Management

I'll guide you through creating a personalized RAG (Retrieval Augmented Generation) chatbot with session management using Chroma vector store, Python, and Gradio. The chatbot will allow users to:

1. Upload multiple files
2. Query information from those files
3. View chat history
4. Delete uploaded files and reset the session

## Implementation

```python
import os
import uuid
import tempfile
import shutil
import gradio as gr
from typing import Dict, List
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    DocxLoader
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Dictionary to store session data
sessions = {}

def get_loader_for_file(file_path):
    """Return the appropriate loader based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return PyPDFLoader(file_path)
    elif ext == '.txt':
        return TextLoader(file_path)
    elif ext == '.csv':
        return CSVLoader(file_path)
    elif ext == '.docx':
        return DocxLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def process_files(session_id, files):
    """Process uploaded files and create/update the vector store"""
    # Create temp directory for this session if it doesn't exist
    if not os.path.exists(f"temp_{session_id}"):
        os.makedirs(f"temp_{session_id}")
    
    # Save uploaded files to temp directory
    file_paths = []
    for file in files:
        temp_path = os.path.join(f"temp_{session_id}", os.path.basename(file.name))
        shutil.copy(file.name, temp_path)
        file_paths.append(temp_path)
    
    # Process all documents
    documents = []
    for file_path in file_paths:
        try:
            loader = get_loader_for_file(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(documents)
    
    # Create or update vector store
    embedding = OpenAIEmbeddings()
    
    # If vector store already exists for this session, add to it
    persist_directory = f"chroma_db_{session_id}"
    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        vector_store.add_documents(split_documents)
    else:
        # Create new vector store
        vector_store = Chroma.from_documents(
            documents=split_documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
    
    vector_store.persist()
    
    # Create or update retrieval chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    
    # Update session data
    if session_id not in sessions:
        sessions[session_id] = {
            "vector_store": vector_store,
            "retrieval_chain": retrieval_chain,
            "file_paths": file_paths,
            "chat_history": []
        }
    else:
        sessions[session_id]["vector_store"] = vector_store
        sessions[session_id]["retrieval_chain"] = retrieval_chain
        sessions[session_id]["file_paths"].extend(file_paths)
    
    return f"Successfully processed {len(files)} files."

def reset_session(session_id):
    """Delete all files and reset the session"""
    if session_id in sessions:
        # Clean up temp directory
        if os.path.exists(f"temp_{session_id}"):
            shutil.rmtree(f"temp_{session_id}")
        
        # Clean up vector store
        if os.path.exists(f"chroma_db_{session_id}"):
            shutil.rmtree(f"chroma_db_{session_id}")
        
        # Remove session data
        del sessions[session_id]
    
    return "Session reset successfully. You can upload new files now."

def chat(session_id, query, history):
    """Process a query using the retrieval chain and update chat history"""
    if session_id not in sessions:
        return "Please upload some files first.", history
    
    try:
        # Get response from retrieval chain
        response = sessions[session_id]["retrieval_chain"].run(query)
        
        # Update chat history
        sessions[session_id]["chat_history"].append((query, response))
        
        # Update the Gradio chat history
        history.append((query, response))
        
        return "", history
    except Exception as e:
        return f"Error: {str(e)}", history

def create_chat_interface():
    """Create a Gradio interface for the chatbot"""
    with gr.Blocks() as demo:
        session_id = gr.State(lambda: str(uuid.uuid4()))
        
        gr.Markdown("# Personalized RAG Chatbot")
        gr.Markdown("Upload files, then ask questions about their content.")
        
        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(file_count="multiple", label="Upload Files")
                upload_button = gr.Button("Process Files")
                reset_button = gr.Button("Reset Session (Delete All Files)")
                status = gr.Textbox(label="Status", interactive=False)
            
        chatbot = gr.Chatbot(label="Chat History")
        msg = gr.Textbox(label="Ask a question about your documents")
        
        upload_button.click(
            fn=process_files, 
            inputs=[session_id, file_upload], 
            outputs=status
        )
        
        reset_button.click(
            fn=reset_session,
            inputs=[session_id],
            outputs=status
        )
        
        msg.submit(
            fn=chat,
            inputs=[session_id, msg, chatbot],
            outputs=[msg, chatbot]
        )
        
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch()
```

## How It Works

### Key Components:

1. **Session Management**:
   - Each user session gets a unique UUID
   - Files, vector stores, and chat history are maintained per session
   - All data is stored in session-specific directories

2. **File Processing**:
   - Multiple file formats supported (PDF, TXT, CSV, DOCX)
   - Files are uploaded, saved to a temp directory, and processed into documents
   - Documents are split into smaller chunks for more effective retrieval

3. **Vector Store Management**:
   - Uses Chroma DB for vector storage
   - Embeddings created using OpenAI's embedding model
   - Vector store is persisted to disk for each session

4. **RAG Implementation**:
   - Uses LangChain's ConversationalRetrievalChain
   - Maintains chat history using ConversationBufferMemory
   - Retrieves relevant document chunks to answer user queries

5. **UI Features**:
   - File upload button for multiple files
   - Reset button to delete all files and start fresh
   - Chat interface with history
   - Status messages for user feedback

### Enhancements You Could Consider

1. Add authentication to truly separate user sessions
2. Implement automatic session cleanup for inactive sessions
3. Add file type validation and size limits
4. Include document metadata in responses (source citation)
5. Add a feature to see what files are currently loaded

This implementation provides a complete solution for a personalized RAG chatbot with proper session management using Gradio, Chroma, and LangChain.