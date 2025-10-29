import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.llm import get_model
from models.embeddings import EmbeddingModel
from utils.rag_utils import RAGSystem
from utils.search_utils import search_web


def get_chat_response(chat_model, messages, system_prompt, rag_context="", search_context=""):
    """Get response from the chat model with optional RAG and search context"""
    try:
        # Build enhanced system prompt with context
        enhanced_prompt = system_prompt
        
        if rag_context:
            enhanced_prompt += f"\n\n**Relevant Information from Documents:**\n{rag_context}"
        
        if search_context:
            enhanced_prompt += f"\n\n{search_context}"
        
        # Prepare messages for the model
        formatted_messages = [SystemMessage(content=enhanced_prompt)]
        
        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"


def initialize_rag_system():
    """Initialize RAG system with embedding model"""
    try:
        if 'embedding_model' not in st.session_state:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = EmbeddingModel()
        
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = RAGSystem(st.session_state.embedding_model)
        
        return st.session_state.rag_system
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None


def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("### Welcome to the Medical Assistant Chatbot!")
    
    st.markdown("""
    ## What This Chatbot Can Do
    
    This intelligent chatbot combines multiple AI capabilities:
    
    1. ** Multi-Provider LLM Support**
       - OpenAI (GPT models)
       - Groq (Llama models - Fast & Free!)
       - Google Gemini (Latest Google AI)
    
    2. ** RAG (Retrieval-Augmented Generation)**
       - Upload your own medical documents (PDF/TXT)
       - Chatbot answers based on YOUR documents
       - Perfect for: Medical guidelines, research papers, protocols
    
    3. ** Live Web Search**
       - Get latest medical information from the web
       - Updates knowledge beyond training data
       - Real-time medical news and research
    
    4. ** Response Modes**
       - **Concise**: Quick, brief answers (2-3 sentences)
       - **Detailed**: Comprehensive, in-depth explanations
    
    ---
    
    ##  Setup Instructions
    
    ### Step 1: Install Dependencies
```bash
    pip install -r requirements.txt
```
    
    ### Step 2: Get API Keys (Choose at least ONE)
    
    #### Option 1: Groq (Recommended - Free & Fast) 
    1. Visit: https://console.groq.com/keys
    2. Sign up with Google/Email
    3. Create API Key
    4. Copy the key (starts with `gsk_`)
    
    #### Option 2: Google Gemini (Free)
    1. Visit: https://makersuite.google.com/app/apikey
    2. Sign in with Google
    3. Create API Key
    4. Copy the key
    
    #### Option 3: OpenAI (Paid)
    1. Visit: https://platform.openai.com/api-keys
    2. Sign up and add payment method
    3. Create API Key
    4. Copy the key
    
    ### Step 3: Enter API Keys
    - Go to the **Chat** page
    - Enter your API key in the sidebar
    - Select your preferred LLM provider
    - Start chatting!
    
    ---
    
    ##  How to Use
    
    ### Basic Chat
    1. Navigate to **Chat** page
    2. Select LLM provider
    3. Type your medical question
    4. Get instant answers!
    
    ### Using RAG (Document Upload)
    1. Click **"Enable RAG"** in sidebar
    2. Upload PDF/TXT medical documents
    3. Wait for processing
    4. Ask questions about your documents!
    
    ### Using Web Search
    1. Check **"Enable Web Search"**
    2. Ask about latest medical info
    3. Chatbot searches web automatically
    4. Get current information!
    
    ### Response Modes
    - **Concise**: For quick facts
    - **Detailed**: For deep explanations
    
    ---
    
    ## Example Use Cases
    
    ### Medical Students
    - Upload textbooks/notes
    - Ask questions about topics
    - Get quick summaries before exams
    
    ### Healthcare Professionals
    - Upload treatment guidelines
    - Query drug interactions
    - Search latest research
    
    ### General Users
    - Ask about symptoms
    - Learn about conditions
    - Understand medications
    
    ---
    
    ##  Important Notes
    
    - **Not a substitute for professional medical advice**
    - Always consult healthcare providers for medical decisions
    - Use for educational and informational purposes only
    - Verify critical information from authoritative sources
    
    ---
    
    ##  Troubleshooting
    
    **"Invalid API Key" error:**
    - Check if key is correct
    - Verify no extra spaces
    - Ensure provider matches key type
    
    **"RAG not working":**
    - Upload valid PDF/TXT files
    - Wait for processing to complete
    - Check file size (keep under 10MB)
    
    **"Search failed":**
    - Check internet connection
    - Try simpler search terms
    - Wait a moment and retry
    
    ---
    
    ##  Ready to Start?
    
    Navigate to the **Chat** page using the sidebar and start exploring!
    """)


def chat_page():
    """Main chat interface page with all features"""
    st.title("Medical Assistant Chatbot")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header(" Configuration")
        
        # LLM Provider Selection
        st.subheader(" LLM Provider")
        provider = st.selectbox(
            "Choose Provider:",
            ["Groq", "Google", "OpenAI"],
            help="Select which AI provider to use"
        )
        
        # API Key Input - Try to load from secrets first
        from config.config import GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY
        
        # Map provider to secret keys
        provider_keys = {
            "groq": GROQ_API_KEY,
            "google": GOOGLE_API_KEY,
            "openai": OPENAI_API_KEY
        }
        
        # Check if key exists in secrets
        secret_key = provider_keys.get(provider.lower(), "")
        
        if secret_key:
            # Use key from secrets
            api_key = secret_key
            st.sidebar.success(f"{provider} configured from secrets!")
        else:
            # Ask user to enter manually if no secret found
            api_key = st.text_input(
                f"{provider} API Key:",
                type="password",
                help=f"Enter your {provider} API key"
            )
        
        st.divider()
        
        # Response Mode Selection (FEATURE 3)
        st.subheader(" Response Mode")
        response_mode = st.radio(
            "Select mode:",
            ["Concise", "Detailed"],
            help="Concise: Brief answers (2-3 sentences)\nDetailed: Comprehensive explanations"
        )
        
        st.divider()
        
        # RAG Configuration (FEATURE 1)
        st.subheader(" RAG (Document Upload)")
        use_rag = st.checkbox(
            "Enable RAG",
            help="Upload documents to answer questions based on your files"
        )
        
        if use_rag:
            uploaded_files = st.file_uploader(
                "Upload Documents:",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                help="Upload medical documents, research papers, or notes"
            )
            
            if uploaded_files:
                if st.button(" Process Documents", use_container_width=True):
                    with st.spinner("Processing documents..."):
                        rag_system = initialize_rag_system()
                        if rag_system:
                            chunks = rag_system.process_documents(uploaded_files)
                            if chunks:
                                success = rag_system.build_index(chunks)
                                if success:
                                    st.success(f" Processed {len(chunks)} chunks from {len(uploaded_files)} files!")
                                    st.session_state.rag_ready = True
                                else:
                                    st.error(" Failed to build index")
                            else:
                                st.error(" No content extracted from files")
            
            if st.session_state.get('rag_ready', False):
                st.info(" RAG system ready!")
        
        st.divider()
        
        # Web Search Configuration (FEATURE 2)
        st.subheader(" Web Search")
        use_search = st.checkbox(
            "Enable Web Search",
            help="Search the web for latest information when needed"
        )
        
        st.divider()
        
        # Clear Chat Button
        if st.button(" Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # System Prompt based on Response Mode
    if response_mode == "Concise":
        system_prompt = """You are a helpful medical assistant. Provide BRIEF, CONCISE answers.
        Keep responses to 2-3 sentences maximum. Be direct and to the point.
        Focus on the most important information only."""
    else:
        system_prompt = """You are a knowledgeable medical assistant. Provide DETAILED, COMPREHENSIVE answers.
        Explain concepts thoroughly, include relevant context, and give complete information.
        Use examples and elaborate on important points."""
    
    # Initialize chat model
    try:
        if api_key:
            chat_model = get_model(provider.lower(), api_key)
            # Don't show success message again if already shown from secrets
            if not secret_key:
                st.sidebar.success(f"{provider} model loaded!")
        else:
            st.warning(" Please enter an API key in the sidebar to start chatting.")
            st.info(" Select a provider and enter your API key on the left")
            st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        avatar = "🩺" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your medical question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag_context = ""
                search_context = ""
                
                # RAG: Retrieve relevant documents
                if use_rag and st.session_state.get('rag_ready', False):
                    rag_system = st.session_state.get('rag_system')
                    if rag_system and rag_system.is_ready():
                        relevant_docs = rag_system.retrieve(prompt)
                        if relevant_docs:
                            rag_context = "\n\n".join(relevant_docs)
                            st.info(f"Found {len(relevant_docs)} relevant document chunks")
                
                # Web Search: Search for latest information
                if use_search:
                    with st.spinner("Searching web..."):
                        search_context = search_web(prompt)
                
                # Generate response
                response = get_chat_response(
                    chat_model,
                    st.session_state.messages,
                    system_prompt,
                    rag_context,
                    search_context
                )
                
                st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    """Main application"""
    st.set_page_config(
        page_title="Medical Assistant Chatbot",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Navigation
    with st.sidebar:
        st.title(" Medical Assistant")
        page = st.radio(
            "Navigate:",
            ["Chat", "Instructions"],
            index=0
        )
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()