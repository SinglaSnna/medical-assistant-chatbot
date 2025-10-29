import os
import sys
import faiss
import numpy as np
from PyPDF2 import PdfReader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_RAG_CHUNKS


class RAGSystem:
    """Retrieval-Augmented Generation system for document search"""
    
    def __init__(self, embedding_model):
        """Initialize RAG system with embedding model"""
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        self.metadatas = []
    
    def extract_text_from_pdf(self, pdf_file):
        """
        Extract text from PDF file
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Extracted text as string
        """
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"✗ Error reading PDF: {e}")
            return ""
    
    def split_text_into_chunks(self, text, chunk_size=None, overlap=None):
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to split
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or CHUNK_SIZE
        overlap = overlap or CHUNK_OVERLAP
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            
            start += (chunk_size - overlap)
        
        return chunks
    
    def process_documents(self, uploaded_files):
        """
        Process uploaded documents and create chunks
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            List of document chunks
        """
        all_chunks = []
        
        for file in uploaded_files:
            try:
                file_extension = file.name.split('.')[-1].lower()
                
                if file_extension == 'pdf':
                    text = self.extract_text_from_pdf(file)
                elif file_extension == 'txt':
                    text = file.read().decode('utf-8')
                else:
                    print(f"Unsupported file type: {file_extension}")
                    continue
                
                if text:
                    chunks = self.split_text_into_chunks(text)
                    all_chunks.extend(chunks)
                    print(f"✓ Processed {file.name}: {len(chunks)} chunks")
            
            except Exception as e:
                print(f"✗ Error processing {file.name}: {e}")
        
        return all_chunks
    
    def build_index(self, documents):
        """
        Build FAISS index from documents
        
        Args:
            documents: List of text chunks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                print("✗ No documents to index")
                return False
            
            print(f"Building index for {len(documents)} documents...")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            self.documents = documents
            
            print(f"✓ Index built successfully with {len(documents)} chunks")
            return True
        
        except Exception as e:
            print(f"✗ Error building index: {e}")
            return False
    
    def retrieve(self, query, k=None):
        """
        Retrieve most relevant documents for a query
        
        Args:
            query: Search query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant document chunks
        """
        k = k or MAX_RAG_CHUNKS
        
        try:
            if self.index is None or len(self.documents) == 0:
                return []
            
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
            
            # Get relevant documents
            results = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    results.append(self.documents[idx])
            
            return results
        
        except Exception as e:
            print(f"✗ Error retrieving documents: {e}")
            return []
    
    def is_ready(self):
        """Check if RAG system is ready to use"""
        return self.index is not None and len(self.documents) > 0