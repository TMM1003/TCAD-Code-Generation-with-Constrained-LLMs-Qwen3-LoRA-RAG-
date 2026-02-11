import os
import re
import json
import pickle
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import torch
import fitz
import numpy as np
import faiss
import sys

# chunk of text from the manual
@dataclass
class Chunk: 
    text: str
    page_num: int
    section: str
    chunk_id: int
    
    def to_dict(self):
        return asdict(self)


class SilvacoRAG: 
    def __init__(self, pdf_path, embedding_model = "BAAI/bge-small-en-v1.5", chunk_size = 512, chunk_overlap = 50, device = "auto"):
        self.pdf_path = pdf_path
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        
        # will be initialized when needed
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.embedding_model = None
        
    def _load_embedding_model(self):
        if self.embedding_model is None:
            
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
                
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
            print(f"embedding model loaded on: {device}")
            
    def extract_text_from_pdf(self):
        
        if not self.pdf_path:
            raise ValueError("PDF path not provided")
            
        print(f"extracting text from: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        
        # get table of contents for section detection
        toc = doc.get_toc()
        page_to_section = {}
        current_section = "Introduction"
        
        for level, title, page in toc:
            if level <= 2:  # Track chapters and major sections
                current_section = title
            page_to_section[page] = current_section
            
        pages_data = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # find the section for this page
            section = current_section
            for toc_page in sorted(page_to_section.keys()):
                if toc_page <= page_num + 1:
                    section = page_to_section[toc_page]
                else:
                    break
                    
            if text.strip():
                pages_data.append({
                    'text': text,
                    'page_num': page_num + 1,
                    'section': section
                })
                
        doc.close()
        print(f"Extracted {len(pages_data)} pages")
        return pages_data
    
    # remove whitespace and header patterns 
    def _clean_text(self, text):
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'Atlas User\'s Manual\n', '', text)
        text = re.sub(r'^\d+\n', '', text, flags=re.MULTILINE)
        return text.strip()
    
    # keep code samples intact, respect section boundaries when possible
    # fall back to size-based splitting 
    def _chunk_text(self, pages_data):
        chunks = []
        chunk_id = 0
        
        # patterns that indicate code blocks
        code_patterns = [
            r'((?:^|\n)(?:MESH|REGION|ELECTRODE|DOPING|MATERIAL|MODEL|CONTACT|SOLVE|SAVE|EXTRACT|\.SUBCKT|\.MODEL|\.TRAN|\.AC|\.DC).*?)(?=\n\n|\n[A-Z][a-z]|\Z)',
        ]
        
        for page_data in pages_data:
            text = self._clean_text(page_data['text'])
            page_num = page_data['page_num']
            section = page_data['section']
            
            if len(text) < 100:
                continue
                
            # split into paragraphs first
            paragraphs = re.split(r'\n\n+', text)
            
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                para_length = len(para.split())
                
                # check if adding this paragraph exceeds chunk size
                if current_length + para_length > self.chunk_size and current_chunk:
                    # save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        page_num=page_num,
                        section=section,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    
                    # start new chunk with overlap
                    if len(current_chunk) > 1:
                        current_chunk = current_chunk[-1:]  # Keep last paragraph for context
                        current_length = len(current_chunk[0].split())
                    else:
                        current_chunk = []
                        current_length = 0
                        
                current_chunk.append(para)
                current_length += para_length
                
            # add in the last chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    page_num=page_num,
                    section=section,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
                
        print(f"created {len(chunks)} chunks")
        return chunks
    
    # use FAISS to build the vector index 
    def build_index(self):
        
        pages_data = self.extract_text_from_pdf()
        self.chunks = self._chunk_text(pages_data)
        self._load_embedding_model()
        
        # generate embeddings
        texts = [chunk.text for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(texts,show_progress_bar=True, convert_to_numpy=True)
        
        # now build with FAISS 
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"index built with {self.index.ntotal} vectors")
        
    def save_index(self, path):
        os.makedirs(path, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(os.path.join(path, "chunks.json"), 'w') as f:
            json.dump(chunks_data, f)
            
        # Save embeddings
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        
        # save metadata
        metadata = {
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'num_chunks': len(self.chunks)
        }
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
            
        print(f"index saved to: {path}")
        
    def load_index(self, path):
        
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "chunks.json"), 'r') as f:
            chunks_data = json.load(f)
        self.chunks = [Chunk(**c) for c in chunks_data]
        
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        self.embedding_model_name = metadata['embedding_model']
        
        print(f"index loaded: {len(self.chunks)} chunks")
    
    # retrieve relevant chunks based on the query 
    def retrieve(self, query, top_k = 5, min_score = 0.3):
        self._load_embedding_model()
        
        # encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'text': chunk.text,
                    'score': float(score),
                    'page_num': chunk.page_num,
                    'section': chunk.section,
                    'chunk_id': chunk.chunk_id
                })
                
        return results
        
    def format_context(self, results, max_tokens = 1500):
        if not results:
            return ""
            
        context_parts = []
        current_tokens = 0
        
        for r in results:
            # rough token estimate (words * 1.3)
            text_tokens = len(r['text'].split()) * 1.3
            
            if current_tokens + text_tokens > max_tokens:
                break
                
            context_parts.append(
                f"[From {r['section']}, p.{r['page_num']}]\n{r['text']}"
            )
            current_tokens += text_tokens
            
        return "\n\n---\n\n".join(context_parts)
        
    def augment_prompt(self, prompt, top_k = 3, max_context_tokens = 1500, context_position = "before"):
        # extract key terms from prompt for better retrieval
        results = self.retrieve(prompt, top_k=top_k)
        context = self.format_context(results, max_tokens=max_context_tokens)
        
        if not context:
            return prompt
            
        context_block = f"""Reference Documentation:{context}---"""
        
        if context_position == "before":
            return context_block + prompt
        else:
            return prompt + "\n\n" + context_block


if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python silvaco_rag.py build <pdf_path> [index_path]")
        print("  python silvaco_rag.py query <index_path> <query>")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "build":
        pdf_path = sys.argv[2]
        index_path = sys.argv[3] if len(sys.argv) > 3 else "silvaco_index"
        
        rag = SilvacoRAG(pdf_path=pdf_path)
        rag.build_index()
        rag.save_index(index_path)
        
    elif command == "query":
        index_path = sys.argv[2]
        query = " ".join(sys.argv[3:])
        
        rag = SilvacoRAG()
        rag.load_index(index_path)
        
        results = rag.retrieve(query, top_k=3)
        print(f"\nQuery: {query}\n")
        for i, r in enumerate(results):
            print(f"--- Result {i+1} (score: {r['score']:.3f}, p.{r['page_num']}) ---")
            print(r['text'][:500])
            print()
    else:
        print("Please use either build or query as a param!")