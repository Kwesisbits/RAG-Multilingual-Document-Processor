from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
import gradio as gr

# ==================== GLOBAL MODEL CACHE ====================
# Loads models once and reuses them

_embedding_model = None
_qa_llm = None
_translation_tokenizer = None
_translation_model = None

# Language code mapping for mBART
LANG_CODES = {
    "en": "en_XX",
    "fr": "fr_XX",
    "es": "es_XX",
    "de": "de_DE",
    "zh": "zh_CN",
    "ja": "ja_XX",
    "it": "it_IT",
    "pt": "pt_XX",
    "ru": "ru_RU",
    "ar": "ar_AR",
    "hi": "hi_IN",
    "ko": "ko_KR"
}

# ==================== MODEL SETUP ====================

def get_embeddings():
    """Initializes HuggingFace embeddings model (cached)"""
    global _embedding_model
    
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        print("✅ Embeddings loaded")
    
    return _embedding_model

def get_qa_llm():
    """Initialize QA model (cached)"""
    global _qa_llm
    
    if _qa_llm is None:
        print("Loading QA model...")
        model_name = "google/flan-t5-base"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        _qa_llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ QA model loaded")
    
    return _qa_llm

def get_translation_model():
    """Initialize translation model (cached) - FIXED for HF Spaces"""
    global _translation_tokenizer, _translation_model
    
    if _translation_tokenizer is None:
        print("Loading translation model...")
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        
        
        _translation_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False 
        )
        
        _translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print("✅ Translation model loaded")
    
    return _translation_tokenizer, _translation_model

# ==================== CORE FUNCTIONS ====================

def document_loader(file):
    """Load PDF document"""
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

def text_splitter(text):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_documents(text)
    return chunks

def vector_database(chunks):
    """Create vector database from chunks"""
    embedding_model = get_embeddings()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

def translate_text(text, target_language="fr"):
    """Translate text using mBART"""
    try:
        tokenizer, model = get_translation_model()
        
        # Set source and target languages
        tokenizer.src_lang = "en_XX"
        target_lang_code = LANG_CODES.get(target_language, "en_XX")
        
        # Tokenizes with proper settings
        encoded = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True,
            padding=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
            model = model.cuda()
        
        # Generates translation
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=1024,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
    
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return f"Translation failed: {str(e)}"

def retriever_qa(file, query=None, target_language=None, full_document=False):
    """Main RAG function"""
    try:
        # Load and process document
        print(f"Loading document: {file}")
        text = document_loader(file)
        chunks = text_splitter(text)
        print(f"Split into {len(chunks)} chunks")
        
        # Full document translation mode
        if full_document and target_language:
            print(f"Translating full document to {target_language}...")
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"Translating chunk {i+1}/{len(chunks)}...")
                translated = translate_text(chunk.page_content, target_language)
                translated_chunks.append(translated)
            
            result = "\n\n".join(translated_chunks)
            print("Translation complete!")
            return result
        
        # QA mode
        print("Creating vector database...")
        vectordb = vector_database(chunks)
        retriever_obj = vectordb.as_retriever(search_kwargs={"k": 3})
        
        print("Getting QA model...")
        llm = get_qa_llm()
        
        print("Running QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=False
        )
        
        response = qa_chain.invoke(query)["result"]
        print("QA complete!")
        
        # Translate response if requested
        if target_language and target_language != "en":
            print(f"Translating answer to {target_language}...")
            response = translate_text(response, target_language)
        
        return response
    
    except Exception as e:
        error_msg = f"Error in retriever_qa: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

# ==================== GRADIO INTERFACE ====================

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.contain {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 24px !important;
    padding: 2.5rem !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem !important;
    text-align: center;
}

.gradio-container p {
    text-align: center;
    color: #4a5568 !important;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.gr-file, .gr-textbox, .gr-dropdown, .gr-radio {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.gr-file:hover, .gr-textbox:focus, .gr-dropdown:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

label {
    color: #2d3748 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

.gr-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

footer {
    display: none !important;
}

@media (max-width: 768px) {
    .contain {
        padding: 1.5rem !important;
    }
    
    h1 {
        font-size: 2rem !important;
    }
}
"""

def app_interface(file, query, mode, target_language):
    """Wrapper function for Gradio interface"""
    if file is None:
        return "⚠️ Please upload a PDF file first."
    
    try:
        full_doc = mode == "Full Document Translation"
        
        if not full_doc and not query:
            return "⚠️ Please enter a question in QA mode."
        
        response = retriever_qa(
            file=file.name,
            query=query if not full_doc else None,
            target_language=target_language if target_language != "en" else None,
            full_document=full_doc
        )
        
        return response
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ Error: {str(e)}\n\nDetails:\n{error_detail}\n\nPlease try again or check your PDF file."

# ==================== GRADIO APP ====================

with gr.Blocks(css=css, theme=gr.themes.Soft()) as rag_app:
    gr.Markdown(
        """
        # 🤖 Multilingual RAG Document Processor
        Upload PDFs, ask questions, or translate documents into 12+ languages using AI.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📄 Upload PDF Document",
                file_types=[".pdf"],
                type="filepath"
            )
            
            mode_input = gr.Radio(
                choices=["Context-Aware QA", "Full Document Translation"],
                label="🎯 Select Mode",
                value="Context-Aware QA"
            )
            
            query_input = gr.Textbox(
                label="❓ Your Question",
                placeholder="What is this document about?",
                lines=3
            )
            
            lang_input = gr.Dropdown(
                choices=[
                    ("English", "en"),
                    ("French", "fr"),
                    ("Spanish", "es"),
                    ("German", "de"),
                    ("Chinese", "zh"),
                    ("Japanese", "ja"),
                    ("Italian", "it"),
                    ("Portuguese", "pt"),
                    ("Russian", "ru"),
                    ("Arabic", "ar"),
                    ("Hindi", "hi"),
                    ("Korean", "ko")
                ],
                value="en",
                label="🌍 Target Language"
            )
            
            submit_btn = gr.Button("🚀 Process Document", variant="primary")
        
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="📝 Response",
                lines=20,
                show_copy_button=True
            )
    
    gr.Markdown(
        """
        ### 💡 How to Use:
        1. **Upload** your PDF document
        2. **Choose mode**: Ask questions or translate entire document
        3. **Select language**: Choose your target language
        4. **Enter question** (for QA mode) or leave blank (for translation)
        5. Click **Process Document**
        
        ### ⚡ Powered by:
        - 🧠 HuggingFace Transformers (Flan-T5 + mBART)
        - 🔍 LangChain RAG Pipeline
        - 📊 ChromaDB Vector Store
        
        ### ⏱️ Note:
        - First translation loads the model (may take 30-60 seconds)
        - Subsequent translations are faster
        - QA mode: 10-30 seconds per query
        """
    )
    
    submit_btn.click(
        fn=app_interface,
        inputs=[file_input, query_input, mode_input, lang_input],
        outputs=output
    )

# ==================== LAUNCH ====================

rag_app.queue()
print("🚀 Starting RAG Application...")
print(f"🔧 Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
rag_app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)