
# 🤖 MSME Scheme Assistant Chatbot

An AI-powered chatbot built using LLMs, sentence embeddings, and MongoDB to assist Indian MSMEs with discovering relevant government schemes. Also includes support for Telugu translations and PDF-based Q&A.

---

## 🧠 Features

- **Udyam ID-based enterprise profiling**
- **Manual enterprise profile input**
- **LLM-generated search queries based on profile**
- **Embedding-based scheme recommendation**
- **Field-level answers (eligibility, benefits, documents, etc.) using LLM**
- **Telugu translation support via IndicTrans2**
- **Upload your own PDF and query it**
- **Built-in Gradio UI for seamless interaction**

---

## 🛠️ Tech Stack

- 🤗 Transformers: LLMs & Tokenizers
- 🧠 SentenceTransformers: Semantic similarity
- 🗃️ MongoDB: Vector database for profiles, schemes, and uploaded PDFs
- 🧩 LangChain: Prompt templates
- 📚 IndicTrans2: English → Telugu translation
- 🖼️ Gradio: UI interface
- 🧮 PyMuPDF: PDF text extraction

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install pymongo sentence-transformers torch transformers langchain_community pymupdf tools
pip install bitsandbytes scipy accelerate datasets sentencepiece
```

> ⚠️ Run in Google Colab for seamless PDF uploads using `google.colab.files`.

---

### 📁 MongoDB Setup

Ensure your MongoDB contains:

- `udyam_profiles` – MSME profile data
- `schemes_chunks_only` – Chunked scheme embeddings
- `schemes_embedded` – Full scheme info
- `uploaded_pdf_temp` – Temporary collection for uploaded PDFs

Update the `mongo_uri` in the script with your connection string.

---

### 🤖 LLMs Used

- **Gemma-2B-IT** base model + LoRA fine-tuned model: `Vipplav/gemma-finetuned-faq`
- **Embedding model:** `BAAI/bge-small-en-v1.5`
- **Translator model:** `ai4bharat/indictrans2-en-indic-1B`

---

## 🧪 Usage

### 🟢 Launch Chatbot

Run the script to start the Gradio interface.

### 💬 Chat Flow

1. Enter Udyam Registration Number or type `manual`
2. If manual:
   - Fill enterprise details interactively
3. Type `show related schemes`
4. Ask about `eligibility`, `apply`, or `documents`

### 🌐 Translate Response

- Click **🌐 Translate Last Scheme Reply** for Telugu version

---

### 📄 PDF Chat

1. Upload a text-based PDF
2. Click **📄 Enable PDF Chat**
3. Ask questions related to uploaded document
4. Use **🌐 Translate PDF Answer** for Telugu translation

---

## 📦 File Structure

- `final_msme_chatbot.py`: End-to-end chatbot + PDF handler
- `README.md`: You're reading it

---

## 🧠 Logic Highlights

- Profile summarization → LLM prompt → Query
- Query embedding → MongoDB chunk similarity match
- Top schemes retrieved and displayed
- Further questions answered via LLM on stored chunk metadata
- Uploaded PDFs stored temporarily, chunked, and queried using top-k similarity

---

## 📝 Notes

- Only **text-based PDFs** are supported (no scanned images or heavy tables).
- Queries and translations are optimized for **short, user-friendly outputs**.
- You can switch between **scheme-based chat** and **PDF chat** seamlessly.

---

## 📍 TODOs / Improvements

- OCR for scanned PDFs (e.g., Tesseract)
- Streamlit-based deployment option
- Caching for repeated query embedding
- Feedback/rating collection

---

## 📜 License

MIT License
