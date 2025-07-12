
import gradio as gr
import torch
import re
import pymongo
from pymongo import MongoClient
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,
    AutoModelForSeq2SeqLM
)
from sentence_transformers import SentenceTransformer, util
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from peft import PeftModel
import pymupdf
import os

# === MongoDB Setup ===
mongo_uri = "mongodb+srv://vipplavai:pravip2025@cluster0.zcsijsa.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["msme_schemes_db"]
udyam_coll = db["udyam_profiles"]
schemes_chunk_coll = db["schemes_chunks_only"]
schemes_info_coll = db["schemes_embedded"]
query_logs_coll = db["query_logs"]
temp_coll = db["uploaded_pdf_temp"]

# === Device Setup ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# === LLM Setup ===
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("Vipplav/gemma-finetuned-faq", use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", 
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model = PeftModel.from_pretrained(
    base_model, 
    "Vipplav/gemma-finetuned-faq", 
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    max_new_tokens=150, 
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=generator)

# === Embedding Model ===
embed_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device=DEVICE
)

# === IndicTrans2 Setup ===
try:
    from IndicTransToolkit.processor import IndicProcessor
    ip = IndicProcessor(inference=True)
    translator_tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True
    )
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True
    ).to(DEVICE).eval()
    TRANSLATION_AVAILABLE = True
    print("Translation models loaded successfully")
except Exception as e:
    print(f"Translation models not available: {e}")
    TRANSLATION_AVAILABLE = False

def translate_to_telugu(text):
    if not TRANSLATION_AVAILABLE:
        return "⚠️ Translation not available. Please install IndicTransToolkit."
    
    if not text.strip(): 
        return "⚠️ Nothing to translate."
    
    try:
        batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang="tel_Telu")
        inputs = translator_tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = translator_model.generate(**inputs, max_length=256, num_beams=5)
        decoded = translator_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return ip.postprocess_batch(decoded, lang="tel_Telu")[0]
    except Exception as e:
        return f"⚠️ Translation error: {e}"

# === Prompt Template ===
rephrase_template = PromptTemplate.from_template("""
You're a helpful assistant guiding Indian MSMEs to the best-matching government schemes.
Based on the enterprise profile, generate a clear, short one-line search query with keywords like state, sector, size, gender, and investment.
Only return the query. Avoid comments.
Enterprise Profile:
{profile_summary}
""")

# === MSME Utilities ===
def normalize_udyam(uid): 
    return uid.strip().upper().replace(" ", "")

def is_valid_udyam(uid): 
    return bool(re.match(r"^UDYAM-[A-Z]{2}-\d{2}-\d{6,7}$", uid))

def get_profile_by_uid(uid):
    uid = normalize_udyam(uid)
    return udyam_coll.find_one({"Udyam_ID": uid}, {"_id": 0}) if is_valid_udyam(uid) else None

def summarize_profile(p):
    return (
        f"The user represents an enterprise named '{p['Enterprise Name']}', based in {p['State']}, "
        f"operating in the {p['Major Activity']} sector. They identify as {p['Gender']}, run a "
        f"{p['Enterprise Type']} sized {p['Organisation Type'].lower()} organization. The enterprise has "
        f"{p['Employment']} employees, with an investment of ₹{p['Investment Cost (In Rs.)']:,} and a turnover "
        f"of ₹{p['Net Turnover (In Rs.)']:,}."
    )

def generate_search_query(profile):
    summary = summarize_profile(profile)
    q = llm.invoke(rephrase_template.format(profile_summary=summary)).strip().split("\n")[0].strip()
    return q, summary

def get_top_matching_schemes(q, top_k=5):
    qe = embed_model.encode(q, convert_to_tensor=True)
    scores = []
    
    for doc in schemes_chunk_coll.find({"rag_chunks": {"$exists": True}}):
        for chunk in doc["rag_chunks"]:
            if "embedding" in chunk:
                ce = torch.tensor(chunk["embedding"]).to(qe.device)
                score = util.cos_sim(qe, ce)[0][0].item()
                scores.append((score, doc["scheme_id"], doc["scheme_name"]))
    
    seen, out = set(), []
    for score, sid, name in sorted(scores, key=lambda x: x[0], reverse=True):
        if sid not in seen:
            out.append({"score": score, "scheme_id": sid, "scheme_name": name})
            seen.add(sid)
        if len(out) == top_k:
            break
    return out

def fetch_scheme_field_llm(scheme_id, query):
    fmap = {
        "eligibility": "eligibility_list",
        "benefits": "key_benefits_list",
        "assistance": "assistance_list",
        "apply": "how_to_apply_list",
        "documents": "required_documents_list"
    }
    key = next((v for k,v in fmap.items() if k in query.lower()), None)
    doc = schemes_info_coll.find_one({"scheme_id": scheme_id})
    
    if key and doc and key in doc:
        text = "\n".join(doc[key][:5])
        p = (
            f"Summarize for business owners:\nScheme: {doc['scheme_name']}\n"
            f"Section: {key.replace('_list','').title()}\n\n{text}"
        )
        return llm.invoke(p).strip()
    return "❌ Ask eligibility, benefits, how to apply, or documents."

# === Chunk Function ===
def chunk_text(text, chunk_size=350, overlap=50):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# === PDF Processing ===
def process_uploaded_pdf(file):
    if file is None:
        return "⚠️ No file uploaded."
    
    try:
        doc = pymupdf.open(file.name)
        full_text = "\n".join([page.get_text().strip() for page in doc])
        
        if not full_text:
            return "❌ No extractable text found. PDF might be scanned."
        
        chunks = chunk_text(full_text)
        doc_chunks = [
            {
                "chunk_id": i, 
                "chunk_text": c, 
                "embedding": embed_model.encode(c).tolist()
            } 
            for i, c in enumerate(chunks)
        ]
        
        temp_coll.delete_many({})
        temp_coll.insert_one({"source": "user_uploaded", "rag_chunks": doc_chunks})
        
        return f"✅ Successfully processed PDF with {len(doc_chunks)} chunks."
    except Exception as e:
        return f"❌ PDF processing failed: {e}"

# === State Management ===
chat_state = {"stage": 0, "profile": {}, "scheme_id": None, "last_bot_msg": "", "summary": ""}
pdf_state = {"last_pdf_msg": ""}

# === MSME Chatbot Logic ===
def chatbot(msg, history):
    if chat_state["stage"] == 0:
        response = "👋 Enter Udyam ID or type 'manual'."
        chat_state["stage"] = 1

    elif chat_state["stage"] == 1:
        if msg.lower().startswith("udyam-"):
            profile = get_profile_by_uid(msg)
            if profile:
                summary = summarize_profile(profile)
                response = f"✅ Profile loaded:\n{summary}\nType 'show related schemes'."
                chat_state.update({"profile": profile, "stage": 3, "summary": summary})
            else:
                response = "❌ Invalid Udyam ID. Try again or type 'manual'."
        elif "manual" in msg.lower():
            response = "📝 What's your enterprise name?"
            chat_state["stage"] = 2
        else:
            response = "Please enter a valid Udyam ID or 'manual'."

    elif chat_state["stage"] == 2:
        fields = [
            "Enterprise Name", "Gender", "Enterprise Type", "Organisation Type",
            "Major Activity", "State", "Investment Cost (In Rs.)",
            "Net Turnover (In Rs.)", "Employment"
        ]
        idx = len(chat_state["profile"])
        key = fields[idx]
        
        if any(x in key for x in ["Cost", "Turnover", "Employment"]):
            try:
                chat_state["profile"][key] = int(msg)
            except ValueError:
                return "Please enter a valid number."
        else:
            chat_state["profile"][key] = msg
            
        if len(chat_state["profile"]) == len(fields):
            summary = summarize_profile(chat_state["profile"])
            response = f"✅ Profile saved:\n{summary}\nType 'show related schemes'."
            chat_state.update({"stage": 3, "summary": summary})
        else:
            response = f"{fields[idx+1]}?"

    elif chat_state["stage"] == 3 and "scheme" in msg.lower():
        query, _ = generate_search_query(chat_state["profile"])
        results = get_top_matching_schemes(query)
        if not results:
            response = "⚠️ No schemes matched."
        else:
            response = "📈 Recommended Schemes:\n" + "\n".join(
                f"{i+1}. {r['scheme_name']} (Score: {round(r['score'],4)})"
                for i, r in enumerate(results)
            ) + "\nAsk about eligibility, docs, or apply."
            chat_state.update({"scheme_id": results[0]["scheme_id"], "stage": 4})

    elif chat_state["stage"] == 4:
        response = fetch_scheme_field_llm(chat_state["scheme_id"], msg)

    else:
        response = "⚠️ Unexpected state. Please restart."

    chat_state["last_bot_msg"] = response
    return response

# === Translate Last Response ===
def translate_last_response():
    return translate_to_telugu(chat_state["last_bot_msg"])

# === PDF Q&A ===
def query_pdf(question):
    doc = temp_coll.find_one({"source": "user_uploaded"})
    if not doc or "rag_chunks" not in doc:
        pdf_state["last_pdf_msg"] = "⚠️ No PDF chunks found. Please upload a PDF first."
        return pdf_state["last_pdf_msg"]

    qv = embed_model.encode(question, convert_to_tensor=True)
    scored = []
    
    for c in doc["rag_chunks"]:
        if "embedding" in c:
            score = util.cos_sim(qv, torch.tensor(c["embedding"]).to(qv.device))[0][0].item()
            scored.append((score, c["chunk_text"]))

    if not scored:
        pdf_state["last_pdf_msg"] = "⚠️ No embeddings to compare."
        return pdf_state["last_pdf_msg"]

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:3]
    context = "\n---\n".join([t[1] for t in top])

    prompt = f"""You are a knowledgeable assistant. Use the following context to answer the question **briefly**.
**Do not** include the context in your answer—only output the answer itself.

Context:
{context}

Question: {question}

Answer:"""

    full = llm.invoke(prompt)
    answer = full.split("Answer:")[-1].strip()
    pdf_state["last_pdf_msg"] = answer
    return answer

def translate_pdf_response():
    return translate_to_telugu(pdf_state["last_pdf_msg"]) if pdf_state["last_pdf_msg"] else "⚠️ Nothing to translate."

# === Gradio UI ===
def create_interface():
    with gr.Blocks(title="MSME Scheme Assistant") as demo:
        gr.Markdown("## 🤖 MSME Scheme Assistant")
        
        # Main chatbot interface
        chatbot_interface = gr.ChatInterface(
            fn=chatbot, 
            title="💬 MSME Chatbot",
            description="Get personalized government scheme recommendations for your MSME"
        )

        # Translation section
        with gr.Row():
            translate_btn = gr.Button("🌐 Translate Last Scheme Reply")
            translate_output = gr.Textbox(
                label="🗣️ Telugu Translation", 
                lines=3, 
                interactive=False
            )
        
        translate_btn.click(
            fn=translate_last_response, 
            outputs=translate_output
        )

        # PDF section
        gr.Markdown("## 📄 Chat with Your PDF")
        
        with gr.Row():
            pdf_upload = gr.File(
                label="Upload PDF", 
                file_types=[".pdf"]
            )
            pdf_status = gr.Textbox(
                label="Upload Status", 
                interactive=False
            )
        
        pdf_upload.upload(
            fn=process_uploaded_pdf,
            inputs=pdf_upload,
            outputs=pdf_status
        )
        
        with gr.Row():
            pdf_question = gr.Textbox(
                label="Ask a question about your PDF",
                placeholder="What is this document about?"
            )
            pdf_ask_btn = gr.Button("Ask PDF")
        
        pdf_answer = gr.Textbox(
            label="📜 PDF Answer", 
            lines=6, 
            interactive=False
        )
        
        with gr.Row():
            pdf_translate_btn = gr.Button("🌐 Translate PDF Answer")
            pdf_translate_output = gr.Textbox(
                label="🗣️ Telugu PDF Translation", 
                lines=3, 
                interactive=False
            )
        
        pdf_ask_btn.click(
            fn=query_pdf,
            inputs=pdf_question,
            outputs=pdf_answer
        )
        
        pdf_translate_btn.click(
            fn=translate_pdf_response,
            outputs=pdf_translate_output
        )

    return demo

if __name__ == "__main__":
    print("Starting MSME Chatbot...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
        share=False
    )
