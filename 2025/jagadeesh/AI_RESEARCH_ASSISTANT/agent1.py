import os
import json
import requests
import tempfile
from pyzotero import zotero
import fitz  # PyMuPDF
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import GoogleGenerativeAI
import streamlit as st

# Environment variables (replace with your actual keys)
ZOTERO_API_KEY = ""
ZOTERO_LIBRARY_ID = ""
ZOTERO_LIBRARY_TYPE = ""
GOOGLE_API_KEY = ""

# Initialize Zotero API
zot = zotero.Zotero(ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE, ZOTERO_API_KEY)

# Initialize LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as pdf:
            return " ".join(page.get_text() for page in pdf)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
def fetch_pdf_from_zotero():
    items = zot.items()
    if not items:
        return None, None
    pdf_items = []
    for item in items:
        if item["data"].get("itemType") == "attachment":
            continue
        try:
            attachments = zot.children(item["key"])
            if any(att["data"]["contentType"] == "application/pdf" for att in attachments):
                pdf_items.append(item)
        except:
            continue
    if not pdf_items:
        return None, None
    titles = [item["data"]["title"] for item in pdf_items]
    keys = [item["key"] for item in pdf_items]
    return titles, keys

def download_pdf(item_key):
    attachments = zot.children(item_key)
    for attachment in attachments:
        if attachment["data"]["contentType"] == "application/pdf":
            pdf_url = attachment["links"].get("enclosure", {}).get("href")
            if not pdf_url:
                continue
            try:
                response = requests.get(pdf_url, headers={"Zotero-API-Key": ZOTERO_API_KEY}, stream=True)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_pdf.write(chunk)
                    return temp_pdf.name
            except Exception as e:
                st.error(f"Error downloading PDF: {e}")
                return None
    return None

def summarize_text(input_text, llm):
    summarization_prompt = """
    You are an AI assistant tasked with creating concise and informative summaries of research papers...
    [Rest of prompt remains the same as in your code]
    Text to summarize:
    {input_text}
    Summary:
    """
    full_prompt = summarization_prompt.format(input_text=input_text)
    try:
        response = llm.invoke(full_prompt)
        return response.strip()
    except Exception as e:
        return f"❌ Exception occurred during summarization: {str(e)}"

summarization_tool = Tool(
    name="Summarization Tool",
    func=lambda text: summarize_text(text, llm),
    description="Summarizes academic papers concisely using the provided chunk text."
)

summarization_agent = initialize_agent(
    tools=[summarization_tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=False,
    max_iterations=1000,
    max_execution_time=120,
    handle_parsing_errors=True
)

def run_summarization(text):
    return summarization_agent.run(f"Summarize this text chunk using the Summarization Tool: {text}")
