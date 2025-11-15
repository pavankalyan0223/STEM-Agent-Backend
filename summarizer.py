import os
import json
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import OLLAMA_URL, MODEL_NAME, HTTP_TIMEOUT

SUMMARY_DIR = "data/summaries"

os.makedirs(SUMMARY_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_into_sections(text):
    """Heuristically split long documents into smaller sections."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(text)


def summarize_text(text):
    """Use the model to generate a concise summary for a text chunk."""
    prompt = (
        "Summarize the following text into a clear, structured explanation with headings "
        "and bullet points where appropriate. Focus on main ideas and key formulas if any.\n\n"
        f"Text:\n{text}"
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": "You are a helpful summarization assistant."},
                     {"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=HTTP_TIMEOUT)
        res.raise_for_status()
        return res.json()["message"]["content"]
    except Exception as e:
        return f"Error summarizing: {e}"


def summarize_pdf(pdf_path):
    """Summarize each section of a given PDF."""
    text = extract_text_from_pdf(pdf_path)
    sections = split_into_sections(text)
    summaries = []

    for i, section in enumerate(sections):
        print(f"Summarizing section {i+1}/{len(sections)} from {os.path.basename(pdf_path)}...")
        summary = summarize_text(section)
        summaries.append({
            "section": i + 1,
            "summary": summary
        })

    summary_path = os.path.join(
        SUMMARY_DIR, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Summary saved to {summary_path}")
    return summary_path


def summarize_all_pdfs(pdf_dir="data/"):
    """Run summarization for all PDFs in the folder."""
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    for pdf in pdfs:
        summarize_pdf(os.path.join(pdf_dir, pdf))
