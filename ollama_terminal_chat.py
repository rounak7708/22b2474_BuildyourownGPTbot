from PyPDF2 import PdfReader
import requests
import json

url = "http://localhost:11434/api/generate"

def read(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_relevant_chunk(pdf_text, question, max_chars=2000):
    keywords = question.lower().split()
    paragraphs = pdf_text.split("\n\n")
    relevant = [p for p in paragraphs if any(word in p.lower() for word in keywords)]
    combined = "\n\n".join(relevant)
    return combined[:max_chars] 
    
pdf = read("C:\\Users\\Gargi Joshi\\Build-your-own-GPT-bot\\week_8\\Indian Laws Overview.pdf")



user_input = input("Hi, how can I help you?\n")

chunk = get_relevant_chunk(pdf, user_input)

prompt = f"""You are an assistant that only answers based on the PDF content below.
If the answer is not in the PDF, respond only with 'IDK'.

PDF Content:
{chunk}

User Question:
{user_input}
"""

payload = {
    "model": "tinyllama",
    "prompt": prompt,
    "stream": False
}

response = requests.post(url, json=payload)
data = response.json()

print("Response from Ollama:")
if "response" in data and data["response"]:
    print(data["response"])
elif "message" in data and "content" in data["message"]:
    print(data["message"]["content"])
else:
    print("IDK")
    
    
