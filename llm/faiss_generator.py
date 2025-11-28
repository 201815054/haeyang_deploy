
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--type",type=str, required=False, default="xlsx", help="documents type (py, pdf, text, xlsx)",choices=['py','pdf','text', 'jsonl', 'xlsx', 'csv'])
parser.add_argument("--path",type=str, required=False)
parser.add_argument("--out_path", type=str, required=False, default="faiss_munsunduk_qna_db/")
args = parser.parse_args()


documents = []

if args.type == 'py':
    from faiss_document import mun_documents
    documents = mun_documents

elif args.type == 'jsonl':
    from langchain_core.documents import Document
    import ujson
    
    documents = []
    with open('./scripts.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = ujson.loads(line)
            # record["text"] 필드가 본문
            content = record.get("text", "")
            metadata = {
                "id": record.get("id", ""),
                "title": record.get("title", ""),
                "section": record.get("section", ""),
                "aliases": record.get("aliases", []),
                "region": record.get("region", []),
                "period": record.get("period", {}),
                "themes": record.get("themes", []),
                "source": record.get("source", ""),
                "order": record.get("order", 0),
            }
            documents.append(Document(page_content=content, metadata=metadata))
    

elif args.type == 'pdf':
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(args.path)
    pdf_doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.split_documents(pdf_doc)
    
elif args.type == 'text':
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    raw_text = ""
    with open(args.path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    document = Document(page_content=raw_text, metadata={"source": "문순득 표해시말"})
    documents = text_splitter.split_documents([document])

elif args.type == 'xlsx':

    df = pd.read_excel(args.path)

    questions = df.iloc[:, 1].astype(str).str.strip()
    questions = questions[questions != ""]

    documents = [Document(page_content=question) for question in questions]

elif args.type == 'csv':

    df = pd.read_csv(args.path, encoding="utf-8-sig")

    # "질문" 열 추출 및 전처리
    questions = df.iloc[:, 1].astype(str).str.strip()
    questions = questions[questions != ""]

    # LangChain Document 리스트로 변환
    documents = [Document(page_content=question) for question in questions]


# 임베딩 모델 불러오기
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Embedding model using device: {device}")

embedding_model = HuggingFaceEmbeddings(
    # model_name="jhgan/ko-sroberta-multitask",
    model_name = "/home/user/hayang/llm/local_model/BGE-m3-ko",
    # model_name="sentence-transformers/all-MiniLM-L6-v2",
    
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# FAISS 벡터 저장소 생성
vectorstore = FAISS.from_documents(documents, embedding_model)

# 4. 로컬에 저장
vectorstore.save_local(args.out_path)
print("데이터 생성이 완료되엇습니다.")