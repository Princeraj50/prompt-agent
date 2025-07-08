def build_team_retriever(domain_doc_path: str, team_id: str):
    import os
    import docx
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.schema import Document
    from langchain.vectorstores import FAISS
    from langchain.embeddings import AzureOpenAIEmbeddings

    def extract_docx_content(path):
        doc = docx.Document(path)
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        tables = []
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    tables.append(" | ".join(cells))
        return "\n".join(paras + tables)

    domain_text = extract_docx_content(domain_doc_path)
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(domain_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embedder = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_endpoint="https://your-resource.openai.azure.com/",
        api_key="your-azure-api-key",
        api_version="2023-07-01-preview"
    )

    retriever_dir = f"{team_id}_retriever"
    db = FAISS.from_documents(docs, embedder)
    db.save_local(retriever_dir)
    print(f"✅ Saved retriever for {team_id} at: {retriever_dir}/index.faiss")





step2:


def classify_reports_by_domain(pdf_dir: str, team_id: str):
    import os
    import fitz
    import logging
    from typing import TypedDict, Dict
    from langchain.vectorstores import FAISS
    from langchain.embeddings import AzureOpenAIEmbeddings
    from langchain.llms import AzureOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    class DomainClassification(TypedDict):
        domain: str
        subdomain: str

    logging.basicConfig(
        filename=f"{team_id}_classification.log",
        filemode="a",
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    results: Dict[str, DomainClassification] = {}
    retriever_path = f"{team_id}_retriever"

    def extract_pdf_text(path: str) -> str:
        try:
            pdf = fitz.open(path)
            return "\n".join([page.get_text() for page in pdf])
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            raise

    try:
        embedder = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint="https://your-resource.openai.azure.com/",
            api_key="your-azure-api-key",
            api_version="2023-07-01-preview"
        )
        vectorstore = FAISS.load_local(retriever_path, embedder)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = AzureOpenAI(
            deployment_name="your-llm-deployment-name",
            azure_endpoint="https://your-resource.openai.azure.com/",
            api_key="your-azure-api-key",
            api_version="2023-07-01-preview",
            temperature=0.1
        )

        prompt = PromptTemplate.from_template("""
You are given a client report and a domain dictionary.

Classify the report by extracting the most relevant domain and subdomain.

Return only this JSON:
{
  "domain": "<domain>",
  "subdomain": "<subdomain>"
}

Domain Dictionary:
{context}

Client Report:
{question}
""")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

    except Exception as setup_error:
        logger.critical(f"{team_id} setup failed: {setup_error}")
        raise RuntimeError("Failed to initialize retriever")

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            try:
                logger.info(f"{team_id} → Classifying: {filename}")
                text = extract_pdf_text(os.path.join(pdf_dir, filename))
                result = qa_chain.run(text)
                results[filename] = result
                logger.info(f"{filename} → {result}")
            except Exception as e:
                logger.error(f"❌ Error on {filename}: {e}")
                results[filename] = {
                    "domain": "Unclassified",
                    "subdomain": f"Error: {str(e)}"
                }

    return results




