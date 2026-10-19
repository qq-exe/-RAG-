import re
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings

logger = logging.getLogger(__name__)

#全局预加载Embedding模型，避免每次请求加载，极大节省时间
# 使用HuggingFaceEmbeddings替代弃用的HuggingFaceBgeEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": "True"},
)

class RAGService:
    """RAG核心服务类：封装文档解析、清洗、切分、入库逻辑"""
    def __init__(self):
        # 用类属性代替全局变量，避免多线程下的状态污染
        self.vector_db = None
        self.bm25_retriever = None

    def _clean_text(self, text: str) -> str:
        # PDF解析出的数据含有大量无关内容，清晰避免污染向量空间
        # 过滤HTML实体，PDF排版残留
        text = text.replace("\u003c","<").replace("\u003e",">")
        # 过滤【1082】这种学术参考文献编号
        text = re.sub(r'\[\d+\]', '', text)
        # 过滤URL链接
        text = re.sub(r'https?://\S+', '', text)
        return text
    
    def ingest_pdf(self, file_path: str) -> int:
        # 解析、清洗、切分并存入向量库
        loader = PyMuPDFLoader(file_path) # 替代PyPDFLoader,中文解析不乱码
        docs = loader.load()

        # 重点：跨页截断解决方案：合并所有页面，在每页开头注入【第N页】标记
        # 解决了物理分页切断句子的痛点，保留了页码溯源能力
        combined_text = ""
        for idx, doc in enumerate(docs):
            cleaned = self._clean_text(doc.page_content)
            combined_text += f"【第{idx+1}页】 {cleaned}\n"

        # 将所有页面封装成一个大的Document，保留源文件信息
        combined_doc = Document(
            page_content=combined_text,
            metadata={"source": docs[0].metadata.get("source", "未知文件")}
        )

        #针对中文优化切分策略：增大chunk_size和overLap,加入中文标点分隔符
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。",  "，", "！", "？", "", " "]
        )
        splits = splitter.split_documents([combined_doc])
        # 过滤掉切分后太短的无效块（如文献参考的残渣）
        splits = [s for s in splits if len(s.page_content.strip()) > 50]

        # 向量化存入Chroma
        self.vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=settings.CHROMA_PERSIST_DIR  
        )

        # 构建BM25关键词检索器
        self.bm25_retriever = BM25Retriever.from_documents(splits)
        self.bm25_retriever.k = settings.TOP_K

        return len(splits)

# 单条实例，全局唯一，避免重复初始化
rag_service = RAGService()
