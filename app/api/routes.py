import uuid
import os
import time
import re
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool #[核心]引入线程池，解决同步阻塞
import ollama
from app.core.rag_service import rag_service
from app.core.retriever import get_retrieval_context
from app.schemas import UploadResponse, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
     # ✅ 企业级修复：使用 uuid 生成安全的临时文件名，保留原后缀
    file_extension = os.path.splitext(file.filename)[1] or ".pdf"
    temp_path = f"./tmp_{uuid.uuid4().hex}{file_extension}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        # 【工程考量】PyMuPDFLoader 和 Chroma 都是同步阻塞操作，
        # 必须用run_in_threadpool 丢进线程池，否则会卡死FastAPI的事件循环，导致并发能力极差。
        chunk_count = await run_in_threadpool(rag_service.ingest_pdf, temp_path)
        return UploadResponse(msg="文档上传成功", chunk_count=chunk_count)
    finally:
        #无论成功与否，都删除临时文件，防止磁盘爆满
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/rag_chat", response_model=ChatResponse)
async def rag_chat(question: str):
    """RAG问答接口：检索+生成+溯源"""
    if not rag_service.vector_db:
        raise HTTPException(status_code=400, detail="知识库为空，请先上传PDF!")
    logger.info(f"用户提问：{question}")
    # 检索召回
    raw_docs = await run_in_threadpool(get_retrieval_context, question, rag_service)
    context_text = "\n\n".join([d.page_content for d in raw_docs])

    # [Prompt优化] 软拒答引导+禁止编造+禁止重复输出
    prompt = f"""你是一个企业知识库助手。请严格基于下面提供的知识库内容回答用户问题。
    如果找不到答案，请说明没有相关信息，并总结相关主题内容，不要编造信息，禁止重复输出。

知识库内容:
{context_text}

用户问题：{question}
"""

    # [参数调优]调用本地Ollama模型，加入repeat_penalty 解决“复读机”问题
    resp = await run_in_threadpool(
        ollama.chat,
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "repeat_penalty": 1.15, "num_predict": 300}
    )

    # 【溯源信息组装】从文本中正则提取页码，只返回前2条最相关的，并截断文本防止前端臃肿
    references = []
    for doc in raw_docs[:2]:
        content = doc.page_content
        print(f"正在检查的内容是: {content[:50]}")
        page_match = re.search(r'【第(\d+)页】', content) 
        page_num = page_match.group(0).replace("【第", "").replace("页】", "") if page_match else "未知"
        references.append({
            "soure_file": doc.metadata.get("source", "").split("/")[-1], #只保留文件名
            "page_number": page_num,
            "chunk_text": content[:200] + "..." #截取前200字，避免返回过长的文本
        })
    
    return ChatResponse(answer=resp["message"]["content"], references=references)

@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口，用于运维监控"""
    return HealthResponse(status="running", time=time.time())