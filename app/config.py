import os

# 解决国内网络访问HuggingFace超时的问题
# 环境变量在导入任何HuggingFace相关库之前设置，否则不生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class Settings:
    """配置类：集中管理所有硬编码参数，避免散落在代码各处，方便后续修改和维护"""
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5" #中文轻量级向量模型，适合CPU运行
    CHROMA_PERSIST_DIR = "./chroma_db"              #向量数据库的持久目录化
    CHUNK_SIZE = 1500                               #调大分块尺寸，避免跨页分开一句话
    CHUNK_OVERLAP = 300                             #分块重叠度，保证上下文连贯
    LLM_MODEL = "qwen2.5:3b"                        #本地Ollama大模型
    TOP_K = 5                                       #最终返回给大模型的文档数量

#实例化一个全局配置对象，其他模块直接'from app.config import settings' 即可使用
settings = Settings()