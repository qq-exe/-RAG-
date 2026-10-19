from pydantic import BaseModel
from typing import List,Optional

#企业级开发必须有严格的数据契约。使用Pydantic定义响应模型，
#FastAPI会自动校验返回的数据格式，同时自动生成Swagger文档中的Schema。
#这样前端能明确知道后端返回的数据长什么样，避免接口联调时候的扯皮。

class UploadResponse(BaseModel):
    msg: str         #提示信息
    chunk_count: int #分块数量

class ChatResponse(BaseModel):
    answer: str             #大模型的最终回答
    references: List[dict]  #溯源信息列表（包含文件名、页码、原文片段）

class HealthResponse(BaseModel):
    status: str     # 健康状态
    time: float     # 当前时间戳