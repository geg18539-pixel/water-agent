# -*- coding: utf-8 -*-
"""
main.py —— FastAPI 后端服务

基于 rag_engine.py 提供智能问数接口（文本答案 + 图表数据 + 会话记忆）。

启动方式：
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
或直接运行：
    python main.py

接口：
    GET  /      -> {"status": "ok"}
    POST /ask   -> 入参 {"question": "...", "session_id": "..."}
                   返回 {"answer": "...", "chart": {type, dates, values} | null}

会话记忆：
    每个 session_id 用内存字典缓存最近 5 轮对话，用于理解“那前郭站呢”这类指代。

依赖安装：
    pip install fastapi uvicorn pydantic
    （另需 rag_engine.py 的依赖：langchain langchain-classic langchain-community langchain-openai httpx sqlalchemy）
"""

from typing import Optional

from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_engine import query_with_chart


app = FastAPI(title="智能问数引擎", description="基于 LangChain SQL 链路的水质数据问答服务", version="1.0.0")

# ==================== CORS 跨域配置 ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 会话记忆（内存缓存最近 5 轮） ====================
MAX_HISTORY = 5
SESSIONS = {}  # session_id -> [{"question": str, "answer": str}, ...]


def get_history(session_id: str):
    return SESSIONS.get(session_id, [])


def append_history(session_id: str, question: str, answer: str):
    history = SESSIONS.get(session_id, [])
    history.append({"question": question, "answer": answer})
    SESSIONS[session_id] = history[-MAX_HISTORY:]  # 只保留最近 5 轮


# ==================== 请求/响应模型 ====================
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    chart: Optional[dict] = None


# ==================== 路由 ====================
@app.get("/")
def root():
    """根路径健康检查。"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """接收自然语言问题（带会话），返回 AI 生成的答案及可选图表数据。"""
    sid = req.session_id or "default"
    history = get_history(sid)

    result = query_with_chart(req.question, history)

    # 记录本轮对话，供后续指代理解
    append_history(sid, req.question, result["answer"])

    return {"answer": result["answer"], "chart": result.get("chart")}


# ==================== 直接运行入口 ====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
