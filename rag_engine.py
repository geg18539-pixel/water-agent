# -*- coding: utf-8 -*-
"""
rag_engine.py —— 智能问数引擎（基于 LangChain SQL 链路，LLMChain 模式）

技术栈：
    - 模型：Ollama 本地部署的 qwen2.5:3b（OpenAI 兼容接口）
    - 数据：同目录下的 water_quality.db（SQLite）
      表 monitoring_data 字段：station_name, record_time, ph_value, cod, ammonia_nitrogen
    - 框架：langchain_community + langchain_openai + langchain_classic
      使用 create_sql_query_chain（生成 SQL）+ QuerySQLDataBaseTool（执行 SQL）+ 最终答案合成。

对外函数：
    - query_database(question, history=None) -> str           纯文本答案
    - query_with_chart(question, history=None) -> dict        文本答案 + 图表数据 + 5 分钟缓存

说明：
    - history：最近几轮对话（list[dict]，每项含 question/answer），用于理解“那前郭站呢”等指代。
    - 缓存：相同问题（含历史上下文）5 分钟内直接命中；未装 Redis，用内存 TTL 缓存。

依赖安装（langchain >= 1.0 时必须额外装 langchain-classic）：
    pip install langchain langchain-classic langchain-community langchain-openai httpx sqlalchemy
"""

import os
import time

import httpx
from sqlalchemy import text

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# create_sql_query_chain 的位置随版本变化：
#   - langchain >= 1.0：迁移到了独立的 langchain-classic 包（需 pip install langchain-classic）
#   - langchain 0.x    ：位于 langchain.chains
try:
    from langchain_classic.chains import create_sql_query_chain  # langchain >= 1.0
except ImportError:  # pragma: no cover
    from langchain.chains import create_sql_query_chain  # langchain 0.x


# ==================== 配置区（按需修改） ====================
OLLAMA_BASE_URL = "http://localhost:11434/v1"   # Ollama 的 OpenAI 兼容接口
MODEL_NAME = "qwen2.5:3b"                        # 本地部署的模型名
DB_FILENAME = "water_quality.db"                 # 数据库文件名（默认与脚本同目录）
TOP_K = 5                                        # 默认最多返回条数（写入提示词，作为 LIMIT 建议）

# 触发图表模式的关键词（时间序列 -> 折线，分布/对比 -> 柱状）
CHART_KEYWORDS = ("趋势", "变化", "走势", "分布", "对比", "每月", "每天", "逐日", "逐月",
                  "按天", "按月", "各监测站", "排行", "排名", "随时间")

# “元问题”关键词：与数据库查询无关，直接返回能力说明
CAPABILITY_KEYWORDS = ("功能", "会什么", "能做什么", "能干嘛", "你会", "你能", "有什么能力",
                       "帮助", "怎么用", "使用说明", "你是谁", "介绍一下")

CAPABILITY_ANSWER = (
    "我是水环境智能助手 🌊，可以针对水质监测数据库回答自然语言问题。\n"
    "我能做的包括：\n"
    "· 查询某监测站的指标数据（如 COD、pH、氨氮）\n"
    "· 计算平均值、最高/最低值、超标记录条数等统计\n"
    "· 按时间范围筛选（如某年某月）\n"
    "· 生成趋势折线图与分布柱状图（试试问「松原监测站COD变化趋势」或「各监测站平均COD」）\n"
    "· 记住对话上下文（比如先问某站，再问「那前郭站呢」）\n\n"
    "你可以直接输入问题，例如：查询松原监测站 2026年7月的平均COD。"
)

# 内存缓存：相同问题 5 分钟内直接返回结果（未装 Redis 时的轻量替代）
CACHE_TTL_SECONDS = 300
CACHE_MAX_SIZE = 200

# 数据库绝对路径：优先取脚本所在目录，保证无论从哪里运行都能定位到数据库
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILENAME)
DB_URI = f"sqlite:///{DB_PATH}"

# Ollama 不校验 key，但 ChatOpenAI 强制要求 api_key 非空，填占位符即可
API_KEY = "ollama"


# ==================== Few-shot 示例（SQL 生成提示词） ====================
# 注意：模板必须包含 {dialect}、{top_k}、{table_info}、{input} 四个变量，
# 前两个由 create_sql_query_chain 自动填充，table_info 运行时注入表结构。
SQL_GEN_PROMPT = PromptTemplate.from_template(
    """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.

Unless the user specifies a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. Order the results to return the most informative data in the database.
Never query for all columns from a table. Only query the columns needed to answer the question. Only use the column names you can see in the tables below. Be careful not to query for columns that do not exist.
For time-range questions (e.g. a month), filter the record_time column using string comparison, for example record_time >= '2026-07-01' AND record_time < '2026-08-01'.
For trend/time-series questions, select record_time first, then the metric, and ORDER BY record_time.
For "各监测站" (per-station) distribution questions, GROUP BY station_name.
If the question contains a pronoun like "那...呢" or "再看看..." referring to a previous turn, resolve it using the conversation history provided above the question.

Here are some (Question -> SQLQuery) examples to follow:

Question: 查询松原监测站 2026年7月的平均COD
SQLQuery: SELECT AVG(cod) AS avg_cod FROM monitoring_data WHERE station_name = '松原监测站' AND record_time >= '2026-07-01' AND record_time < '2026-08-01'

Question: 一共有哪些监测站
SQLQuery: SELECT DISTINCT station_name FROM monitoring_data

Question: 松原监测站 2026年7月的最高pH值和最低pH值分别是多少
SQLQuery: SELECT MAX(ph_value) AS max_ph, MIN(ph_value) AS min_ph FROM monitoring_data WHERE station_name = '松原监测站' AND record_time >= '2026-07-01' AND record_time < '2026-08-01'

Question: 松原监测站氨氮浓度超过 0.5 的记录有多少条
SQLQuery: SELECT COUNT(*) AS cnt FROM monitoring_data WHERE station_name = '松原监测站' AND ammonia_nitrogen > 0.5

Question: 松原监测站 COD 的变化趋势
SQLQuery: SELECT record_time, cod FROM monitoring_data WHERE station_name = '松原监测站' ORDER BY record_time

Question: 各监测站的平均COD
SQLQuery: SELECT station_name, AVG(cod) AS avg_cod FROM monitoring_data GROUP BY station_name

Only use the following tables:
{table_info}

Question: {input}
SQLQuery:"""
)


# ==================== 最终答案合成提示词 ====================
ANSWER_PROMPT = PromptTemplate.from_template(
    """根据下面的问题和 SQL 查询结果，用简洁、准确的中文回答。如果结果为空或无数据，请如实说明，不要编造数字。

问题：{question}
SQL 查询：{query}
查询结果：{result}

回答："""
)


def _build_llm() -> ChatOpenAI:
    """构建指向 Ollama 的 LLM。temperature=0 让 SQL 生成更稳定。"""
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        api_key=API_KEY,
        temperature=0,
        # 关键：trust_env=False 禁用系统代理，避免本地 localhost 请求被 VPN/代理
        # 拦截，导致 OpenAIConnectionError: Connection error。
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
    )


def _build_db() -> SQLDatabase:
    """连接 SQLite 数据库。"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"未找到数据库文件：{DB_PATH}，请确认 water_quality.db 与脚本在同一目录。"
        )
    return SQLDatabase.from_uri(DB_URI)


def _extract_sql(text: str) -> str:
    """从模型输出中提取干净的 SQL 语句，兼容不同版本 chain 的返回格式。"""
    text = (text or "").strip()
    if "SQLQuery:" in text:
        text = text.split("SQLQuery:", 1)[1]
    for marker in ("SQLResult:", "\nAnswer:", "\nQuestion:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    sql = text.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("模型未能生成有效的 SQL 语句，请尝试换一种问法。")
    # 校验：必须以合法查询关键字开头，否则是模型生成的自然语言而非 SQL
    first_word = sql.split(None, 1)[0].upper() if sql.split() else ""
    if first_word not in ("SELECT", "WITH", "PRAGMA", "EXPLAIN"):
        raise ValueError("模型没有生成 SQL 查询（可能问题与数据库无关），请换一种问法。")
    return sql


def _run_sql_rows(db: SQLDatabase, sql: str):
    """直接执行 SQL 并返回结构化结果：(列名列表, 行列表)。用于提取图表数据。"""
    engine = getattr(db, "_engine", None) or getattr(db, "engine", None)
    if engine is None:
        raise RuntimeError("无法访问数据库引擎。")
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        rows = result.fetchall()
    return cols, rows


def _is_chart_question(question: str) -> bool:
    return any(k in question for k in CHART_KEYWORDS)


def _is_capability_question(question: str) -> bool:
    return any(k in question for k in CAPABILITY_KEYWORDS)


def _format_rows(rows, limit: int = 30) -> str:
    """把查询结果行格式化为文本，供答案合成使用。"""
    lines = [str(tuple(r)) for r in rows[:limit]]
    return "\n".join(lines) if lines else "（无数据）"


def _build_chart(question: str, cols, rows):
    """把两列多行的查询结果转换为图表数据。不符合则返回 None。"""
    if not rows or len(rows) < 2 or len(cols) < 2:
        return None
    labels, values = [], []
    for r in rows:
        try:
            labels.append(str(r[0]))
            values.append(float(r[1]))
        except (TypeError, ValueError, IndexError):
            return None
    chart_type = "line" if any(k in question for k in ("趋势", "变化", "走势", "时间", "每天", "每月", "逐")) else "bar"
    return {"type": chart_type, "dates": labels, "values": values}


def _with_history(question: str, history):
    """把历史对话拼到当前问题前，帮助模型理解“那前郭站呢”这类指代。"""
    if not history:
        return question
    lines = ["以下是之前几轮对话，用于理解当前问题中的指代（如“那前郭站呢”指上一个站）："]
    for turn in history[-5:]:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            lines.append(f"用户：{q}")
        if a:
            lines.append(f"助手：{a[:200]}")
    lines.append(f"当前问题：{question}")
    return "\n".join(lines)


# ==================== 内存缓存（5 分钟 TTL） ====================
_ANSWER_CACHE = {}  # key -> (timestamp, result)


def _cache_key(question: str, history) -> str:
    h = ""
    if history:
        h = "|".join(f"{t.get('question')}::{str(t.get('answer'))[:60]}" for t in history[-5:])
    return f"{question}{h}"


def _cache_get(key: str):
    entry = _ANSWER_CACHE.get(key)
    if not entry:
        return None
    ts, val = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        _ANSWER_CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: str, val):
    if len(_ANSWER_CACHE) >= CACHE_MAX_SIZE:
        oldest = min(_ANSWER_CACHE, key=lambda k: _ANSWER_CACHE[k][0])
        _ANSWER_CACHE.pop(oldest, None)
    _ANSWER_CACHE[key] = (time.time(), val)


# 单例缓存：避免每次提问都重新加载模型/数据库连接
_LLM = None
_DB = None
_GENERATE_SQL = None
_EXECUTE_SQL = None
_ANSWER_CHAIN = None


def _get_components():
    """懒加载并缓存各链路组件。"""
    global _LLM, _DB, _GENERATE_SQL, _EXECUTE_SQL, _ANSWER_CHAIN
    if _LLM is None:
        _LLM = _build_llm()
        _DB = _build_db()
        _GENERATE_SQL = create_sql_query_chain(_LLM, _DB, prompt=SQL_GEN_PROMPT, k=TOP_K)
        _EXECUTE_SQL = QuerySQLDataBaseTool(db=_DB)
        _ANSWER_CHAIN = ANSWER_PROMPT | _LLM
    return _LLM, _DB, _GENERATE_SQL, _EXECUTE_SQL, _ANSWER_CHAIN


def _synthesize_answer(question: str, sql: str, result_text: str, answer_chain) -> str:
    answer_msg = answer_chain.invoke(
        {"question": question, "query": sql, "result": result_text}
    )
    answer = answer_msg.content if hasattr(answer_msg, "content") else str(answer_msg)
    answer = (answer or "").strip()
    return answer if answer else "已查询到结果，请查看下方图表。"


def query_database(question: str, history=None) -> str:
    """
    对外主函数：接收自然语言问题，返回 AI 生成的答案（纯文本）。

    :param question: 自然语言问数问题
    :param history: 最近几轮对话，用于理解指代
    :return: AI 生成的答案字符串；出错时返回友好的错误提示。
    """
    q = (question or "").strip()
    if not q:
        return "请输入一个有效的问数问题，例如：查询松原监测站 2026年7月的平均COD。"
    if _is_capability_question(q):
        return CAPABILITY_ANSWER

    try:
        _, _, generate_sql, execute_sql, answer_chain = _get_components()

        # 步骤 1：自然语言（含历史上下文）-> SQL
        input_text = _with_history(q, history)
        raw_sql = generate_sql.invoke({"question": input_text})
        sql = _extract_sql(raw_sql)

        # 步骤 2：执行 SQL
        result = execute_sql.invoke(sql)
        result_text = result if isinstance(result, str) else str(result)

        # SQL 执行出错时，不给模型喂错误文本，直接返回友好提示
        if result_text.lstrip().lower().startswith("error"):
            return "抱歉，这次生成的查询没能正确执行，请换个说法再试一次。"

        # 步骤 3：结果 -> 自然语言答案
        return _synthesize_answer(q, sql, result_text, answer_chain)
    except FileNotFoundError as e:
        return f"抱歉，查询出错：{e}"
    except ValueError as e:
        return f"抱歉，查询出错：{e}"
    except Exception as e:  # noqa: BLE001 —— 兜底捕获，保证对调用方友好
        return f"抱歉，查询过程中出现问题：{type(e).__name__}: {e}"


def query_with_chart(question: str, history=None) -> dict:
    """
    对外主函数（带图表 + 缓存）：返回文本答案 + 图表数据。

    :param question: 自然语言问数问题
    :param history: 最近几轮对话，用于理解指代
    :return: {"answer": str, "chart": {"type": "line"/"bar", "dates": [...], "values": [...]} 或 None}
    """
    q = (question or "").strip()
    if not q:
        return {"answer": "请输入一个有效的问数问题，例如：查询松原监测站 2026年7月的平均COD。", "chart": None}
    if _is_capability_question(q):
        return {"answer": CAPABILITY_ANSWER, "chart": None}

    cache_key = _cache_key(q, history)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # 非图表类问题直接走纯文本链路
    if not _is_chart_question(q):
        result = {"answer": query_database(q, history), "chart": None}
        _cache_set(cache_key, result)
        return result

    try:
        _, db, generate_sql, _, answer_chain = _get_components()

        # 1) 生成 SQL（含历史上下文 + 时间序列/分组示例）
        input_text = _with_history(q, history)
        raw_sql = generate_sql.invoke({"question": input_text})
        sql = _extract_sql(raw_sql)

        # 2) 结构化执行，拿到列名与行
        cols, rows = _run_sql_rows(db, sql)

        # 3) 尝试提取图表数据
        chart = _build_chart(q, cols, rows)

        # 4) 合成文本答案
        answer = _synthesize_answer(q, sql, _format_rows(rows), answer_chain)

        result = {"answer": answer, "chart": chart}
        _cache_set(cache_key, result)
        return result
    except Exception:  # noqa: BLE001 —— 图表路径失败时回退到纯文本
        result = {"answer": query_database(q, history), "chart": None}
        _cache_set(cache_key, result)
        return result


# ==================== 简单自测入口 ====================
if __name__ == "__main__":
    demo_questions = [
        "查询松原监测站 2026年7月的平均COD",
        "松原监测站COD变化趋势",
        "各监测站平均COD",
        "你有哪些功能",
    ]
    for q in demo_questions:
        print("=" * 60)
        print(f"[问题] {q}")
        res = query_with_chart(q)
        print(f"[回答] {res['answer']}")
        if res.get("chart"):
            print(f"[图表] type={res['chart']['type']}, 点数={len(res['chart']['values'])}")
        print()
