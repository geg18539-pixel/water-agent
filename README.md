# 🌊 水环境智能助手

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5--3B-orange.svg)](https://ollama.com)

基于本地大模型 (Qwen2.5) 和 RAG 技术的水环境监测数据智能问答系统。用户通过自然语言提问，系统自动生成 SQL 查询并返回分析结果。

## ✨ 功能特点

- **智能问答**：理解自然语言，自动查询数据库[reference:16]
- **本地部署**：基于 Ollama 本地运行，数据安全
- **可视化界面**：清晰展示查询结果[reference:17]
- **RAG 技术**：提供精准的 Few-shot 示例

## 📸 功能截图

![对话界面](screenshots/chat.png)
*图1：智能问答对话界面*
<img width="2357" height="1431" alt="屏幕截图 2026-09-05 185602" src="https://github.com/user-attachments/assets/969c4f95-1d99-452e-946b-205c908588fa" />

![数据查询](screenshots/query.png)
*图2：监测数据查询结果展示*
<img width="2399" height="1444" alt="屏幕截图 2026-09-05 184707" src="https://github.com/user-attachments/assets/db4d186c-790c-45c7-b1ee-93f50f666f9e" />
<img width="2391" height="1437" alt="屏幕截图 2026-09-05 184621" src="https://github.com/user-attachments/assets/e9164a8f-9128-4a79-a2c6-511552d74b23" />

## 🛠️ 技术栈

- **后端**: Python, FastAPI, LangChain
- **大模型**: Ollama, Qwen2.5:3b
- **前端**: Vue 3 (CDN), HTML5, CSS3
- **数据库**: SQLite

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Ollama
- Git

### 安装与运行

1.  **克隆项目**
    ```bash
    git clone https://github.com/你的用户名/water-agent.git
    cd water-agent
    ```

2.  **安装依赖**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # 或 .\venv\Scripts\activate  # Windows
    pip install -r requirements.txt
    ```

3.  **启动 Ollama 服务**
    ```bash
    ollama serve
    ```

4.  **初始化数据库**
    ```bash
    python db_init.py
    ```

5.  **启动应用**
    ```bash
    uvicorn main:app --reload
    ```
    打开浏览器访问 `http://localhost:8000` 即可。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

## 📄 许可证

[MIT](LICENSE)
