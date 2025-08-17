# PDF Vector QA

## 简介
- 基于 Qdrant + DeepSeek Embedding + FastAPI 的轻量知识库检索系统
- 支持 PDF 文档分段入库，向量化检索

## 快速开始

1. **环境变量**
   - 复制 `.env.example` 为 `.env`，填写你的 DeepSeek API Key。

2. **构建并启动服务**
   ```bash
   docker-compose up --build

