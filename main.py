import os
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import UploadFile, File
import shutil

# 加载环境变量
load_dotenv()

app = FastAPI()

# 1. 初始化大模型客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://ly.ai3322.site/v1" 
)

# 2. 初始化 RAG 向量数据库
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="my_knowledge", embedding_function=embedding_fn)

# 3. 启动时，重新读取 data 文件夹里的最新文档（确保资料是最新的）
def init_rag_database():
    documents = []
    metadatas = []
    ids = []
    
    if not os.path.exists(DATA_DIR):
        print("⚠️ 警告：data 文件夹不存在，请先创建！")
        return
        
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = [chunk.strip() for chunk in content.split("\n") if chunk.strip()]
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk_index": i})
                ids.append(f"{filename}_{i}")
                
    # 清空旧数据，重新写入
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
        
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"✅ RAG 知识库初始化完成，共加载 {len(documents)} 个文本块。")

# 服务启动时自动执行一次
init_rag_database()

# 定义请求的数据结构
class ChatRequest(BaseModel):
    prompt: str

# 根路径测试
@app.get("/")
def read_root():
    return {"message": "Welcome to RAG Doc QA API!"}

# 聊天接口 (接入 RAG)
@app.post("/chat")
def chat_with_llm(request: ChatRequest):
    user_question = request.prompt
    print(f"\n🔍 用户提问: {user_question}")

    # 🎯 第一步：去向量数据库里检索相关文档
    results = collection.query(
        query_texts=[user_question],
        n_results=2  # 找最相关的 2 段话
    )
    
    retrieved_docs = results['documents'][0] if results['documents'] else []
    
    # 🎯 第二步：把检索到的资料拼接成上下文
    context = "\n".join(retrieved_docs)
    print(f"📚 检索到的参考资料:\n{context}")

    # 🎯 第三步：构造包含上下文的 Prompt
    # 关键：要求大模型必须根据资料回答，不要瞎编
    system_prompt = f"""你是一个专业的AI助手。
请严格根据以下参考资料回答用户的问题。如果参考资料中没有相关信息，请直接说“根据提供的资料，我无法回答这个问题”，不要自己编造。

参考资料：
{context}
"""

    try:
        # 🎯 第四步：把资料和问题一起发给大模型
        response = client.chat.completions.create(
            model="gpt-5.5", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]
        )
        reply = response.choices[0].message.content
        return {"reply": reply, "retrieved_docs": retrieved_docs}
    except Exception as e:
        print(f"\n🚨 具体报错详情: {str(e)}\n")
        return {"error": f"调用大模型失败，原因: {str(e)}"}
    # 🎯 新增：文件上传接口
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 确保 data 目录存在
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, file.filename)
        
        # 保存上传的文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 上传后自动重建知识库
        init_rag_database()
        
        return {"message": f"文件 {file.filename} 上传成功，知识库已更新！"}
    except Exception as e:
        return {"error": str(e)}

# 🎯 新增：手动重建知识库接口
@app.post("/reload")
def reload_knowledge_base():
    init_rag_database()
    return {"message": "知识库已重新加载！"}