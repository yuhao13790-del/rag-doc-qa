import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = FastAPI()

# 初始化大模型客户端 (这里以 DeepSeek 为例，如果没有API Key可以去官网申请一个)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://ly.ai3322.site/v1" 
)

# 定义请求的数据结构
class ChatRequest(BaseModel):
    prompt: str

# 根路径测试
@app.get("/")
def read_root():
    return {"message": "Welcome to RAG Doc QA API!"}

# 聊天接口
@app.post("/chat")
def chat_with_llm(request: ChatRequest):
    try:
        # 尝试调用大模型
        response = client.chat.completions.create(
            model="gpt-5.5", 
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手。"},
                {"role": "user", "content": request.prompt}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        # 如果报错，把具体的错误抓出来，打印在终端里
        print(f"\n🚨 具体报错详情: {str(e)}\n")
        return {"error": f"调用大模型失败，原因: {str(e)}"}