import streamlit as st
import requests
import os

# 后端 API 地址
API_URL = "http://127.0.0.1:8000"

# 设置网页标题和图标
st.set_page_config(page_title="RAG 智能问答助手", page_icon="🤖", layout="wide")
st.title("🤖 RAG 智能问答助手")

# 🎯 1. 侧边栏：文件上传区
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("上传你的文档 (支持 txt)", type=["txt"])
    
    if st.button("上传并更新知识库") and uploaded_file is not None:
        with st.spinner("正在上传并更新知识库..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                st.success(response.json().get("message", "上传成功"))
            else:
                st.error("上传失败，请检查后端是否启动")
                
    st.divider()
    if st.button("🔄 手动重新加载知识库"):
        with st.spinner("正在重新加载..."):
            response = requests.post(f"{API_URL}/reload")
            if response.status_code == 200:
                st.success("知识库已刷新！")

# 🎯 2. 主界面：聊天记录区域
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果 AI 附带了参考资料，用折叠框展示
        if message["role"] == "assistant" and "docs" in message:
            with st.expander("📚 查看 AI 参考的资料"):
                for doc in message["docs"]:
                    st.info(doc)

# 🎯 3. 底部输入框，等待用户提问
if prompt := st.chat_input("请输入你的问题..."):
    # 把用户的问题展示在界面上
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用后端 API 获取回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={"prompt": prompt})
                if response.status_code == 200:
                    data = response.json()
                    reply = data.get("reply", "抱歉，我没能回答这个问题。")
                    docs = data.get("retrieved_docs", [])
                    
                    st.markdown(reply)
                    if docs:
                        with st.expander("📚 查看 AI 参考的资料"):
                            for doc in docs:
                                st.info(doc)
                    
                    # 保存到聊天记录
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reply, 
                        "docs": docs
                    })
                else:
                    st.error(f"请求失败，状态码: {response.status_code}")
            except Exception as e:
                st.error(f"连接后端失败，请确保后端已启动。错误: {e}")