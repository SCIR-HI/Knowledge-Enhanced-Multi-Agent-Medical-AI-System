from utils import determine_difficulty, Agent
from md_agent import process_diff_query
from one_agent import process_base_query
from med_agent import process_mid_query
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import threading
from retriever import Retriever
import queue
import time
from typing import Generator, Dict, Any
from openai import OpenAI
from config import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
    DEFAULT_FAISS_VERSION,
    GENERATION_CONFIG_BASE,
    MAX_CONVERSATION_HISTORY,
    MAX_CONVERSATION_TURNS,
    MODEL_NAME,
    OPENAI_API_KEY,
    SERVE_URL,
    SERVER_HOST,
    SERVER_PORT,
    STREAM_RETRIEVER_MIN_SCORE,
)

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

client = OpenAI(
    base_url=SERVE_URL,
    api_key=OPENAI_API_KEY,
)
retrieverr = Retriever(DEFAULT_FAISS_VERSION, min_score=STREAM_RETRIEVER_MIN_SCORE)
# ✨ 新增：全局消息队列，用于收集中间过程
message_queues = {}
conversation_history = {}

def get_conversation_context(session_id: str, max_turns: int = MAX_CONVERSATION_TURNS) -> str:
    """
    获取对话上下文，只包含最近几轮的用户问题和最终答案
    Args:
        session_id: 会话ID
        max_turns: 最大保留的对话轮数
    Returns:
        格式化的对话上下文字符串
    """
    if session_id not in conversation_history:
        return ""
    
    history = conversation_history[session_id]
    # 只取最近的max_turns轮对话
    recent_history = history[-max_turns:] if len(history) > max_turns else history
    
    if not recent_history:
        return ""
    
    context = "以下是之前的对话历史：\n"
    for i, turn in enumerate(recent_history, 1):
        context += f"用户问题: {turn['question']}\n"
        context += f"助手回答: {turn['answer']}\n"
    
    context += "\n请基于以上对话历史回答当前问题。\n\n"
    return context

def save_conversation_turn(session_id: str, question: str, answer: str):
    """
    保存一轮对话
    Args:
        session_id: 会话ID
        question: 用户问题
        answer: 最终答案
    """
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({
        "question": question,
        "answer": answer,
        "timestamp": int(time.time())
    })
    
    # 可选：限制历史记录长度，避免内存无限增长
    max_history_length = MAX_CONVERSATION_HISTORY
    if len(conversation_history[session_id]) > max_history_length:
        conversation_history[session_id] = conversation_history[session_id][-max_history_length:]

def create_question_with_context(question: str, session_id: str) -> str:
    """
    将当前问题与历史上下文结合
    Args:
        question: 当前用户问题
        session_id: 会话ID
    Returns:
        包含上下文的完整问题
    """
    context = get_conversation_context(session_id)
    if context:
        return f"{context}当前问题: {question}"
    else:
        return question
generation_config_base = GENERATION_CONFIG_BASE.copy()
generation_config_base["model"] = MODEL_NAME

# ✨ 新增：流式输出辅助类
class StreamHelper:
    @staticmethod
    def format_sse(data: Dict[str, Any]) -> str:
        """格式化SSE数据"""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    @staticmethod
    def send_step(session_id: str, agent_name: str, description: str, details: list = None):
        """发送执行步骤到队列"""
        if session_id not in message_queues:
            return
            
        step_data = {
            "type": "agent_step",
            "step": {
                "agent": agent_name,
                "description": description,
                "details": details or [],
                "status": "processing",
                "timestamp": int(time.time() * 1000)
            }
        }
        message_queues[session_id].put(step_data)
    
    @staticmethod
    def send_output(session_id: str, agent_name: str, content: str, status: str = "completed"):
        """发送智能体输出到队列"""
        if session_id not in message_queues:
            return
            
        output_data = {
            "type": "agent_output",
            "output": {
                "agentName": agent_name,
                "content": content,
                "status": status,
                "isIncremental": False
            }
        }
        message_queues[session_id].put(output_data)
    
    @staticmethod
    def send_final(session_id: str, content: str, query: str):
        """发送最终结果"""
        if session_id not in message_queues:
            return
            
        final_data = {
            "type": "final_result",
            "content": content,
            "originalQuery": query
        }
        message_queues[session_id].put(final_data)
    
    @staticmethod
    def send_complete(session_id: str, query: str):
        """发送完成信号"""
        if session_id not in message_queues:
            return
            
        complete_data = {
            "type": "complete",
            "originalQuery": query,
            "timestamp": int(time.time() * 1000)
        }
        message_queues[session_id].put(complete_data)

# ✨ 新增：包装函数，在调用原函数前后添加流式输出
def create_callback(session_id: str, default_agent_name: str):
    """为特定会话创建回调函数，支持动态智能体名称"""
    def callback(step_type: str, content: str, agent_name: str = None, details: list = None):
        """
        增强的回调函数参数：
        - step_type: 'step' 或 'output' 或 'incremental'
        - content: 步骤描述或输出内容
        - agent_name: 可选，指定智能体名称，如果不提供则使用默认名称
        - details: 可选的详细信息列表
        """
        # 如果没有指定 agent_name，使用默认名称
        current_agent_name = agent_name or default_agent_name
        
        if step_type == 'step':
            StreamHelper.send_step(session_id, current_agent_name, content, details)
        elif step_type == 'output':
            StreamHelper.send_output(session_id, current_agent_name, content)
        elif step_type == 'incremental':
            # 增量输出
            output_data = {
                "type": "agent_output",
                "output": {
                    "agentName": current_agent_name,
                    "content": content,
                    "status": "processing",
                    "isIncremental": True
                }
            }
            if session_id in message_queues:
                message_queues[session_id].put(output_data)
    
    return callback

# ✨ 修改包装函数，传入回调
def process_base_query_with_callback(question, session_id):
    """使用回调函数版本的 process_base_query"""
    agent_name = "基础分析智能体"
    StreamHelper.send_step(session_id, "基础分析智能体", "处理简单问题")
    StreamHelper.send_step(session_id, "基础分析智能体", "生成回答", ["理解问题", "检索知识", "组织答案"])
    # 调用原函数，传入OpenAI客户端
    callback = create_callback(session_id, agent_name)
    result = process_base_query(question, client, retrieverr,callback=callback)
    StreamHelper.send_output(session_id, "基础分析智能体", result)
    return result

def process_mid_query_with_callback(question, session_id):
    """使用回调函数版本的 process_mid_query"""
    agent_name = "中等难度分析系统"
    callback = create_callback(session_id, agent_name)
    
    # 调用原函数，传入OpenAI客户端和回调函数
    final_decision, multiAgent = process_mid_query(question, client, callback=callback)
    
    return final_decision, multiAgent

def process_diff_query_with_callback(question, session_id):
    """使用回调函数版本的 process_diff_query"""
    agent_name = "高难度分析系统"
    callback = create_callback(session_id, agent_name)
    
    # 调用原函数，传入OpenAI客户端和回调函数
    final_decision, multiAgent = process_diff_query(question, client,callback=callback)
    
    return final_decision, multiAgent

# 修改后台处理函数
def background_process(question: str, session_id: str, difficulty: str,test_mode = False):
    """后台执行分析过程"""
    try:
        if test_mode:
            question_with_context = question
        else:
            question_with_context = create_question_with_context(question, session_id)
        # print(question_with_context)
        # 1. 难度评估（保持原有方式或也可以添加回调）
        if difficulty in ['simple', 'medium', 'hard']:
            difficulty_map = {'simple': '简单', 'medium': '中等', 'hard': '困难'}
            difficulty = difficulty_map[difficulty]
        else:
            # 这里也可以添加回调
            difficulty = determine_difficulty(question_with_context)
        
        # 2. 根据难度选择处理方式（使用回调版本）
        multiAgent = ""
        if difficulty == "简单":
            final_decision = process_base_query_with_callback(question_with_context, session_id)
        elif difficulty == "中等" or difficulty == "困难":
            final_decision, multiAgent = process_diff_query_with_callback(question_with_context, session_id)
        else:
            final_decision = "未知难度，无法处理"
        save_conversation_turn(session_id, question, final_decision)
        # 3. 发送最终结果和完成信号
        StreamHelper.send_final(session_id, final_decision, question)
        time.sleep(1)
        StreamHelper.send_complete(session_id, question)
        
    except Exception as e:
        error_data = {
            "type": "error",
            "error": f"处理过程中出现错误: {str(e)}"
        }
        if session_id in message_queues:
            message_queues[session_id].put(error_data)


# ✨ 新增：流式接口
@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    data = await request.json()
    question = data.get("query")
    session_id = data.get("id")
    print(session_id)
    enable_multi_agent = data.get("enableMultiAgent", False)
    difficulty = data.get("difficulty")

    
    if not enable_multi_agent:
        # 如果不启用流式，返回普通响应
        return await chat_endpoint(request)
    
    # 清空并初始化消息队列
    if session_id in message_queues:
        while not message_queues[session_id].empty():
            try:
                message_queues[session_id].get_nowait()
            except queue.Empty:
                break
    else:
        message_queues[session_id] = queue.Queue()
    
    # 启动后台处理线程
    thread = threading.Thread(target=background_process, args=(question, session_id, difficulty))
    thread.daemon = True
    thread.start()
    
    # 流式生成器
    async def stream_generator():
        while True:
            try:
                # 非阻塞获取消息
                if session_id in message_queues:
                    try:
                        message = message_queues[session_id].get_nowait()
                        yield StreamHelper.format_sse(message)
                        # 如果是完成信号，结束流
                        if message.get("type") == "complete":
                            break
                            
                    except queue.Empty:
                        # 队列为空，等待一下
                        await asyncio.sleep(0.1)
                        continue
                else:
                    break
                    
            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": f"流式传输错误: {str(e)}"
                }
                yield StreamHelper.format_sse(error_data)
                break
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# ✨ 保留：原有的非流式接口
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    question = data.get("query")
    session_id = data.get("id")
    
    # 原有逻辑保持不变
    if session_id in message_queues:
        while not message_queues[session_id].empty():
            try:
                message_queues[session_id].get_nowait()
            except queue.Empty:
                break
    else:
        message_queues[session_id] = queue.Queue()

    if session_id == '40101':
        difficulty = "简单"
    elif session_id == '40102':
        difficulty = "中等"
    elif session_id == '40103':
        difficulty = "困难"
    else:
        difficulty = determine_difficulty(question)
    
    multiAgent = ""
    parsedSchedule = ""
    
    if difficulty == "简单":
        final_decision = process_base_query(question, client)
        print(final_decision)
    elif difficulty == "中等":
        final_decision, multiAgent = process_mid_query(question, client)
        print(final_decision)
    elif difficulty == "困难":
        final_decision, multiAgent = process_diff_query(question, client)
        print(multiAgent)
    
    return {
        "最终结果": final_decision,
        "难度": difficulty,
        "多智能体结果": multiAgent,
        "多智能体调度结果": parsedSchedule,
        "id": session_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
