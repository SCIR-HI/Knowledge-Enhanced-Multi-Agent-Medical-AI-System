import json
from openai import OpenAI
from config import SERVE_URL,MODEL_NAME

# vLLM部署的Qwen2.5-14B-Instruct在线服务配置
client = OpenAI(
    base_url=SERVE_URL,  # 替换为你的vLLM服务地址
    api_key="not-needed"  # vLLM服务通常不需要真实的API key
)

# 模型名称 - 使用vLLM部署的模型路径或别名
MODEL_NAME = MODEL_NAME  # 或者vLLM服务中设置的模型别名

# 生成配置参数（对应原来的vLLM参数）
generation_config_base = {
    "model": MODEL_NAME,
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 16384,
    "frequency_penalty": 0.05,  # 对应repetition_penalty的替代
    "stop": None,
    "stream": True,  # 启用流式输出
}

generation_config_greedy = {
    "model": MODEL_NAME,
    "temperature": 0.0,
    "max_tokens": 16384,
    "stop": None,
    "stream": False,  # 启用流式输出
}


class Agent:
    def __init__(self, client, role_message, role, examplers=None):
        self.role = role
        self.client = client
        self.model_name = MODEL_NAME
        self.messages = [
            {"role": "system", "content": role_message},
        ]
        if examplers is not None:
            for exampler in examplers:
                self.messages.append({"role": "user", "content": exampler['question']})
                self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message, callback=None, agent_name=None):
        """
        支持流式输出的聊天方法
        callback: 回调函数，用于处理流式输出 callback(type, content, agent_name)
        agent_name: 可选的智能体名称，用于回调标识
        """
        self.messages.append({"role": "user", "content": message})
        
        # 根据角色选择生成参数
        if self.role == '医学初步评估专家' or self.role == '招募者':
            config = generation_config_greedy.copy()
        else:
            config = generation_config_base.copy()
        
        config["model"] = self.model_name
        config["messages"] = self.messages
        config["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        
        try:
            if config.get("stream", False):
                # 流式输出
                response_stream = self.client.chat.completions.create(**config)
                full_response = ""
                
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        
                        # 如果有回调函数，发送增量内容
                        if callback:
                            callback('incremental', content, agent_name or self.role)
                
                # 流式输出完成后，添加到消息历史
                self.messages.append({"role": "assistant", "content": full_response})
                return full_response
            else:
                # 非流式输出（保持原有逻辑）
                response = self.client.chat.completions.create(**config)
                content = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": content})
                return content
                
        except Exception as e:
            print(f"API调用出错: {e}")
            return ""
    
    def temp_responses(self, message, callback=None, agent_name=None):
        """
        支持流式输出的多温度响应方法
        """
        self.messages.append({"role": "user", "content": message})
        temperatures = [0.1]
        responses = {}
        
        for temperature in temperatures:
            try:
                config = {
                    "model": self.model_name,
                    "messages": self.messages,
                    "temperature": temperature,
                    "max_tokens": 8192,
                    "stream": True,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
                }
                
                response_stream = self.client.chat.completions.create(**config)
                full_response = ""
                
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        
                        # 如果有回调函数，发送增量内容
                        if callback:
                            callback('incremental', content, agent_name or self.role)
                
                responses[temperature] = full_response
                
            except Exception as e:
                print(f"温度 {temperature} 的API调用出错: {e}")
                responses[temperature] = ""
        
        return responses

def remove_json_markers(text):
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:].lstrip()
    if text.endswith('```'):
        text = text[:-3].rstrip()
    
    return text
def determine_difficulty(question, callback=None):
    if callback:
        callback('step', '开始分析问题', '难度评估智能体', ['解析用户输入', '识别关键信息', '确定任务方向'])
    
    difficulty_prompt = """现在，给定如下的医疗查询，您需要确定它的难度/复杂程度：\n{}\n\n\
请从以下选项中选择：
1）简单：查询涉及基础医学知识或常识性问答，通常有标准答案或事实性描述。常见的问题类型涉及疾病定义与分类、常见症状识别、标准检查项目说明以及指南中直接规定的治疗方式或流程。常见的任务类型涉及问答检索、知识补全及医学术语解释。一些简单的日常交流问候也划分到简单
2）中等：查询涉及简单的推理，涉及复杂病因、合并症、诊断路径等。常见的问题类型涉及辅助诊断分析、多种治疗方案选择与对比、多学科因素交叉问题以及风险评估模型的解释与应用。常见的任务类型涉及多来源意见征集、临床路径推荐及交叉知识解释。
3）困难：查询涉及复杂的推理，需要多源知识整合、模型间协作讨论才能完成推理。常见的问题类型涉及个体化诊疗方案制定、病情发展趋势预测与风险评估、医学伦理相关情境判断及不典型病例的综合分析。常见的任务类型涉及病因分析、个案分析推理、诊疗意见推荐。
返回格式如下（请确保是合法 JSON）：
{{ "理由": "你的理由", "决策": "简单"/"中等"/"困难" }}
    """
    
    print(question)
    medical_agent = Agent(
        client, 
        role_message='你是进行初步评估的医学专家，你的工作是决定医疗查询的难度/复杂程度。', 
        role='医学初步评估专家'
    )
    
    # 使用流式输出
    # response = medical_agent.chat(difficulty_prompt.format(question), callback=callback, agent_name='难度评估智能体')
    response = medical_agent.chat(difficulty_prompt.format(question))
    print(response)
    response=remove_json_markers(response)
    try:
        response_json = json.loads(response)
        ans = ''
        if '简单' in response_json["决策"]:
            ans = '简单'
        elif '中等' in response_json["决策"]:
            ans = '中等'
        elif '困难' in response_json["决策"]:
            ans = '困难'
        
        result_content = f"## 难度评估结果\n\n**问题难度**: {ans}\n\n**评估理由**: {response_json.get('理由', '无详细理由')}"
        if callback:
            callback('output', result_content, '难度评估智能体')
        return ans
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return "简单"  # 默认返回中等难度