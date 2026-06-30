from md_agent import process_diff_query
from openai import OpenAI
model_name = "/home/whli/model/Qwen3-32B-AWQ"
client = OpenAI(
    base_url="http://localhost:8000/v1",  # 替换为你的vLLM服务地址
    api_key="not-needed"  # vLLM服务通常不需要真实的API key
)
question="问题：较为常见视幻觉的痴呆类型是 \n选项：A:帕金森痴呆，B:血管性痴呆，C:路易体痴呆，D:额颞叶痴呆,E:Alzheimer病 \n 请你一步一步地解决这个问题，以确保选择正确的答案。\n 请在输出最后回复所选选项的字母，并放入\\boxed{{}}中，如\\boxed{{A}}，\\boxed{{B}}，\\boxed{{C}}，\\boxed{{D}}或\\boxed{{E}}。"
final_decision, multiAgent = process_diff_query(question, client)
print(final_decision)