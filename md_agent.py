import json
import random
import time
import concurrent.futures
from threading import Lock
from tqdm import tqdm
from termcolor import cprint
from pptree import Node
from pptree import *
from prettytable import PrettyTable 
from utils import Agent
from openai import OpenAI
from retriever import Retriever
from config import DEFAULT_FAISS_VERSION

# 线程安全锁
callback_lock = Lock()
result_lock = Lock()
interaction_lock = Lock()
retriever = Retriever(DEFAULT_FAISS_VERSION)
def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]
    print(info)
    count = 0
    for expert, hierarchy in info:
        
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = '独立'
        print(hierarchy)
        if '>'  in hierarchy:
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()
            print(1)
            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1
    print("fin")
    return agents

def collect_initial_opinion_concurrent(agent_name, agent, question, fewshot_examplers,client, callback=None, agent_index=0, total_agents=0):
    """并发收集单个专家的初步意见"""
    if callback:
        with callback_lock:
            callback('step', f'正在咨询专业意见 ({agent_index+1}/{total_agents})', agent_name=agent_name)
    rag_knowledge_raw = retriever.retrieve_docs_multi_channel(question=question, model=client,is_polish=True,need_dim=agent_name)
    main_docs = rag_knowledge_raw["main_docs"]
    sub_docs = rag_knowledge_raw["sub_docs"]
    rag_knowledge = []
    if main_docs is not None:
        rag_knowledge = rag_knowledge + [data['a'] for data in main_docs]
    if sub_docs is not None:
        rag_knowledge = rag_knowledge + [data['a'] for data in sub_docs]
    question=question + "相关信息为：" + '\t'.join(rag_knowledge)
    print(agent_name,question)
    try:
        opinion = agent.chat(f'''结合你的专业知识，针对给定的医疗问题给出您的答案。\n\n问题：{question}\n\n您的答案应遵循以下格式。\n\n答案：''',callback=callback, agent_name=agent_name)
        
        if callback:
            with callback_lock:
                callback('output', f'## 专业分析\n\n{opinion}', agent_name=agent_name)
        
        return agent_name, opinion
    except Exception as e:
        error_msg = f"专家 {agent_name} 分析过程中出现错误: {str(e)}"
        print(error_msg)
        if callback:
            with callback_lock:
                callback('output', f'## {agent_name} 专家分析遇到问题\n\n{error_msg}', agent_name=agent_name)
        return agent_name, error_msg

def collect_final_answer_concurrent(agent_index, agent, question, callback=None, round_num=0):
    """并发收集单个专家的最终答案"""
    agent_name = agent.role
    
    try:
        response = agent.chat(f"""既然您已经与其他医学专家进行了互动，请回顾您的专业知识和本轮讨论中其他专家的评论，并对给定的问题给出您当前的最终答案：
{question}
答案：""",callback=callback, agent_name=agent_name)
        
        if callback:
            with callback_lock:
                callback('output', f'## 轮次 {round_num} 最终观点\n\n{response}', agent_name=agent_name)
        
        return agent_name, response
    except Exception as e:
        error_msg = f"专家 {agent_name} 最终答案收集出现错误: {str(e)}"
        print(error_msg)
        if callback:
            with callback_lock:
                callback('output', f'## {agent_name} 最终答案收集遇到问题\n\n{error_msg}', agent_name=agent_name)
        return agent_name, error_msg

def check_participation_concurrent(agent_index, agent, context, callback=None):
    """并发检查专家是否参与讨论"""
    agent_name = agent.role
    agent_id_str = f"代理 {agent_index+1}"
    
    try:
        participate = agent.chat(f"""根据您团队中其他医学专家的意见（如下所示）:

{context}

现在是交流阶段，我建议你与其他专家进行讨论，可以向他们问问题或者反驳他们的观点或对于不确定的部分进行讨论，输出Yes进行交流，如果你百分百确认自己正确且没有任何交流的必要，则输出No不进行交流。请说明您是否想与专家交谈,请只回答"Yes"或"No",不要输出任何其他额外内容""")
        # """根据您团队中其他医学专家的意见（如下所示）:

# {context}

# 有任何你没把握的地方都推荐你进行交流讨论，请说明您是否想与任何专家交谈,请只回答"Yes"或"No",不要输出任何其他额外内容"""
        print(participate)
        wants_to_participate = 'yes' in participate.lower().strip()
        return agent_index, agent_id_str, agent_name, wants_to_participate, participate
    except Exception as e:
        error_msg = f"检查专家 {agent_name} 参与意愿时出错: {str(e)}"
        print(error_msg)
        return agent_index, agent_id_str, agent_name, False, error_msg

def run_concurrent_initial_opinions(agent_dict, question, fewshot_examplers,client, callback=None, max_workers=None):
    """并发收集所有专家的初步意见"""
    if max_workers is None:
        max_workers = min(len(agent_dict), 4)
    
    opinions = {}
    initial_report = ""
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        agent_items = list(agent_dict.items())
        
        for i, (agent_name, agent) in enumerate(agent_items):
            future = executor.submit(
                collect_initial_opinion_concurrent,
                agent_name, agent, question, fewshot_examplers,client, callback, i, len(agent_items)
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                agent_name, opinion = future.result()
                with result_lock:
                    opinions[agent_name.lower()] = opinion
                    initial_report += f"({agent_name.lower()}): {opinion}\n"
            except Exception as exc:
                print(f'专家初步意见收集产生异常: {exc}')
    
    return opinions, initial_report

def run_concurrent_final_answers(medical_agents, question, callback=None, round_num=0, max_workers=None):
    """并发收集所有专家的最终答案"""
    if max_workers is None:
        max_workers = min(len(medical_agents), 4)
    
    final_answers = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, agent in enumerate(medical_agents):
            future = executor.submit(
                collect_final_answer_concurrent,
                i, agent, question, callback, round_num
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                agent_name, response = future.result()
                with result_lock:
                    final_answers[agent_name] = response
                print(f"    代理 ({agent_name}) 的答案已收集。")
            except Exception as exc:
                print(f'专家最终答案收集产生异常: {exc}')
    
    return final_answers

def run_concurrent_participation_check(medical_agents, context, callback=None, max_workers=None):
    """并发检查所有专家的参与意愿"""
    if max_workers is None:
        max_workers = min(len(medical_agents), 4)
    
    participation_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, agent in enumerate(medical_agents):
            future = executor.submit(
                check_participation_concurrent,
                i, agent, context, callback
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                with result_lock:
                    participation_results.append(result)
            except Exception as exc:
                print(f'专家参与检查产生异常: {exc}')
    
    # 按索引排序，保持原始顺序
    participation_results.sort(key=lambda x: x[0])
    return participation_results

def process_diff_query(question, client, callback=None,need_rag=False):
    """
    处理复杂度的医疗查询，适用于中文场景。
    callback: 可选的回调函数 callback(step_type, content, agent_name=None, details=None)
    """
    
    # if need_rag:
    #     rag_knowledge_raw = retriever.retrieve_docs_multi_channel(question=question, model=client)
    #     main_docs = rag_knowledge_raw["main_docs"]
    #     sub_docs = rag_knowledge_raw["sub_docs"]
    #     rag_knowledge = []
    #     if main_docs is not None:
    #         rag_knowledge = rag_knowledge + [data['a'] for data in main_docs]
    #     if sub_docs is not None:
    #         rag_knowledge = rag_knowledge + [data['a'] for data in sub_docs]
    #     question=question + "参考信息为：" + '\t'.join(rag_knowledge)

    # 第1步：专家招募
    cprint("[信息] 第1步：专家招募", 'yellow', attrs=['blink'])
    
    recruit_prompt = f"""您是一位经验丰富的医学专家，负责招募一组具有不同身份背景的专家，并要求他们讨论并解决给定的医疗问题。"""
    tmp_agent = Agent(client, role_message=recruit_prompt, role='招募者')
    
    num_agents = 3
    
    if callback:
        callback('step', f'招募 {num_agents} 名医学专家', agent_name='专家招募系统')

    recruited = tmp_agent.chat(f"""问题：{question}\n
您需要招募 {num_agents} 名具有不同医学专业知识的专家。考虑到医疗问题所涉及到的不同专业知识，您会招募哪些专家以便更好地做出准确的回答？
此外，您还需要指定专家之间的沟通结构，或者说明他们是独立工作的。

例如,如果你要招募五名专家，他们的沟通结构为：呼吸科医生 == 新生儿科医生 == 医学遗传学家 == 儿科医生 > 心脏病医生，你的输出应该如下：
1. 儿科医生 - 专注于婴幼儿、儿童和青少年的医疗保健。 - 层级结构：独立
2. 心脏病医生 - 专注于心血管相关疾病的诊断和治疗。 - 层级结构：儿科医生 > 心脏病医生
3. 呼吸科医生 - 专注于呼吸系统疾病的诊断和治疗。 - 层级结构：独立
4. 新生儿科医生 - 专注于新生儿的护理，特别是早产儿或出生时有医疗问题的新生儿。 - 层级结构：独立
5. 医学遗传学家 - 专注于基因和遗传的研究。 - 层级结构：独立

请严格按照上述格式回答，不要包含任何您的理由及其他多余信息。""",callback=callback, agent_name='专家招募系统')
    
    cprint("招募信息", 'yellow', attrs=['blink'])
    print(recruited)
    
    if callback:
        callback('output', f'## 专家招募结果\n\n{recruited}', agent_name='专家招募系统')
    
    # 解析招募信息
    agents_info = [agent_info.split(" - 层级结构：") for agent_info in recruited.split('\n') if '- 层级结构：' in agent_info]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

    # Agent Emojis
    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    # 解析层级结构
    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    # 创建代理列表字符串
    agent_list = ""
    for i, agent in enumerate(agents_data):
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
            agent_list += f"代理 {i+1}: {agent_role} - {description}\n"
        except:
            agent_role = agent[0].split('-')[0].strip().lower()
            description = agent[0].split('-')[1].strip().lower() if '-' in agent[0] else "（无详细描述）"
            agent_list += f"代理 {i+1}: {agent_role} - {description}\n"

    if callback:
        callback('step', '初始化专家团队', agent_name='团队管理系统')

    # 初始化Agent实例
    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except IndexError:
            agent_role = agent[0].split('-')[0].strip().lower()
            description = agent[0].split('-')[1].strip().lower() if '-' in agent[0] else "（无详细描述）"
        except Exception as e:
            print(f"解析代理信息时出错: {agent[0]}, 错误: {e}")
            continue

        inst_prompt = f"""您是一名 {agent_role}领域专家，专长是 {description}。您的工作是与团队中的其他医学专家协作。"""
        _agent = Agent(client, role_message=inst_prompt, role=agent_role)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    # 生成专家团队总结
    agent_summary = "## 专家团队组建完成\n\n"
    for idx, agent in enumerate(agents_data):
        try:
            agent_info = f"**专家 {idx+1}** ({agent_emoji[idx]}): {agent[0].split('-')[0].strip()}\n"
            agent_summary += agent_info
            print(f"代理 {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except IndexError:
            agent_info = f"**专家 {idx+1}** ({agent_emoji[idx]}): {agent[0].strip()}\n"
            agent_summary += agent_info
            print(f"代理 {idx+1} ({agent_emoji[idx]}): {agent[0].strip()}")
        except Exception as e:
            print(f"打印代理信息时出错: {agent[0]}, 错误: {e}")

    if callback:
        callback('output', agent_summary, agent_name='团队管理系统')

    # 准备Few-shot示例
    fewshot_examplers = ""
    reasoning_gen_agent = Agent(client, role_message='您是一个专业的医学代理。', role='医学专家')

    print()
    # 第2步：协作决策制定
    if callback:
        callback('step', '开始协作决策制定', agent_name='协作系统')
    
    cprint("[信息] 第2步：协作决策制定", 'yellow', attrs=['blink'])
    cprint("[信息] 第2.1步：层级结构选择", 'yellow', attrs=['blink'])
    try:
        print_tree(hierarchy_agents[0], horizontal=False)
    except IndexError:
        print("[警告] 未能生成层级结构树（可能没有招募到代理或解析失败）。")
    print()

    # 设置交互轮次和回合数
    num_rounds = 2
    num_turns = 2
    num_agents = len(medical_agents)
    if num_agents == 0:
        print("[错误] 未成功招募任何医学代理，无法进行协作。")
        if callback:
            callback('output', "## 错误\n\n未能招募到有效的医学专家，无法进行协作分析。", agent_name='系统错误')
        return {"error": "未能招募代理"}

    # 初始化交互日志
    interaction_log = {f'轮次 {round_num}': {f'回合 {turn_num}': {f'代理 {source_agent_num}': {f'代理 {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}
    print_log = {}
    
    # 第2.2步：参与式辩论 - 使用并发收集初步意见
    if callback:
        callback('step', '收集专家初步意见', agent_name='意见收集系统')
    
    cprint("[信息] 第2.2步：参与式辩论", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    
    print("[信息] 获取代理们的初步意见...")
    
    # 【并发优化1】：并发收集各专家初步意见
    if callback:
        callback('step', '并发收集专家初步意见', agent_name='并发协调器')
    
    round_opinions[1], initial_report = run_concurrent_initial_opinions(
        agent_dict, question, fewshot_examplers,client, callback
    )
    
    print(initial_report)
    print_log["初步意见"] = initial_report
    
    final_answer = None
    agent_assistant_role = '医学助理'
    agent_assistant_instruction = "您是一名医学助理，擅长基于来自不同领域专家的多种意见进行总结和综合。"
    print_log["交流过程"] = ""
    
    # 开始多轮辩论
    for n in range(1, num_rounds+1):
        print(f"== 轮次 {n} ==")
        round_name = f"轮次 {n}"
        
        if callback:
            callback('step', f'第 {n} 轮专家辩论', agent_name='辩论协调器')

        # 创建总结代理
        agent_rs = Agent(client, role_message=agent_assistant_instruction, role=agent_assistant_role)
        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

        # 生成阶段性总结
        report = agent_rs.chat(f'''这里有一些来自不同医学领域专家的报告。\n\n{assessment}\n\n您需要完成以下步骤：
1. 仔细全面地考虑以下报告。
2. 从以下报告中提取关键知识。
3. 基于这些知识进行全面和总结性的分析。
4. 您的最终目标是基于以下报告得出一个精炼和综合的报告。\n
您应该严格按照以下格式输出：\n关键知识：[您的关键知识总结]\n总体分析：[您的总体分析]''')
        print(f"  轮次 {n} 总结报告已生成。")

        if callback:
            callback('incremental', f'## 轮次 {n} 阶段性总结\n\n{report}', agent_name='总结分析师')

        num_yes_total_round = 0
        round_interactions = ""
        
        # 执行多回合交互
        for turn_num in range(num_turns):
            turn_name = f"回合 {turn_num + 1}"
            print(f"  |_{turn_name}")

            num_yes_turn = 0
            current_turn_interactions = []

            # 收集先前评论作为上下文
            all_prior_comments = ""
            if n > 1 or turn_num > 0:
                for r in range(1, n + 1):
                    for t in range(1, (turn_num + 1 if r == n else num_turns + 1)):
                        round_key = f"轮次 {r}"
                        turn_key = f"回合 {t}"
                        if round_key in interaction_log and turn_key in interaction_log[round_key]:
                            for source_agent_idx, targets in interaction_log[round_key][turn_key].items():
                                for target_agent_idx, comment in targets.items():
                                    if comment:
                                        all_prior_comments += f"{source_agent_idx} -> {target_agent_idx}: {comment}\n"

            context_for_participation_prompt = assessment if n == 1 and turn_num == 0 else all_prior_comments if all_prior_comments else assessment

            # 【并发优化2】：并发检查每个代理是否参与讨论
            if callback:
                callback('step', f'并发检查专家参与意愿', agent_name='参与协调器')
            
            participation_results = run_concurrent_participation_check(
                medical_agents, context_for_participation_prompt, callback
            )

            # 处理参与结果和后续交互
            for agent_index, agent_id_str, agent_name, wants_to_participate, participate_response in participation_results:
                agent_v = medical_agents[agent_index]
                
                print(participate_response)
                if wants_to_participate:
                    num_yes_turn += 1
                    num_yes_total_round += 1

                    chosen_expert_str = agent_v.chat(f"""请输入您想与之交谈的专家编号（只输入数字，多个用逗号隔开）：
{agent_list}
例如，如果您想与 代理 1. 儿科医生 交谈，请只返回 1。如果您想与多位专家交谈，请返回 1,2""")

                    try:
                        chosen_experts_indices = [int(ce.strip()) for ce in chosen_expert_str.replace('，', ',').split(',') if ce.strip().isdigit()]
                    except ValueError:
                        print(f"  [警告] {agent_id_str} 返回了无效的专家编号: '{chosen_expert_str}'，跳过此回合的发言。")
                        continue
                    
                    for ce_idx in chosen_experts_indices:
                        if 1 <= ce_idx <= len(medical_agents):
                            target_agent_id_str = f"代理 {ce_idx}"
                            specific_question = agent_v.chat(f"""请首先简单重申您的医学专业领域，然后向您选择的专家 ({target_agent_id_str}. {medical_agents[ce_idx-1].role}) 提出您的意见或问题,或者回答他之前提出的问题。请在有足够把握时，以简洁的理由清晰表达，力求说服对方。""")
                            
                            interaction_text = f"    {agent_id_str} ({agent_emoji[agent_index]} {medical_agents[agent_index].role}) -> {target_agent_id_str} ({agent_emoji[ce_idx-1]} {medical_agents[ce_idx-1].role}) : {specific_question}"
                            print(interaction_text)
                            round_interactions += f"\n{interaction_text}"
                            print_log["交流过程"] += f"\n{interaction_text}"
                            
                            # 线程安全地记录交互
                            with interaction_lock:
                                if round_name not in interaction_log: 
                                    interaction_log[round_name] = {}
                                if turn_name not in interaction_log[round_name]: 
                                    interaction_log[round_name][turn_name] = {f'代理 {i+1}': {} for i in range(num_agents)}
                                if agent_id_str not in interaction_log[round_name][turn_name]: 
                                    interaction_log[round_name][turn_name][agent_id_str] = {}

                                interaction_log[round_name][turn_name][agent_id_str][target_agent_id_str] = specific_question
                                current_turn_interactions.append(f"{agent_id_str} -> {target_agent_id_str}: {specific_question}")
                        else:
                            print(f"  [警告] {agent_id_str} 选择了无效的专家编号: {ce_idx}，跳过。")
                else:
                    silence_text = f"    {agent_id_str} ({agent_emoji[agent_index]} {agent_v.role}): \U0001f910 (选择不发言)"
                    print(silence_text)
                    round_interactions += f"\n{silence_text}"
                    print_log["交流过程"] += f"\n{silence_text}"

            if num_yes_turn == 0:
                print(f"  回合 {turn_num + 1} 中无代理发言，结束此轮次。")
                break

        # 发送本轮交互情况
        if callback and round_interactions:
            callback('incremental', f'### 轮次 {n} 专家交互记录\n{round_interactions}\n\n', agent_name='交互记录器')

        # 检查是否有代理在本轮发言
        if num_yes_total_round == 0 and n > 1:
            print(f"轮次 {n} 中无代理进行有效讨论，提前结束辩论。")
            if callback:
                callback('step', f'轮次 {n} 无有效讨论，提前结束', agent_name='辩论协调器')
            break

        # 【并发优化3】：并发收集本轮最终答案
        if callback:
            callback('step', f'并发收集轮次 {n} 最终答案', agent_name='答案收集器')
        
        print(f"  轮次 {n} 结束，收集中间答案...")
        
        tmp_round_final_answer = run_concurrent_final_answers(
            medical_agents, question, callback, n
        )

        round_answers[round_name] = tmp_round_final_answer
        final_answer = tmp_round_final_answer

    # 第3步：最终决策
    if callback:
        callback('step', '开始最终决策阶段', agent_name='最终决策系统')

    # 生成交互日志表格
    print('\n交互日志摘要')
    myTable = PrettyTable([''] + [f"代理 {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"代理 {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('---')
            else:
                agent_i_str = f"代理 {i}"
                agent_j_str = f"代理 {j}"
                i2j = False
                j2i = False
                # 检查所有轮次和回合的交互
                for r_idx in range(1, num_rounds + 1):
                    r_key = f"轮次 {r_idx}"
                    if r_key in interaction_log:
                        for t_idx in range(1, num_turns + 1):
                            t_key = f"回合 {t_idx}"
                            if t_key in interaction_log[r_key]:
                                # 检查 i -> j
                                if agent_i_str in interaction_log[r_key][t_key] and \
                                   agent_j_str in interaction_log[r_key][t_key][agent_i_str] and \
                                   interaction_log[r_key][t_key][agent_i_str][agent_j_str]:
                                    i2j = True
                                # 检查 j -> i
                                if agent_j_str in interaction_log[r_key][t_key] and \
                                   agent_i_str in interaction_log[r_key][t_key][agent_j_str] and \
                                   interaction_log[r_key][t_key][agent_j_str][agent_i_str]:
                                    j2i = True
                            if i2j and j2i: 
                                break
                        if i2j and j2i: 
                            break

                # 根据交互情况添加符号
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u2709 ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u2709 ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u21F5 ({i}<->{j})')

        myTable.add_row(row)

    print(myTable)

    cprint("\n[信息] 第3步：最终决策", 'yellow', attrs=['blink'])

    # 创建最终决策者
    moderator_role = "最终决策者"
    moderator_instruction = "您是最终的医疗决策者，负责审查来自不同医学专家的所有意见并做出最终决定。"
    moderator = Agent(client, role_message=moderator_instruction, role=moderator_role)

    if callback:
        callback('step', '创建最终决策者', agent_name='决策系统')

    # 准备最终答案字符串
    final_answer_str = ""
    final_ans = {}
    if final_answer:
        final_answer_str = "\n".join([f"专家 {role}: {ans}" for role, ans in final_answer.items()])
        final_ans = {role: ans for role, ans in final_answer.items()}
    else:
        final_answer_str = initial_report
        print("[警告] 未能从辩论中获取最终答案，将使用初步意见进行最终决策。")

    # 生成最终决策
    final_decision_content = moderator.chat(f"""请根据以下各位代理给出的最终答案（或初步意见），审阅每一位的观点，并通过多数投票（或综合判断）的方式，对原始问题做出最终的回答。

各位代理的意见:
{final_answer_str}

原始问题: {question}

您的最终答案应严格遵循以下格式：
答案：""",callback=callback, agent_name='最终决策者')
    
    final_decision = {'majority_vote_decision': final_decision_content}

    print(f"{moderator_role} 的最终决策:", final_decision_content)

    if callback:
        callback('output', f'## 最终医疗决策\n\n{final_decision_content}', agent_name='最终决策者')

    if callback:
        callback('step', '复杂分析完成', agent_name='系统总结')

    return final_decision_content, final_ans
