from utils import Agent
import time
import concurrent.futures
from threading import Lock
callback_lock = Lock()
result_lock = Lock()

def analyze_domain_concurrent(domain, client, question, callback=None, total_domains=0, current_index=0):
    """单个领域专家分析函数 - 用于并发执行"""
    domain_clean = domain.strip()
    
    # 线程安全的进度报告
    if callback:
        with callback_lock:
            callback('incremental', f'📋 正在协调 {domain_clean} 领域专家 (进度: {current_index+1}/{total_domains})\n', 
                    agent_name='专家协调中心')
    
    # 每个领域专家独立工作
    if callback:
        with callback_lock:
            callback('step', f'接收分析任务', agent_name=f'{domain_clean}专家', details=[
                f'领域: {domain_clean}',
                '分析医疗情况',
                '提供专业建议'
            ])
    
    domain_role = f"你是{domain_clean}领域的医学专家。从您的专业领域出发,全面、详细地回答患者的问题。"
    domain_prompt = f"请仔细检查本问题中概述的医疗情况:{question}." \
                   f"利用你的医学专业知识,全面、详细地解释所描述的情况。" \
                   f"随后,找出并强调您认为最令人担忧或最值得注意的问题方面。"
    
    domain_agent = Agent(client, role_message=domain_role, role=f'{domain_clean}领域专家')
    
    if callback:
        with callback_lock:
            callback('incremental', f'🧠 {domain_clean}专家正在深度分析中...\n', agent_name=f'{domain_clean}专家')
    
    try:
        response = domain_agent.chat(domain_prompt)
        print(response)
        
        # 发送每个专家的分析结果
        if callback:
            with callback_lock:
                callback('output', f'## {domain_clean} 专业分析报告\n\n{response}', agent_name=f'{domain_clean}专家')
                callback('step', f'完成专业分析', agent_name=f'{domain_clean}专家', details=['分析完成', '报告已提交', '等待综合评估'])
        
        return domain, response
    except Exception as e:
        error_msg = f"分析过程中出现错误: {str(e)}"
        print(f'{domain_clean}专家分析失败: {error_msg}')
        if callback:
            with callback_lock:
                callback('output', f'## {domain_clean} 专家分析遇到问题\n\n{error_msg}', agent_name=f'{domain_clean}专家')
        return domain, error_msg

# 主要的并发执行代码
def run_concurrent_domain_analysis(domains, client, question, callback=None, max_workers=None):
    """并发运行领域专家分析"""
    domain_analysis = {}
    
    # 如果未指定最大工作线程数，默认为CPU核心数
    if max_workers is None:
        max_workers = min(len(domains), 4)  # 限制最大线程数，避免资源过度占用
    
    # 使用ThreadPoolExecutor进行并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        futures = []
        for i, domain in enumerate(domains):
            future = executor.submit(
                analyze_domain_concurrent, 
                domain, client, question, callback, len(domains), i
            )
            futures.append(future)
        
        # 获取并发执行结果
        for future in concurrent.futures.as_completed(futures):
            try:
                domain, response = future.result()
                with result_lock:
                    domain_analysis[domain] = response
            except Exception as exc:
                print(f'领域专家分析产生异常: {exc}')
    
    return domain_analysis

def process_mid_query(question, client, callback=None):
    
    NUM_QD = 3
    
    # 第一步：系统初始化
    if callback:
        callback('step', '启动中等难度分析流程', details=['初始化多智能体系统', '配置领域分类专家', '准备协作框架'])
    
    # 第二步：领域分类智能体工作
    if callback:
        callback('step', '配置领域分类专家', agent_name='领域分类智能体', details=['加载专家模型', '设置分类规则', '准备分析'])
    
    role_message = "你是一位医学专家,擅长将特定医疗场景分类到特定的医学领域。"
    domain_classfi_agent = Agent(client, role_message=role_message, role='领域评估专家')
    
    question_domain_format = "医学领域: " + " | ".join(["领域" + str(i) for i in range(NUM_QD)])
    domain_classfi_prompt = f"您需要完成以下步骤:" \
            f"1. 仔细阅读问题中提出的医疗问题:{question}. \n" \
            f"2. 根据其中的医疗场景,将问题分为三个不同的医学子领域 \n" \
            f"3. 请一步一步思考,给出你的分类理由，并将你的分类结果放在输出的最后\n" \
            f"4. 领域分类输出的格式为 {question_domain_format}."
    
    if callback:
        callback('step', '执行领域分类分析', agent_name='领域分类智能体', details=['分析医疗场景', '识别关键特征', '匹配医学领域'])
        callback('incremental', '🔍 正在分析问题的医疗场景...\n', agent_name='领域分类智能体')
    
    try:
        response = domain_classfi_agent.chat(domain_classfi_prompt)
        print(response)
        
        if callback:
            callback('output', f'## 领域分类分析过程\n\n{response}', agent_name='领域分类智能体')
        
        # 第三步：结果解析模块
        if callback:
            callback('step', '启动结果解析', agent_name='解析处理模块', details=['提取分类信息', '验证格式正确性', '准备专家分配'])
        
        domains = response.rsplit(":")[-1].strip().split(" | ")
        print(domains)
        
        if callback:
            callback('output', f'## 识别的医学领域\n\n' + '\n'.join([f'- **{domain.strip()}**' for domain in domains]), 
                    agent_name='解析处理模块')

        # 第四步：多领域专家协调中心
        if callback:
            callback('step', '启动多领域专家协作', agent_name='专家协调中心', details=['分配专家任务', '并行分析处理', '监控分析进度'])
        
        # 选择使用并发处理还是串行处理
        USE_CONCURRENT = True  # 可以设置为False来使用原来的串行处理
        
        if USE_CONCURRENT:
            # 使用并发处理
            domain_analysis = run_concurrent_domain_analysis(domains, client, question, callback)
        else:
            # 原来的串行处理方式
            domain_analysis = {}
            
            for i, domain in enumerate(domains):
                domain_clean = domain.strip()
                
                # 专家协调中心报告进度
                if callback:
                    callback('incremental', f'📋 正在协调 {domain_clean} 领域专家 (进度: {i+1}/{len(domains)})\n', 
                            agent_name='专家协调中心')
                
                # 每个领域专家独立工作
                if callback:
                    callback('step', f'接收分析任务', agent_name=f'{domain_clean}专家', details=[
                        f'领域: {domain_clean}',
                        '分析医疗情况',
                        '提供专业建议'
                    ])
                
                domain_role = f"你是{domain_clean}领域的医学专家。从您的专业领域出发,全面、详细地回答患者的问题。"
                domain_prompt = f"请仔细检查本问题中概述的医疗情况:{question}." \
                               f"利用你的医学专业知识,全面、详细地解释所描述的情况。" \
                               f"随后,找出并强调您认为最令人担忧或最值得注意的问题方面。"
                
                domain_agent = Agent(client, role_message=domain_role, role=f'{domain_clean}领域专家')
                
                if callback:
                    callback('incremental', f'🧠 {domain_clean}专家正在深度分析中...\n', agent_name=f'{domain_clean}专家')
                
                try:
                    response = domain_agent.chat(domain_prompt)
                    print(response)
                    domain_analysis[domain] = response
                    
                    # 发送每个专家的分析结果
                    if callback:
                        callback('output', f'## {domain_clean} 专业分析报告\n\n{response}', agent_name=f'{domain_clean}专家')
                        
                        callback('step', f'完成专业分析', agent_name=f'{domain_clean}专家', details=['分析完成', '报告已提交', '等待综合评估'])
                except Exception as e:
                    error_msg = f"分析过程中出现错误: {str(e)}"
                    print(f'{domain_clean}专家分析失败: {error_msg}')
                    domain_analysis[domain] = error_msg
                    if callback:
                        callback('output', f'## {domain_clean} 专家分析遇到问题\n\n{error_msg}', agent_name=f'{domain_clean}专家')
        
        # 协调中心汇总
        if callback:
            callback('output', f'## 专家协调总结\n\n✅ 已收集 {len(domains)} 个领域的专家报告\n\n**参与专家:**\n' + 
                    '\n'.join([f'- {domain.strip()}专家' for domain in domains]), 
                    agent_name='专家协调中心')
        
        # 第五步：综合分析专家
        if callback:
            callback('step', '启动综合分析阶段', agent_name='综合分析专家', details=['接收所有专家报告', '交叉验证分析', '整合医学观点'])
        
        synthesizer_role = "你是一名医学决策者,擅长根据不同领域专家的多位专家进行总结和综合。"
        
        # 构建综合分析的提示词，处理可能的错误情况
        reports_text = ""
        for i, domain in enumerate(domains):
            if i < len(domains):
                reports_text += f"[报告{i}]:{domain_analysis.get(domain, '该专家未能提供有效分析')}.\n\n"
        
        synthesizer_prompt = f"以下是来自不同医学领域专家的一些报告.\n" \
                            f"{reports_text}" \
                            f"您需要完成以下步骤:" \
                            f"1. 请仔细、全面地考虑以下报告。" \
                            f"2. 从以下报告中提取关键知识。" \
                            f"3. 在医学知识的基础上,得出全面、详细的分析。" \
                            f"4. 您的最终目标是在上述报告的基础上派生出一个全面且精炼的综合报告。" 
        
        synthesizer_agent = Agent(client, role_message=synthesizer_role, role='综合分析专家')
        
        if callback:
            callback('step', '执行综合分析', agent_name='综合分析专家', details=['分析报告一致性', '识别关键信息', '权衡不同观点'])
            callback('incremental', '📚 正在研读所有专家报告...\n', agent_name='综合分析专家')
            callback('incremental', '🔍 提取关键医学信息...\n', agent_name='综合分析专家')
            callback('incremental', '⚖️ 权衡不同专业观点...\n', agent_name='综合分析专家')
            callback('incremental', '📝 形成综合医学结论...\n', agent_name='综合分析专家')
        
        final_response = synthesizer_agent.chat(synthesizer_prompt)
        
        # 第六步：输出最终结果
        if callback:
            callback('output', f'## 多领域综合分析报告\n\n{final_response}', agent_name='综合分析专家')
            
            callback('step', '综合分析完成', agent_name='综合分析专家', details=['多智能体协作成功', '综合报告已生成'])
        
        return final_response, domain_analysis
    
    except Exception as e:
        error_msg = f"处理中等难度查询时发生错误: {str(e)}"
        print(error_msg)
        if callback:
            callback('output', f'## 系统错误\n\n{error_msg}', agent_name='系统')
        return error_msg, {}