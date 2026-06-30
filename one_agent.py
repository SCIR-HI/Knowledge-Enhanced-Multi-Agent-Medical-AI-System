from config import MODEL_NAME


generation_config_base = {
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 16384,
    "frequency_penalty": 0.05,
    "stop": None,
    "stream": True,
}


def _collect_rag_context(question, client, retriever):
    if retriever is None:
        return ""

    rag_knowledge_raw = retriever.retrieve_docs_multi_channel(
        question=question,
        model=client,
    )
    rag_knowledge = []
    for key in ("main_docs", "sub_docs"):
        docs = rag_knowledge_raw.get(key)
        if docs:
            rag_knowledge.extend(data["a"] for data in docs if "a" in data)

    return "\n".join(rag_knowledge)


def process_base_query(question, client, retriever=None, callback=None):
    """Process a simple medical query with optional retrieved context."""
    try:
        rag_context = _collect_rag_context(question, client, retriever)

        messages = [
            {
                "role": "system",
                "content": "你是京东方-哈工大多智能体医生助手，请回答下面的简单医学问题。",
            }
        ]
        if rag_context:
            messages.append(
                {
                    "role": "user",
                    "content": f"检索到的信息为：\n{rag_context}\n\n你要回答的问题为：{question}",
                }
            )
        else:
            messages.append({"role": "user", "content": f"你要回答的问题为：{question}"})

        if callback:
            callback(
                "step",
                "开始分析问题",
                "基础分析智能体",
                ["理解问题内容", "组织专业答案"],
            )

        config = generation_config_base.copy()
        config.update(
            {
                "messages": messages,
                "model": MODEL_NAME,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        )

        response_stream = client.chat.completions.create(**config)
        full_response = ""
        for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                if callback:
                    callback("incremental", content, "基础分析智能体")

        if callback:
            callback("output", f"## 基础分析结果\n\n{full_response}", "基础分析智能体")

        return full_response
    except Exception as e:
        error_msg = f"基础查询处理出错: {str(e)}"
        if callback:
            callback("output", f"## 错误信息\n\n{error_msg}", "基础分析智能体")
        return error_msg
