# -*- coding: utf-8 -*-
"""无边框表 LLM 结构化：判定候选区域是否真表，并转成 Markdown。

为什么交给 LLM（面试可讲）：
- 列对齐启发式只负责"定位候选"（召回优先，宁可多检），精度不足——
  作者列表、结构图标签、公式、参考文献都可能被误判为表格；
- 让 LLM 直接看候选内容：判断"这是不是表格"对 LLM 是很容易的任务，
  是表格就整理成 Markdown，不是就返回 NOT_TABLE——判定与结构化一步完成；
- 这就是"启发式定位 + LLM 结构化"的分工，误报由模型兜底。

失败回退：LLM 调用失败时返回空表列表，调用方让候选区域照常走文本流，
内容不丢。
"""
import json
import re

from openai import OpenAI

from ..config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# 每个候选送入 LLM 的最大行数（截断超长候选，防止 token 爆炸）
MAX_LINES = 40

TABLE_PROMPT = """你是 PDF 表格提取器。下面是某 PDF 页面中检测到的若干个候选区域，
每个区域由若干行文本组成（行内的列用两个空格分隔）。
请逐个判断每个候选是否真的是一张表格（如方法对比的数值表、数据集统计表、排名表）。
- 若是表格：整理成 Markdown 表格输出，保留原始行列结构与数字，不要增删数据；
- 若不是表格（网络结构图标签、作者列表、公式、参考文献、正文片段等）：输出 "NOT_TABLE"。
只输出一个 JSON 对象，不要任何解释：
{{"tables": [{{"idx": 0, "markdown": "| ... |\\n| --- |\\n| ... |"}}, {{"idx": 1, "markdown": "NOT_TABLE"}}]}}

候选区域:
{blocks}
"""


def _build_blocks(candidates: list[list[str]]) -> str:
    parts = []
    for i, lines in enumerate(candidates):
        body = "\n".join(lines[:MAX_LINES])
        parts.append(f"=== idx {i} ===\n{body}")
    return "\n\n".join(parts)


def structure_candidates(candidates: list[list[str]]) -> list[str | None]:
    """对候选区域做 LLM 判定 + 结构化。

    Args:
        candidates: 每个候选区域的行文本列表（按阅读顺序）。

    Returns:
        与 candidates 等长的列表；元素为 Markdown 表格文本，或 None（NOT_TABLE / 失败）。
    """
    if not candidates:
        return []
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                    timeout=60, max_retries=1)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": TABLE_PROMPT.format(blocks=_build_blocks(candidates))}],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""

    # 容错解析：剥离 ```json 代码块，取第一个 { 到最后一个 }
    content = re.sub(r"```(?:json)?", "", content).strip()
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return [None] * len(candidates)
    try:
        data = json.loads(content[start:end + 1])
    except Exception:
        return [None] * len(candidates)

    result: list[str | None] = [None] * len(candidates)
    for item in data.get("tables", []):
        idx = int(item.get("idx", -1))
        md = str(item.get("markdown", ""))
        if 0 <= idx < len(candidates) and md.strip() and md.strip().upper() != "NOT_TABLE":
            # 只接受真正的 Markdown 表格行（含 | 分隔符）
            if "|" in md:
                result[idx] = md
    return result
