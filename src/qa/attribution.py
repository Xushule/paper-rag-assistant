# -*- coding: utf-8 -*-
"""引用校验（posthoc attribution，OpenScholar 同款机制）。

为什么需要（面试常问）：
- RAG 生成的回答带着 [n] 引用，但引用是否真的支撑了对应断言，
  目前是"生成后映射"——模型说 [2] 支持某结论，没人检查；
- posthoc attribution：回答生成后，再让 LLM 逐条核对"回答中每个
  引用对应的断言，是否被该片段原文支持"，给出 supported / partial /
  unsupported 判定与理由；
- 这是 OpenScholar 论文中的关键组件，也是竞品调研里我们与业界
  的主要差距之一。

实现：
- 一次 LLM 调用（DeepSeek）：输入 = 问题 + 回答 + 编号片段，输出 JSON；
- 校验失败不阻塞主流程（返回 None，由调用方静默跳过）。
"""
import json
import re

from openai import OpenAI

from ..config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# 每条片段送入校验器的最大字符数（回答通常 3-6 句，5 条 × 400 字 ≈ 2.5K tokens）
MAX_CHARS = 400

# References 章节名（大小写不敏感）
REF_SECTION_RE = re.compile(r"reference", re.IGNORECASE)


def needs_full_snippet(content: str, section: str = "") -> bool:
    """校验用片段是否需要给全文：表格块 / References 块。

    背景（两处真实踩坑）：
    - Markdown 表格块可能 600~1300 字符，截断 400 后关键数据行
      （如 Ours 31.57）落在截断点之后，校验器把"支持"误判成"不支持"；
    - References 块更长（实测 665~1787 字符），截断后列表后部的
      引文名（RainDiffusion / Uformer / SwinIR 等）看不到，
      回答说"参考文献提及 XX"时被误判为 unsupported。
    """
    if "|" in content and content.count("|") >= 4:
        return True  # 表格块给全文
    if section and REF_SECTION_RE.search(section):
        return True  # References 块给全文
    return False


def _snippet(c: dict) -> str:
    """校验用片段文本：普通片段截断，表格块 / References 块给全文。"""
    text = c.get("snippet", "") or ""
    if needs_full_snippet(text, c.get("section", "")):
        return text  # 表格 / References 块给全文
    return text[:MAX_CHARS]

VERIFY_PROMPT = """你是论文回答的引用校验器（posthoc attribution）。
下面有一个用户问题、一段带 [编号] 引用的模型回答，以及检索到的片段。
请逐条判断：回答中每个 [编号] 引用所支撑的断言，是否真的被对应片段支持。
只输出 JSON 数组，不要任何解释：
[{{"cite": 1, "verdict": "supported", "reason": "..."}}]

判定标准：
- supported：片段内容直接支持该引用处的断言（数字/事实与原文一致）；
- partial：片段只能部分支持（如缺关键细节、数字对不上、或需要推断）；
- unsupported：片段无法支持该断言，或与片段内容矛盾。

用户问题: {question}

模型回答:
{answer}

检索片段:
{items}
"""


def verify_citations(question: str, answer: str, citations: list[dict]) -> list[dict] | None:
    """校验回答中的每条引用是否被片段支持。

    Args:
        question: 用户问题。
        answer: 模型生成的回答（含 [n] 引用标记）。
        citations: 与 [n] 对应的片段元数据列表（含 source/section/page/snippet）。

    Returns:
        [{"cite": 1, "verdict": "supported|partial|unsupported", "reason": "..."}, ...]，
        仅含回答中实际出现的引用编号；解析失败返回 None（调用方静默跳过）。
    """
    if not citations or not answer:
        return None

    items = "\n".join(
        f"[{i + 1}] ({c['source']} / {c['section']} / 第{c['page']}页)\n{_snippet(c)}"
        for i, c in enumerate(citations)
    )
    prompt = VERIFY_PROMPT.format(question=question, answer=answer, items=items)

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                    timeout=60, max_retries=1)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""

    # 容错解析：模型可能包 ```json 代码块，只取第一个 [ 到最后一个 ]
    content = re.sub(r"```(?:json)?", "", content).strip()
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end <= start:
        return None
    data = json.loads(content[start:end + 1])

    # 只保留回答中实际出现的引用编号，过滤越界项
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    result = []
    for item in data:
        cite = int(item.get("cite", -1))
        verdict = item.get("verdict", "")
        if cite in used and verdict in ("supported", "partial", "unsupported"):
            result.append({
                "cite": cite,
                "verdict": verdict,
                "reason": str(item.get("reason", ""))[:120],
            })
    return result or None
