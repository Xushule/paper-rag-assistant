# -*- coding: utf-8 -*-
"""轻量查询改写：中文问题 → 追加英文关键词，提升 BM25 对英文论文的召回。

为什么需要（面试常问）：
- 论文是英文的，用户用中文问，向量路靠 bge-m3 的跨语言能力能兜住语义；
- 但 BM25 是关键词匹配，中文"夜间雨滴"匹配不到英文正文里的 night/raindrop；
- 这里用一张中英术语词典把中文关键词翻译成英文补进查询词，
  BM25 路就能命中专有名词与术语（可进一步用 LLM 改写，第一版用规则更稳）。
"""
CN_EN_TERMS = {
    "数据集": "dataset", "数据": "dataset data", "方法": "method approach",
    "模型": "model network", "网络": "network", "贡献": "contribution",
    "实验": "experiment", "结果": "result", "效果": "performance result",
    "夜间": "night", "夜晚": "night", "白天": "day", "雨滴": "raindrop",
    "去雨": "deraining derain", "去除": "removal remove",
    "雨线": "rain streak", "模糊": "blur blurry", "对焦": "focus defocus",
    "图像恢复": "image restoration", "恢复": "restoration",
    "对比": "comparison compare", "指标": "metric PSNR SSIM",
    "损失": "loss", "训练": "training", "测试": "testing test",
    "注意力": "attention", "生成对抗": "generative adversarial GAN",
    "扩散模型": "diffusion model", "基准": "benchmark",
    "困难": "challenge challenging", "难点": "challenge challenging",
    "基线": "baseline", "消融": "ablation", "架构": "architecture",
    "低光": "low-light", "弱光": "low-light", "光照": "illumination lighting",
    "传感器": "sensor", "摄像机": "camera", "自动驾驶": "autonomous driving",
    "合成": "synthetic", "真实": "real-world real", "配对": "paired",
    "无监督": "unsupervised", "有监督": "supervised", "损失函数": "loss function",
}


def expand_query(query: str) -> str:
    """把中文问题里的常见术语翻译成英文关键词，追加到原问题后。"""
    en_terms = [en for zh, en in CN_EN_TERMS.items() if zh in query]
    # 去重保序
    seen, unique = set(), []
    for term in en_terms:
        for w in term.split():
            if w not in seen:
                seen.add(w)
                unique.append(w)
    return query + " " + " ".join(unique)
