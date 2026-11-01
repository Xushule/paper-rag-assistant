# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
path = r"D:\论文阅读问答助手\eval\golden_set.json"
g = json.load(open(path, encoding="utf-8"))
items = g.get("questions", g.get("items", g)) if isinstance(g, dict) else g
new = {
 "all-02": {
   "question": "对比一下这五篇论文有什么不同？",
   "reference": "五篇定位不同：AGAN 是单图像雨滴去除方法（注意力 GAN）；NightRain 是夜间视频去雨方法（扩散模型）；Raindrop Clarity 是双聚焦昼夜数据集论文；NTIRE2025 是基于 Raindrop Clarity 的挑战赛方法与结果报告；Restormer 是通用图像复原 Transformer 方法（MDTA/GDFN，覆盖去雨、去模糊、去噪等任务）。"
 },
 "all-04": {
   "question": "这五篇论文里哪些是提出方法的，哪些是数据集的？",
   "reference": "方法论文：AGAN（雨滴去除）、NightRain（夜间视频去雨）、Restormer（通用图像复原）；数据集/报告：Raindrop Clarity（数据集）、NTIRE2025（挑战赛结果报告）。"
 },
 "all-14": {
   "question": "这五篇论文分别是什么类型、什么出处？",
   "reference": "四篇 arXiv 论文：AGAN（单图像雨滴去除方法，2017）、NightRain（夜间视频去雨方法，2024）、Raindrop Clarity（双聚焦数据集，2024）、Restormer（通用图像复原 Transformer，2022）；一篇挑战赛报告：NTIRE2025 Day and Night Raindrop Removal（2025）。"
 },
 "all-15": {
   "question": "库中五篇论文的 arXiv 编号分别是什么？",
   "reference": "AGAN：1711.10098；NightRain：2401.00729；Raindrop Clarity：2407.16957；NTIRE2025：2504.12711；Restormer：2111.09881。"
 },
}
n = 0
for it in items:
    qid = it.get("id")
    if qid in new:
        it["question"] = new[qid]["question"]
        it["reference"] = new[qid]["reference"]
        n += 1
json.dump(g, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated %d items -> %s" % (n, path))
