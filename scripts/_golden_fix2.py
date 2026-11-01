# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
path = r"D:\论文阅读问答助手\eval\golden_set.json"
g = json.load(open(path, encoding="utf-8"))
items = g.get("questions", g.get("items", g)) if isinstance(g, dict) else g
fix = {
 "all-01": "库内共 5 篇论文：AGAN、NightRain、NTIRE2025 挑战赛报告、Raindrop Clarity、Restormer。",
 "all-07": "AGAN 使用自己采集的雨滴数据集（Sony A6000/Canon EOS 60）；NightRain 使用 SynNightRain 合成数据集加真实夜间视频；Raindrop Clarity 和 NTIRE2025 使用 Raindrop Clarity 数据集；Restormer 使用多个图像复原基准数据集（如去雨 Rain100H/Test100、去模糊 GoPro 等）。",
}
n = 0
for it in items:
    if it.get("id") in fix:
        it["reference"] = fix[it["id"]]
        n += 1
json.dump(g, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated %d items" % n)
