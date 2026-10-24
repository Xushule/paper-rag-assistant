# -*- coding: utf-8 -*-
"""定位错题相关 chunk 全文：RobotCar 缺陷、排行榜表、Restormer 方法、AGAN/NightRain 数据集块。"""
import pickle, sys
sys.path.insert(0, r"D:\论文阅读问答助手")
chunks = pickle.load(open(r"D:\论文阅读问答助手\data\chunks.pkl", "rb"))
keys = ["RobotCar", "2.38", "148", "X-Restormer", "PolyRain", "restormer", "Restormer",
        "SynNightRain", "create", "own dataset", "our own", "dataset of", "comparison"]
for c in chunks:
    low = c.content
    for k in ["RobotCar", "2.38", "148M", "X-Restormer", "PolyRain", "SynNightRain", "our dataset", "own dataset"]:
        if k in low:
            print("=" * 70)
            print("CID:", c.chunk_id, "| section:", c.section, "| page:", c.page)
            print(" ".join(low.split())[:700])
            break
