# -*- coding: utf-8 -*-
import pickle, sys
sys.path.insert(0, r"D:\论文阅读问答助手")
chunks = pickle.load(open(r"D:\论文阅读问答助手\data\chunks.pkl", "rb"))

hits = []
for c in chunks:
    if "Data Collection" in c.content or "sphere pan-tilt" in c.content or "3.1" in c.content:
        hits.append(c)

print("命中块数:", len(hits))
for c in hits[:10]:
    print("---", c.chunk_id, c.source, "len=%d" % len(c.content))
    print(c.content[:250].replace("\n", "\\n"))
