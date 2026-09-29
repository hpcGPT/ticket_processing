import json
import random
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "top_250_qa.json"
pairs = json.load(open(path))

bp = [p for p in pairs if re.search(r"pending solution acceptance|auto[- ]?closed", p["answer"], re.I)]
print(len(bp), "answers still contain auto-close boilerplate")

multi = [p for p in pairs if len(re.findall(r"^(Hi|Hello|Dear)\b", p["answer"], re.M)) >= 2]
print(len(multi), "answers have 2+ greetings (possible jumbled threads)")

random.seed(0)
for p in random.sample(pairs, 5):
    print("\n" + "=" * 60)
    print("Q:", p["question"])
    print("A:", p["answer"][:600])
