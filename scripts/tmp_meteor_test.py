import json
import time

from pycocoevalcap.meteor.meteor import Meteor

preds = json.load(open("results/official_coco/sft_full_vlm/predictions_coco.json"))[:100]
with open("dataset/coco2017/annotations/captions_val2017.json", encoding="utf-8") as file:
    anns = json.load(file)
refs = {}
for item in anns["annotations"]:
    refs.setdefault(item["image_id"], []).append(item["caption"])
res = {int(p["image_id"]): [p["caption"]] for p in preds}
gts = {key: refs[key] for key in res}
print(f"testing METEOR on {len(res)} images", flush=True)
started = time.time()
try:
    score, per = Meteor().compute_score(gts, res)
    print(f"METEOR OK score={score} elapsed={time.time() - started:.0f}s", flush=True)
except Exception as exc:
    print(f"METEOR FAILED: {type(exc).__name__}: {str(exc)[:500]}", flush=True)
