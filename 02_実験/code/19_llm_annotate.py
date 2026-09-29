"""R-J: local open-weight LLMs as independent reference annotators (substitute for human
annotation, which is not available). LLM judgments are NOT ground truth.

Models (locally cached; see plan addendum 3): Qwen2.5-14B-Instruct (mlx-community 4-bit, MLX)
and Llama-3.1-8B-Instruct (Ollama llama3.1:8b, Q4). Greedy decoding (temperature 0), JSON output.
Inputs: 04_人手注釈キット/comments_annotation_sheet.csv (800), titles_annotation_sheet.csv (501).
Usage: python3 19_llm_annotate.py --model qwen14|llama
Outputs: results/llm_annot/{model}_comments.csv, {model}_titles.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "04_人手注釈キット"
OUT = ROOT / "02_実験" / "results" / "llm_annot"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"qwen14": "mlx-community/Qwen2.5-14B-Instruct-4bit", "llama": "llama3.1:8b"}
EMOS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
TARGETS = ["ai_tech", "ai_org", "ai_product", "people", "other", "none"]

SYS_COMMENT = """You are a careful annotator of emotions in online comments. You read a Hacker News story title and one comment on it.
Rate each emotion that the WRITER EXPRESSES in the comment (not the topic of the news) on 0-2: 0 = absent, 1 = weak, 2 = clear.
anger: anger, annoyance, blame. disgust: disgust, contempt. fear: fear, anxiety, worry, concern. joy: joy, admiration, hope, gratitude.
sadness: sadness, disappointment, regret. surprise: surprise, unexpected discovery, puzzlement. neutral: 2 if no emotion is expressed.
Judge sarcasm by its intended emotion.
Then choose the TARGET of the expressed emotion: ai_tech (AI technology, capabilities, risks), ai_org (AI companies, executives, researchers),
ai_product (a specific AI product or service such as ChatGPT or Copilot), people (other users, the public, or persons unrelated to AI),
other (a non-AI object such as a company, government, technology, or society), none (no target or no emotion).
Answer with JSON only, e.g. {"anger":0,"disgust":0,"fear":1,"joy":0,"sadness":0,"surprise":0,"neutral":0,"target":"ai_tech"}"""

SYS_TITLE = """You classify Hacker News story titles. Answer 1 if the main topic of the title is artificial intelligence or machine learning
(AI/ML systems, research, companies, policy, or AI hardware), otherwise 0. Autonomous driving or robotics without an AI focus is 0.
Answer with JSON only, e.g. {"is_ai_topic":1}"""


def parse(text: str, keys: list[str]) -> dict:
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return {k: d.get(k) for k in keys}


def run(model_key: str, prompts: list[list[dict]], keys: list[str], bs: int, max_new: int) -> list[dict]:
    outs = []
    if model_key == "qwen14":
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
        model, tok = load(MODELS[model_key])
        sampler = make_sampler(temp=0.0)
        for i, p in enumerate(prompts):
            text = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            d = generate(model, tok, prompt=text, max_tokens=max_new, sampler=sampler, verbose=False)
            outs.append(parse(d, keys) | {"raw": d.strip()[:300]})
            if i % 100 == 0:
                print(f"  {model_key}: {i}/{len(prompts)}", flush=True)
    else:
        for i, p in enumerate(prompts):
            r = requests.post("http://localhost:11434/api/chat", timeout=300, json={
                "model": MODELS[model_key], "messages": p, "stream": False, "format": "json",
                "options": {"temperature": 0, "num_predict": max_new, "seed": 0}})
            d = r.json()["message"]["content"]
            outs.append(parse(d, keys) | {"raw": d.strip()[:300]})
            if i % 100 == 0:
                print(f"  {model_key}: {i}/{len(prompts)}", flush=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    com = pd.read_csv(KIT / "comments_annotation_sheet.csv")
    prompts = [[{"role": "system", "content": SYS_COMMENT},
                {"role": "user", "content": f"Title: {r.title}\nComment: {str(r.text)[:1500]}"}] for r in com.itertuples()]
    res = run(args.model, prompts, EMOS + ["target"], args.bs, 80)
    out = pd.concat([com[["item_id"]], pd.DataFrame(res)], axis=1)
    out.to_csv(OUT / f"{args.model}_comments.csv", index=False)
    print("comments parsed:", out["target"].notna().mean().round(3))
    tit = pd.read_csv(KIT / "titles_annotation_sheet.csv")
    prompts = [[{"role": "system", "content": SYS_TITLE}, {"role": "user", "content": f"Title: {r.title}"}]
               for r in tit.itertuples()]
    res = run(args.model, prompts, ["is_ai_topic"], args.bs, 16)
    out = pd.concat([tit[["item_id"]], pd.DataFrame(res)], axis=1)
    out.to_csv(OUT / f"{args.model}_titles.csv", index=False)
    print("titles parsed:", out["is_ai_topic"].notna().mean().round(3))


if __name__ == "__main__":
    main()
