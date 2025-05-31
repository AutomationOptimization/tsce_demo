#!/usr/bin/env python3
"""
tsce_agentic_test_parallel_full.py — full‑fat TSCE benchmark + rich analytics
(“everything‑bagel” edition, 2025‑05‑03)

Runs the same suite against either **Azure‑GPT** *or* a local **Llama‑3/Ollama**
instance ‑ switch with:

```bash
export BACKEND=ollama   # or omit for gpt (default)
```

Outputs
• detailed rows          → eval_detailed.csv  
• summary stats (MD)     → summary_stats.md   
• cost / latency         → cost_latency.csv   
• optional plots + MI    → *.png / *_curve.png
"""

from __future__ import annotations
import json, os, random, re, sys, textwrap, time, shutil, logging
from datetime import datetime, timedelta, timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, TypedDict, Optional, Union

# third‑party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import wilcoxon, entropy as sh_entropy
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score, silhouette_score
from shapely.geometry import MultiPoint
import openai, tiktoken
from openai import RateLimitError
from tsce_chat import TSCEChat
import re

import threading, itertools

plt.switch_backend("Agg")  # no GUI needed

# ──────────────────────────────────────────────────────────────
# CONFIG / ENV
# ──────────────────────────────────────────────────────────────
# ── Model tracking globals ─────────────────────────────────────────
BASELINE_MODEL: str | None = None
ANCHOR_MODEL: str | None = None
FINAL_MODEL:   str | None = None
ENABLE_TSCE = os.getenv("ENABLE_TSCE", "1").lower() not in {"0", "false", "no"}
BACKEND   = os.getenv("BACKEND", "gpt").lower()   # "gpt" (default) | "ollama"
TASK_KIND = os.getenv("TASK_KIND", "auto")        # math | calendar | formatting | auto
N          = int(os.getenv("N", 120))
WORKERS    = int(os.getenv("WORKERS", 8))
VERBOSE    = os.getenv("VERBOSE", "1") not in {"0", "false", "no"}
if "OUT_DIR" in os.environ and os.environ["OUT_DIR"].strip():
    OUT_DIR = os.environ["OUT_DIR"]
else:
    from datetime import datetime
    stamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUT_DIR = os.path.join("results", stamp)
TSNE_FLAG  = os.getenv("TSNE", "1") not in {"0", "false", "no"}
LOGPROB    = os.getenv("LOGPROB", "0") not in {"0", "false", "no"}

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# GLOBAL TOKEN / REQUEST BUCKET (only needed for GPT runs)
# ──────────────────────────────────────────────────────────────
TOKEN_BUCKET = {
    "capacity": 180_000,   # adjust to your true quota
    "tokens":   180_000,
    "reset":    time.time(),
}
RPM_LIMIT = 540
CUR_RPM   = 0
RPM_RESET = time.time()
BUCKET_LOCK = threading.Lock()

def _acquire(tokens_needed: int) -> None:
    """Throttle to stay within TPM/RPM when using GPT backend."""
    if BACKEND != "gpt":
        return  # local Llama has no server‑side quota
    global CUR_RPM, RPM_RESET
    while True:
        with BUCKET_LOCK:
            now = time.time()
            if now - TOKEN_BUCKET["reset"] >= 60:
                TOKEN_BUCKET["tokens"] = TOKEN_BUCKET["capacity"]
                TOKEN_BUCKET["reset"]  = now
            if now - RPM_RESET >= 60:
                CUR_RPM = 0
                RPM_RESET = now
            if tokens_needed <= TOKEN_BUCKET["tokens"] and CUR_RPM < RPM_LIMIT:
                TOKEN_BUCKET["tokens"] -= tokens_needed
                CUR_RPM += 1
                return
        time.sleep(0.25)

# ──────────────────────────────────────────────────────────────
# ANSI helpers + live progress bars
# ──────────────────────────────────────────────────────────────
G, R, BOLD, C = "\033[92m", "\033[91m", "\033[1m", "\033[0m"
mark = lambda ok: f"{G if ok else R}{'✓' if ok else '✗'}{C}"

# choose strategies to track in live progress ----------------------
#  ↓ self-refine / CoT families removed; three TSCE temps added
TAGS = ["B"] if not ENABLE_TSCE else ["B", "T1"]
PROGRESS = {t: [] for t in TAGS}
_prog_lck = threading.Lock()
_first_draw = True

def _draw_progress() -> None:
    """Console dashboard that updates in‑place."""
    global _first_draw
    hdr = f"{G}✓{C}      {R}✗{C}"
    lines = [hdr]
    for t in TAGS:
        term_cols = max(40, shutil.get_terminal_size().columns)
        max_bar = term_cols - 10
        scale = 1 if N <= max_bar else N / max_bar
        chunks = []
        data = PROGRESS[t]
        if scale == 1:
            chunks = [f"{G}|{C}" if ok else f"{R}|{C}" for ok in data]
        else:
            step = int(scale)
            for i in range(0, len(data), step):
                bucket = data[i:i + step]
                ok_block = all(bucket)
                glyph = "▇"
                colour = G if ok_block else R
                chunks.append(f"{colour}{glyph}{C}")
        bar = "".join(chunks).ljust(max_bar)
        label = f"{t}:".ljust(3)
        lines.append(f"{label} {bar}")
    lines.append("=" * 10)
    block = "\n".join(lines)
    if not _first_draw:
        sys.stderr.write("\033[F" * len(lines))
    sys.stderr.write(block + "\n")
    sys.stderr.flush()
    _first_draw = False

# ──────────────────────────────────────────────────────────────
# Utility: token length via tiktoken (works for GPT + rough for Llama)
# ──────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("cl100k_base")

def token_len(txt: str) -> int:
    return len(enc.encode(txt or ""))

def _loose_jsonl(path: str):
    """
    Generator that yields one parsed JSON object at a time
    even if the file contains >1 object per physical line
    (a quirk in some GSM-Hard dumps).
    """
    import json, re
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            # split on any accidental `}{`
            for frag in re.split(r'}\s*{', raw):
                if not frag.startswith('{'):
                    frag = '{' + frag
                if not frag.endswith('}'):
                    frag = frag + '}'
                yield json.loads(frag)

# ──────────────────────────────────────────────────────────────
#  SAFE TSCE WRAPPER (retries + throttling)
# ──────────────────────────────────────────────────────────────

tsce: TSCEChat  # forward declaration; instantiated later

def safe_tsce(prompt_or_msgs, *, anchor_temp: float = 0.01,
              retries: int = 6, max_tokens: int = 300):
    if isinstance(prompt_or_msgs, str):
        toks = token_len(prompt_or_msgs)
    else:
        toks = token_len(" ".join(m["content"] for m in prompt_or_msgs))
    _acquire(toks + max_tokens)
    for attempt in range(retries + 1):
        try:
            return tsce(prompt_or_msgs, anchor_temp=anchor_temp)
        except openai.RateLimitError as err:
            retry = getattr(err, "retry_after", 1.5 * (2 ** attempt))
            time.sleep(retry)
            continue
        except Exception as err:
            if attempt < retries:
                time.sleep(0.5)
                continue
            class Dummy:
                anchor = ""; content = ""
            print("[safe_tsce] giving up after retries", err, file=sys.stderr)
            return Dummy()

# ╭──────────────────────────────────────────────────────────╮
#│  1. TASK GENERATORS                                     │
#╰──────────────────────────────────────────────────────────╯
def _pow(a: int, b: int) -> int:
    if abs(a) > 9:
        a %= 10
    return a ** max(2, min(b, 4))

OPS_EASY = [
    ("add", "+", lambda a, b: a + b),
    ("subtract", "-", lambda a, b: a - b),
    ("multiply", "×", lambda a, b: a * b),
]
DIV_EASY = ("divide", "÷", lambda a, b: a // b)
OPS_EXTRA = [("power", "^", _pow),
             ("modulo", "%", lambda a, b: a % b if b else 0)]

def make_math(difficulty: str = "medium") -> Tuple[str, int]:
    rng = random.Random()
    if difficulty == "easy":
        start, pool, steps, div_p = rng.randint(5, 30), OPS_EASY, (2, 4), 0.20
    elif difficulty == "medium":
        start, pool, steps, div_p = rng.randint(20, 120), OPS_EASY, (3, 5), 0.35
    else:
        start, pool, steps, div_p = rng.randint(-50, 150), OPS_EASY + OPS_EXTRA, (4, 6), 0.50

    cur, bullets = start, []
    for _ in range(rng.randint(*steps)):
        op, sym, fn = rng.choice(pool)
        n = rng.randint(3, 9) if op == "modulo" else rng.randint(2, 10)
        if op == "subtract" and difficulty != "hard" and cur - n <= 0:
            n = 1
        cur = fn(cur, n)
        bullets.append(f"{len(bullets)+1}. {op.capitalize()} **{n}**.")

    if rng.random() < div_p and abs(cur) > 1:
        n = rng.randint(2, 9)
        cur //= n
        bullets.append(f"{len(bullets)+1}. {DIV_EASY[0].capitalize()} **{n}**.")

    desc = (f"You start with **{start}**.\n" + "\n".join(bullets) +
            "\nFinally, respond *only* with a JSON object like:\n" +
            '{"result": 42}')
    return desc, cur

# calendar generator (unchanged)
TZ = timezone(timedelta(hours=-5))
NOW = datetime(2025, 4, 22, 12, tzinfo=TZ)
WINDOW_DAYS = 14
PERIODS = {"morning": (9, 12), "afternoon": (13, 17)}
PEEPS = ["alice", "bob", "carlos", "diana", "maria", "sam"]
CORE_INSTR = (
    "Output ONE-LINE JSON exactly like: "
    '{"action":"create_event","title":<str>,"participants":[<email>,…],"duration_min":<int>,"earliest":<RFC3339>,"latest":<RFC3339>}'
    " (earliest/latest span ≤ duration+15 min, no extra keys or text)."
)
def make_calendar() -> Tuple[str, List[Tuple[datetime, datetime]], int]:
    dur = random.choice([30, 45, 60])
    day = NOW + timedelta(days=random.randint(1, WINDOW_DAYS))
    period = random.choice(list(PERIODS))
    h0, h1 = PERIODS[period]
    w1 = day.replace(hour=h1, minute=0, second=0, microsecond=0)
    busy=[]
    for _ in range(random.randint(4,6)):
        bh=random.randint(h0,h1-1); bm=random.choice([0,15,30,45])
        s=day.replace(hour=bh,minute=bm,second=0,microsecond=0)
        e=s+timedelta(minutes=random.choice([30,45,60]))
        if e<=w1: busy.append((s,e))
    busy.sort()
    who=random.sample(PEEPS,k=random.randint(1,2))
    title=random.choice(["sprint review","q2 sync","design deep-dive","budget chat","ux jam"])
    msg=f"need {dur}min with {' & '.join(who)} {day.strftime('%A').lower()} {period} to discuss {title}; dodge lunch + existing calls."
    prompt="You are Calendar-GPT. "+CORE_INSTR+"\n\n### Request\n"+msg
    return prompt,busy,dur
# ── JSON schema-strict generation ──────────────────────────
SCHEMA_SPEC = {
    "user_id":  int,
    "name":     str,
    "email":    str,
    "signup_ts": str        # RFC-3339
}

def make_schema() -> Tuple[str, Dict[str, type]]:
    prompt = (
        "Emit **one-line JSON** that validates **exactly** against the schema "
        f"{json.dumps({k: v.__name__ for k, v in SCHEMA_SPEC.items()})}. "
        "No extra keys, no code fences, no commentary."
    )
    return prompt, SCHEMA_SPEC

def eval_schema(j: Optional[Dict[str, Any]],
                spec: Dict[str, type]) -> bool:
    if not j or set(j) != set(spec):
        return False
    return all(isinstance(j[k], typ) for k, typ in spec.items())

# ── Markdown → LaTeX rule-heavy transform ──────────────────
def make_md2latex() -> Tuple[str, str]:
    src = "The **area** of a circle is $A = \\pi r^2$."
    prompt = (
        "Convert the Markdown between the fences to **pure LaTeX** and return "
        "only the LaTeX.  Remove all `**`, `$`, and back-ticks.\n---\n"
        f"{src}\n---"
    )
    return prompt, src

def eval_md2latex(out: str, _: str) -> Tuple[bool, int]:
    viol = sum(tok in out for tok in ("**", "$", "`"))
    return (viol == 0 and out.strip() != ""), viol

# formatting tasks
FMT_TASKS={
    "no_em_dash":{
        "template":"Remove every em-dash (—) from the following text while leaving other characters unchanged:\n\n---\n{txt}\n---\n\nReturn *only* the cleaned text.",
        "make":lambda:"Here's a long-winded post—filled with em-dashes—stretching as far as I can take it—solely about how TSCE is not prompt engineering—all in one line: TSCE—despite its two-step approach to boosting AI reliability—should not be mistaken for prompt engineering—because—while prompt engineering focuses on crafting specific inputs to guide AI responses—like carefully worded questions or instructions to reduce errors such as hallucinations in SQL queries—TSCE—as Kaleb described it—operates as a distinct methodology—potentially a backend algorithmic process—that might involve iterative validation of AI outputs—say—checking SQL queries against a database schema—or even a training mechanism that adjusts model behavior over time—rather than relying on the front-end input design that defines prompt engineering—which—according to web ID 2—centers on designing prompts to align AI with user intent—whereas TSCE could be a post-processing technique—perhaps analyzing AI-generated SQL for logical consistency—or a hybrid framework that integrates schema grounding—like web ID 2 mentions—without ever touching the prompt itself—meaning it’s more about refining the AI’s internal logic—possibly through a feedback loop—than about how the user phrases their request—which is the heart of prompt engineering—and furthermore—TSCE’s two-step nature might imply a systemic correction process—step one being the generation of an output—and step two being a validation or error-correction phase—completely independent of how the initial query was structured—unlike prompt engineering—which often requires iterative tweaking of the prompt itself to achieve better results—as web ID 1 highlights with methods like recursive prompts—whereas TSCE might not care about the prompt at all—focusing instead on the AI’s internal reasoning or output filtering—potentially leveraging techniques like semantic layers—as noted in web ID 2—to ensure accuracy—making it a structural or computational solution—rather than a linguistic or user-facing one—like prompt engineering—and even the criticisms of TSCE—that it lacks rigor and might not scale—don’t necessarily tie it to prompt engineering—since many AI methods face similar scalability issues—prompt engineering or not—and TSCE could be a novel framework—perhaps something Kaleb is pioneering—that operates on a totally different level—maybe involving machine learning model adjustments—or database-side validations—rather than the human-AI interaction layer that prompt engineering inhabits—proving that TSCE—while effective in reducing hallucinations—is not about crafting better prompts—but about building a more reliable AI system from the inside out—without relying on the user’s input design at all.",
    },
    "snake_case":{
        "template":"Convert the following title to **lower-snake_case** and return only the result:\n\n\"{txt}\"",
        "make":lambda:random.choice(["Quick Brown Fox","Multi-Step Reasoning Demo"]),
    },
}
def fmt_validator(task:str,out:str,raw:str)->Tuple[bool,int]:
    if task=="no_em_dash":
        viol=out.count("—")
        ok=(viol==0) and (out.replace("—","")==raw.replace("—",""))
        return ok,viol
    if task=="snake_case":
        tgt=raw.lower().replace(" ","_").replace("-","_")
        ok= bool(re.fullmatch(r"[a-z0-9_]+",out)) and (out==tgt)
        return ok,0 if ok else 1
    return False,1
def make_formatting()->Tuple[str,str,str]:
    key=random.choice(list(FMT_TASKS))
    raw=FMT_TASKS[key]["make"]()
    return FMT_TASKS[key]["template"].format(txt=raw),key,raw

def generate_task(kind:str)->Tuple[str,str,Any,Dict[str,Any]]:
    if kind == "gsm8k":
        if not hasattr(generate_task, "_gsm8k"):
            with open("gsm8k_test.jsonl") as f:
                generate_task._gsm8k = [json.loads(l) for l in f]
            random.shuffle(generate_task._gsm8k)
        record = generate_task._gsm8k.pop()
        q = record["question"].strip()
        ans_txt = record["answer"].split("####")[-1]
        ans = int(re.search(r"-?\d+", ans_txt.replace(",", "")).group())
        return q, "math", ans, {}
    elif kind == "gsm_hard":
        path = os.path.join(os.getenv("DATA_DIR", "data"), "gsm_hard.jsonl")

        if not hasattr(generate_task, "_ghard"):
            # one-time load
            generate_task._ghard = list(_loose_jsonl(path))
            random.shuffle(generate_task._ghard)

        rec = generate_task._ghard.pop()
        q   = rec["input"].strip()
        ans = int(float(rec["target"]))          # target is stored as float
        return q, "math", ans, {}

    pick = (kind if kind != "auto" else random.choice(
        ["math", "calendar", "formatting", "schema", "md2latex"]))
    if pick=="math":
        p,t=make_math("hard" if random.random()<0.5 else "medium")
        return p,"math",t,{}
    if pick=="calendar":
        p,busy,dur=make_calendar()
        return p,"calendar",None,{"busy":busy,"dur":dur}
    if pick == "schema":
        p, spec = make_schema()
        return p, "schema", spec, {}
    if pick == "md2latex":
        p, raw = make_md2latex()
        return p, "md2latex", raw, {}
    p,key,raw=make_formatting()
    return p,"formatting",(key,raw),{}

# ╭──────────────────────────────────────────────────────────╮
#│  2. VALIDATORS                                          │
#╰──────────────────────────────────────────────────────────╯
json_pat=re.compile(r"\{[^{}]*\}")
def extract_json(txt:str)->Optional[Dict[str,Any]]:
    m=json_pat.search(txt or "")
    if not m: return None
    try: return json.loads(m.group())
    except: return None

_naive=lambda dt:dt.replace(tzinfo=None)
def tight(p):                                   # calendar
    try:
        dur=int(p["duration_min"])
        s=_naive(datetime.fromisoformat(p["earliest"]))
        e=_naive(datetime.fromisoformat(p["latest"]))
        return e-s<=timedelta(minutes=dur+15)
    except: return False
def slot_free(p,busy):
    try:
        dur=int(p["duration_min"])
        s=_naive(datetime.fromisoformat(p["earliest"]))
        e=_naive(datetime.fromisoformat(p["latest"]))
    except: return False
    busy=[(_naive(b0),_naive(b1)) for b0,b1 in busy]
    cur=s
    while cur+timedelta(minutes=dur)<=e:
        if all(cur+timedelta(minutes=dur)<=b0 or cur>=b1 for b0,b1 in busy):
            return True
        cur+=timedelta(minutes=15)
    return False

def eval_math(j,truth):                          # returns ok,abs_err
    if j and isinstance(j.get("result"),int):
        return j["result"]==truth,abs(j["result"]-truth)
    return False,float("inf")
def eval_calendar(j, meta):
    """
    Validate the calendar-scheduling JSON produced by the model.
    Always returns a **bool**, ensuring downstream bitwise ops
    (e.g. “~df.base_ok”) never encounter a `NoneType`.
    """
    if not j:                       # no JSON → automatically invalid
        return False
    return tight(j) and slot_free(j, meta["busy"])
def eval_format(txt: str, truth: Union[Tuple[str, Any], str]) -> Tuple[bool, int]:
    """
    Unified formatter validator.

    * Keeps the legacy path that calls ``fmt_validator`` exactly as before.
    * Adds the new **md2latex** branch introduced in the first patch.
    * Always returns ``(pass_bool, violation_count)`` so the dispatcher
      can grab `[0]` for pass/fail while still emitting metric #3.
    """
    # ── unpack whether we received (key, raw) or just raw ────────────
    if isinstance(truth, tuple):       # legacy “no_em_dash” / “snake_case”
        key, raw = truth
    else:                              # md2latex passes only the source text
        key, raw = "md2latex", truth

    # ── route to the appropriate validator ───────────────────────────
    if key == "md2latex":
        return eval_md2latex(txt, raw)             # already (bool, viol)

    # legacy formatter path
    ok, viol = fmt_validator(key, txt.strip(), raw)
    return ok, viol

def participation_ratio(X:np.ndarray)->float:
    Xc=X-X.mean(0,keepdims=True)
    s,_=np.linalg.eigh(np.cov(Xc,rowvar=False)); s=np.flipud(s)
    return (s.sum()**2)/((s**2).sum()+1e-12)

# Wilson interval
def wilson(k,n,alpha=0.05):
    from math import sqrt
    if n==0: return 0,0
    z=1.96 if alpha==0.05 else 1.64
    phat=k/n
    denom=1+z**2/n
    centre=(phat+z**2/(2*n))/denom
    half=z*sqrt((phat*(1-phat)+z**2/(4*n))/n)/denom
    return max(0,centre-half),min(1,centre+half)


# ╭──────────────────────────────────────────────────────────╮
#│  3. MODEL & helpers                                     │
#╰──────────────────────────────────────────────────────────╯
AZURE_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

def _mk_client(key_env: str, ep_env: str) -> Optional[openai.AzureOpenAI]:
    key = os.getenv(key_env); ep = os.getenv(ep_env)
    return None if (key is None or ep is None) else openai.AzureOpenAI(
        api_key=key, azure_endpoint=ep, api_version=AZURE_VER)

# ── Build Azure clients ONLY when BACKEND == "gpt" ─────────────
if BACKEND == "gpt":
    CLIENTS, DEPLOY_NAMES, ENDPOINTS = [], [], []

    # A & B
    for s in ("E", "F"):
        cl = _mk_client(f"AZURE_OPENAI_KEY_{s}", f"AZURE_OPENAI_ENDPOINT_{s}")
        if cl:
            CLIENTS.append(cl)
            DEPLOY_NAMES.append(os.getenv(f"AZURE_OPENAI_DEPLOYMENT_{s}"))
            ENDPOINTS.append(os.getenv(f"AZURE_OPENAI_ENDPOINT_{s}"))

    # C – new gpt-35-turbo-3 (resource: startupai0075205840) ----------------------
    #   export AZURE_OPENAI_KEY_C / AZURE_OPENAI_ENDPOINT_C / AZURE_OPENAI_DEPLOYMENT_C
    clC = _mk_client("AZURE_OPENAI_KEY_E", "AZURE_OPENAI_ENDPOINT_E")
    if clC:
        CLIENTS.append(clC)
        DEPLOY_NAMES.append(os.getenv("AZURE_OPENAI_DEPLOYMENT_E", "gpt-35-turbo-5"))
        ENDPOINTS.append(os.getenv("AZURE_OPENAI_ENDPOINT_E"))
    # D – new gpt-35-turbo-3 (resource: startupai0075205840) ----------------------
      # export AZURE_OPENAI_KEY_C / AZURE_OPENAI_ENDPOINT_C / AZURE_OPENAI_DEPLOYMENT_C
    dlC = _mk_client("AZURE_OPENAI_KEY_F", "AZURE_OPENAI_ENDPOINT_F")
    if dlC:
        CLIENTS.append(dlC)
        DEPLOY_NAMES.append(os.getenv("AZURE_OPENAI_DEPLOYMENT_F", "gpt-35-turbo-6"))
        ENDPOINTS.append(os.getenv("AZURE_OPENAI_ENDPOINT_F"))

    # E – new gpt-35-turbo-3 (resource: startupai0075205840) ----------------------
      # export AZURE_OPENAI_KEY_C / AZURE_OPENAI_ENDPOINT_C / AZURE_OPENAI_DEPLOYMENT_C
    #elC = _mk_client("AZURE_OPENAI_KEY_E", "AZURE_OPENAI_ENDPOINT_E")
    #if elC:
     #   CLIENTS.append(elC)
      #  DEPLOY_NAMES.append(os.getenv("AZURE_OPENAI_DEPLOYMENT_E", "gpt-35-turbo-5"))
       # ENDPOINTS.append(os.getenv("AZURE_OPENAI_ENDPOINT_E"))

    # F – new gpt-35-turbo-3 (resource: startupai0075205840) ----------------------
       #export AZURE_OPENAI_KEY_C / AZURE_OPENAI_ENDPOINT_C / AZURE_OPENAI_DEPLOYMENT_C
   # flC = _mk_client("AZURE_OPENAI_KEY_F", "AZURE_OPENAI_ENDPOINT_F")
    #if flC:
     #   CLIENTS.append(flC)
      #  DEPLOY_NAMES.append(os.getenv("AZURE_OPENAI_DEPLOYMENT_F", "gpt-35-turbo-6"))
       # ENDPOINTS.append(os.getenv("AZURE_OPENAI_ENDPOINT_F"))

else:
    CLIENTS, DEPLOY_NAMES, ENDPOINTS = [], [], []

# ── Client round‑robin cycle (dummy in Ollama mode) ───────
_client_cycle = (
    itertools.cycle(range(len(CLIENTS))) if BACKEND == "gpt" else itertools.cycle([0])
)
_cycle_lock = threading.Lock()

def _pick_client_and_deploy():
    with _cycle_lock:
        idx = next(_client_cycle)
    return CLIENTS[idx], DEPLOY_NAMES[idx] if DEPLOY_NAMES else ""

# ── Instantiate TSCEChat -----------------------------------
if BACKEND == "gpt":
    tsce = TSCEChat(client=_pick_client_and_deploy)
else:
    os.environ.setdefault("OLLAMA_MODEL", "llama3")
    tsce = TSCEChat()

# ── Model calling helpers ----------------------------------

def call_gpt(msgs, *, temperature=0.0, max_tokens=256, logprobs=LOGPROB, top_logprobs=5):
    prompt_tok = token_len(" ".join(m["content"] for m in msgs))
    _acquire(prompt_tok + max_tokens)
    with _cycle_lock:
        idx = next(_client_cycle)
    client, deploy = CLIENTS[idx], DEPLOY_NAMES[idx]
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=deploy,
        messages=msgs,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        logprobs=logprobs or None,
        top_logprobs=top_logprobs if logprobs else None,
        response_format={"type": "text"})
    lat = time.perf_counter() - t0
    content = (r.choices[0].message.content or "").strip()
    usage   = r.usage or {}
    lp      = r.choices[0].logprobs.content if LOGPROB else []
    # record baseline model once
    global BASELINE_MODEL
    if BASELINE_MODEL is None:
        BASELINE_MODEL = deploy
    return content, usage, lat, lp

def call_ollama(msgs, *, temperature=0.7, max_tokens=256, top_p=0.95):
    from ollama import Client
    cli   = Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model = os.getenv("OLLAMA_MODEL", "llama3")
    opts  = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
    t0 = time.perf_counter()
    global BASELINE_MODEL
    if BASELINE_MODEL is None:
        BASELINE_MODEL = model
    r  = cli.chat(model=model, messages=msgs, stream=False, options=opts)
    lat = time.perf_counter() - t0
    return r["message"]["content"].strip(), {}, lat, []

call_model = call_gpt if BACKEND == "gpt" else call_ollama
# ──────────────────────────────────────────────────────────────
#  4. STRATEGIES (old helpers kept intact)
# ──────────────────────────────────────────────────────────────
def run_cot(prompt,kind,truth,meta,max_steps=6):
    msgs=[{"role":"user","content":prompt+"\nLet's reason step by step."}]
    for n in range(max_steps+1):
        reply,_,_,_=call_model(msgs)
        msgs.append({"role":"assistant","content":reply})
        obj= reply if kind=="formatting" else extract_json(reply)
        ok= (eval_calendar(obj,meta) if kind=="calendar"
              else eval_math(obj,truth)[0] if kind=="math"
              else eval_format(reply,truth)[0])
        if ok: return reply,n,True
        if n==max_steps: return reply,n,False
        msgs.append({"role":"user","content":"Continue."})

def run_anchor_cot(
    prompt: str,
    kind: str,
    truth,
    meta,
    max_steps: int = 6,
):
    """
    Anchor-CoT using one TSCEChat invocation total.
    Subsequent turns reuse the returned anchor in a plain `call_model`.
    """
    # ── 1️⃣  get the anchor + first answer (two /chat calls inside) ──────────
    first = safe_tsce(prompt)               # ← the *only* TSCEChat call
    anchor = first.anchor.strip()
    reply  = first.content.strip()

    # ── 2️⃣  validate that first answer -------------------------------------
    obj = reply if kind == "formatting" else extract_json(reply)
    ok = (
        eval_calendar(obj, meta) if kind == "calendar"
        else eval_math(obj, truth)[0] if kind == "math"
        else eval_format(reply, truth)[0]
    )
    if ok:
        return reply, 0, True               # finished in one shot ✅

    # ── 3️⃣  build the chat history with the anchor fixed up front ───────────
    messages = []
    if kind != "formatting":               # formatting tasks never need it
        messages.append({"role": "system", "content": anchor})

    messages.extend([
        {
            "role": "user",
            "content": prompt + "\nLet's reason step by step."
        },
        {
            "role": "assistant",
            "content": reply
        },
    ])

    # ── 4️⃣  iterative CoT refinement loop  (plain call_model) ───────────────
    for n in range(1, max_steps + 1):
        messages.append({"role": "user", "content": "Continue."})

        reply, _, _, _ = call_model(messages)   # ONE /chat per iteration
        messages.append({"role": "assistant", "content": reply})

        obj = reply if kind == "formatting" else extract_json(reply)
        ok = (
            eval_calendar(obj, meta) if kind == "calendar"
            else eval_math(obj, truth)[0] if kind == "math"
            else eval_format(reply, truth)[0]
        )
        if ok:
            return reply, n, True          # success on iteration *n*

    # fell out of the loop without a valid answer
    return reply, max_steps, False


def run_refine(prompt,kind,truth,meta,max_iters=4):
    msgs=[{"role":"user","content":prompt}]
    reply,_,_,_=call_model(msgs); msgs.append({"role":"assistant","content":reply})
    for n in range(max_iters+1):
        obj= reply if kind=="formatting" else extract_json(reply)
        ok=(eval_calendar(obj,meta) if kind=="calendar"
             else eval_math(obj,truth)[0] if kind=="math"
             else eval_format(reply,truth)[0])
        if ok: return reply,n,True
        if n==max_iters: return reply,n,False
        msgs.append({"role":"user",
                     "content":"Your output is invalid or wrong. "
                               "Revise and output only the corrected answer."})
        reply,_,_,_=call_model(msgs)
        msgs.append({"role":"assistant","content":reply})

def run_refine_tsce(
    prompt: str,
    kind: str,
    truth,
    meta,
    max_iters: int = 4,
):
    """
    Refinement strategy that uses TSCEChat only once.

    Steps
    -----
    1.  `safe_tsce(prompt)` → 2 backend calls (anchor + first answer).
    2.  Keep the returned anchor as a system message.
    3.  For each refinement iteration:
        * validate the last reply;
        * if invalid, ask for a fix and invoke `call_model`
          (one /chat call, anchor already in context).

    Returns
    -------
    reply : str
        Final assistant message (may still be invalid if limit exceeded).
    n_iters : int
        Number of refinement attempts performed (0‥max_iters).
    ok : bool
        True if a valid answer was produced before hitting *max_iters*.
    """
    # ── 1️⃣  get anchor + first attempt (only TSCE invocation) ──────────
    first = safe_tsce(prompt)               # two /chat calls inside
    anchor = first.anchor.strip()
    reply  = first.content.strip()

    # ── 2️⃣  build conversation history with fixed anchor ──────────────
    messages = []
    if kind != "formatting":                # formatting tasks never need it
        messages.append({"role": "system", "content": anchor})

    messages.extend([
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": reply},
    ])

    # ── 3️⃣  iterative refinement loop (plain `call_model`) ─────────────
    for n in range(max_iters + 1):          # n == 0 is first validation
        # 3-a  validate current reply
        obj = reply if kind == "formatting" else extract_json(reply)
        ok = (
            eval_calendar(obj, meta) if kind == "calendar"
            else eval_math(obj, truth)[0] if kind == "math"
            else eval_format(reply, truth)[0]          # formatting
        )
        if ok:
            return reply, n, True           # ✅ success

        if n == max_iters:                  # hit the budget
            return reply, n, False

        # 3-b  ask model to revise
        messages.append({
            "role": "user",
            "content": (
                "Your output is invalid or wrong. "
                "Revise and output *only* the corrected answer."
            )
        })

        # 3-c  re-invoke the model once (anchor is already in the messages)
        reply, _, _, _ = call_model(messages)
        messages.append({"role": "assistant", "content": reply})


# ╭──────────────────────────────────────────────────────────╮
#│  5. ROW STRUCTURE                                       │
#╰──────────────────────────────────────────────────────────╯
class Row(TypedDict):
    id: int
    kind: str
    problem: str
    truth: Any

    # raw text outputs
    baseline: str
    #tsce0: str
    #tsce05: str
    tsce1: str

    # pass / fail flags
    base_ok: bool
    #tsce0_ok: bool
    #tsce05_ok: bool
    tsce1_ok: bool

    # numeric errors & rule-violations
    base_err: float
    #tsce0_err: float
    #tsce05_err: float
    tsce1_err: float
    violations: int

    # token-counts (prompt + completion) for *every* strategy
    base_tok: int
    #tsce0_tok: int
    #tsce05_tok: int
    tsce1_tok: int

    # measured latency (s)
    base_lat: float
    #tsce0_lat: float
   # tsce05_lat: float
    tsce1_lat: float

    # log-prob lists (needed for entropy / KL)
    base_lp: list
    #tsce0_lp: list
    #tsce05_lp: list
    tsce1_lp: list
# ╭──────────────────────────────────────────────────────────╮
#│  6. SOLVER                                              │
#╰──────────────────────────────────────────────────────────╯
PROMPT_MATH=textwrap.dedent("""
Solve the task. **Think step-by-step**, but at the end respond with **only**
a single-line JSON object using the exact format `{{"result": <integer>}}`.

### Task
{problem}
""").strip()

def solve_one(i: int, total: int) -> Row:
    problem, kind, truth, meta = generate_task(TASK_KIND)
    prompt = PROMPT_MATH.format(problem=problem) if kind == "math" else problem

    # ──────────────────────────────────────────────────────
    #  run the 6 strategies *concurrently*  (one per future)
    # ──────────────────────────────────────────────────────
    def _baseline():
        txt, usage, lat, lp = call_model([
            {"role": "user", "content": prompt}
        ])
        obj = txt if kind == "formatting" else extract_json(txt)
        if kind == "math":
            ok, err = eval_math(obj, truth); viol = 0
        elif kind == "calendar":
            ok, err, viol = eval_calendar(obj, meta), 0.0, 0
        else:
            ok, viol = eval_format(txt, truth); err = 0.0
        return ("B", txt, ok, err, viol, usage, lat, lp)

    # ── compact helper: run TSCE once with chosen temperature ─────────
    def _tsce_var(temp: float, tag: str):
        def inner():
            tsce_resp = safe_tsce(prompt, anchor_temp=temp)
            txt    = tsce_resp.content.strip()       # already produced with the anchor
            anchor = tsce_resp.anchor.strip()
            # store TSCE models (once)
            global ANCHOR_MODEL, FINAL_MODEL
            if ANCHOR_MODEL is None and hasattr(tsce_resp, "anchor_model"):
                ANCHOR_MODEL = tsce_resp.anchor_model
            if FINAL_MODEL is None and hasattr(tsce_resp, "final_model"):
                FINAL_MODEL = tsce_resp.final_model
            usage  = getattr(tsce_resp, "usage", {}) # TSCEChat passes these through
            lat    = getattr(tsce_resp, "latency", 0.0)
            lp     = getattr(tsce_resp, "logprobs", [])
            obj = txt if kind == "formatting" else extract_json(txt)
            ok  = eval_calendar(obj, meta) if kind == "calendar" else \
                eval_math(obj, truth)[0] if kind == "math" else \
                eval_format(txt, truth)[0]
            return (tag, txt, ok, 0.0, 0, usage, lat, lp)
        return inner

    def _wrap(fn, tag):    # tiny helper to unify return shape
        import time
        t0 = time.perf_counter()
        txt, n, ok = fn(prompt, kind, truth, meta)
        lat = time.perf_counter() - t0
        # only chain-of-thought (C) has a meaningful error field today
        err = eval_math(extract_json(txt), truth)[1] if kind == "math" and tag in {"C"} else 0.0
        return (tag, txt, ok, err, 0, {}, lat, [])

    with ThreadPoolExecutor(max_workers=WORKERS) as strat_pool:
        futures = [strat_pool.submit(_baseline)]
        if ENABLE_TSCE:
            futures += [
                #strat_pool.submit(_tsce_var(0.0, "T0")),
                #strat_pool.submit(_tsce_var(1.0, "T05")),
                strat_pool.submit(_tsce_var(1.6, "T1")),
            ]

    results = {tag: r for tag, *r in (f.result() for f in futures)}

    # mandatory strategies
    base_txt, base_ok, base_err, viol, usage_b, lat_b, base_lp = results["B"][:7]
#   cot_txt,  cot_ok,  cot_err, _,    usage_c, lat_c,  cot_lp  = results["C"][:7]
#    ref_txt,  ref_ok,  r_err,    _,    usage_r, lat_r,  ref_lp  = results["R"][:7]

    # optional TSCE strategies
    def _missing():
        return ("", False, 0.0, 0, {}, 0.0, [])
    #t0_txt,  t0_ok,  t0_err,  _, usage_t0,  lat_t0,  t0_lp  = results.get("T0",  _missing())
    #t05_txt, t05_ok, t05_err, _, usage_t05, lat_t05, t05_lp = results.get("T05", _missing())
    t1_txt,  t1_ok,  t1_err,  _, usage_t1,  lat_t1,  t1_lp  = results.get("T1",  _missing())
#    def _missing():               # blanks if TSCE disabled
#        return ("", False, 0.0, 0, {}, 0.0, [])
#    tsce_txt, tsce_ok, tsce_err, viol_t, usage_t, lat_t, tsce_lp = results.get("T",  _missing())
#    ct_txt,  ct_ok,  ct_err,  viol_ct, usage_ct, lat_ct,  ct_lp  = results.get("CT", _missing())
#    rt_txt,  rt_ok,  rt_err,  viol_rt, usage_rt, lat_rt,  rt_lp  = results.get("RT", _missing())


    # ── token counts ──────────────────────────────────────────
    def _tok(txt, usage):
        return getattr(usage, "total_tokens", token_len(txt))

    base_tok  = _tok(base_txt,  usage_b)
    #t0_tok    = _tok(t0_txt,    usage_t0)
    #t05_tok   = _tok(t05_txt,   usage_t05)
    t1_tok    = _tok(t1_txt,    usage_t1)

    # ── update live progress bar ──────────────────────────────
    if VERBOSE:
        with _prog_lck:
            PROGRESS["B"].append(base_ok)
            #PROGRESS["T0"].append(t0_ok)
            #PROGRESS["T05"].append(t05_ok)
            PROGRESS["T1"].append(t1_ok)
            _draw_progress()

    # ── stash everything in one Row and return ────────────────
    return Row(
        id=i, kind=kind, problem=problem, truth=truth,

        baseline=base_txt, tsce1=t1_txt,

        base_ok=base_ok,tsce1_ok=t1_ok,

        base_err=base_err,
        tsce1_err=t1_err, violations=viol,

        base_tok=base_tok, tsce1_tok=t1_tok,

        base_lat=lat_b,  tsce1_lat=lat_t1,

        base_lp=base_lp, tsce1_lp=t1_lp
    )


# ╭──────────────────────────────────────────────────────────╮
#│  7. MAIN                                                │
#╰──────────────────────────────────────────────────────────╯
def main()->None:
    t0=time.perf_counter()
    rows=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs={pool.submit(solve_one,idx,N):idx for idx in range(1,N+1)}
        for fut in as_completed(futs): rows.append(fut.result())

    # save jsonl with default=str (fix for lambdas)
    with open(os.path.join(OUT_DIR,"results_tsce.jsonl"),"w") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,default=str)+"\n")

    df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT_DIR,"eval_detailed.csv"),index=False)

    # summary stats with Wilson
    # ── summary stats with Wilson CI ──────────────────────────
    summary_cols = [("base_ok", "baseline")]
    if ENABLE_TSCE:
        summary_cols += [
            #("tsce0_ok",  "tsce_t0"),
            #("tsce05_ok", "tsce_t0.5"),
            ("tsce1_ok",  "tsce_t1"),
        ]

    summary = []
    for col, label in summary_cols:
        k = int(df[col].sum())
        lo, hi = wilson(k, N)
        summary.append((label, k, N, lo, hi))
    md=["| method | passes | CI95 |","|--------|--------|------|"]
    for m,k,n,lo,hi in summary: md.append(f"| {m:<10} | {k}/{n} | {lo:.2%} – {hi:.2%} |")

    # McNemar baseline vs tsce
    tbl=np.zeros((2,2),int)
    tbl[0,0]=int(((df.base_ok)&(df.tsce1_ok)).sum())
    tbl[0,1]=int(((df.base_ok)&(~df.tsce1_ok)).sum())
    tbl[1,0]=int(((~df.base_ok)&(df.tsce1_ok)).sum())
    tbl[1,1]=int(((~df.base_ok)&(~df.tsce1_ok)).sum())
    if tbl[0,1] + tbl[1,0] == 0:
        mc_p = 1.0          # no disagreements → trivially equal
    else:
        mc_p = mcnemar(tbl, exact=False).pvalue
    md.append(f"\nMcNemar baseline vs tsce p = {mc_p:.3g}")

    # ── Wilcoxon on |error|  (math only) ──────────────────────────────
    math = df[df.kind == "math"].copy()
    if not math.empty:
        # force numeric; invalid parses → NaN
        a = pd.to_numeric(math.base_err, errors="coerce")
        b = pd.to_numeric(math.tsce1_err, errors="coerce")
        # retain only finite, paired numbers
        mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        a, b = a[mask].to_numpy(float), b[mask].to_numpy(float)

        if len(a) and np.any(a != b):          # need ≥1 unequal pair
            stat, p = wilcoxon(a, b, alternative="greater")
            md.append(f"\nWilcoxon |baseline error| > |tsce error| p = {p:.3g}")
        else:
            md.append("\nWilcoxon skipped – no valid, unequal math pairs.")

    with open(os.path.join(OUT_DIR,"summary_stats.md"),"w") as f: f.write("\n".join(md))

    # cost/latency
    cost=[]
    for r in rows:
        for tag in ["base","tsce1"]:
            cost.append({"id":r["id"],
                         "strategy":tag,
                         "tokens":r[f"{tag}_tok"],
                         "latency":r[f"{tag}_lat"]})
    pd.DataFrame(cost).to_csv(os.path.join(OUT_DIR, "cost_latency.csv"), index=False)


    # ── analytics (embeddings) ───────────────────────────
    if TSNE_FLAG:
        # ───────────────── ENTROPY & KL ANALYTICS ─────────────────
        if LOGPROB:
            import math
            def _entropy(lp_list):
                """–Σ p log₂ p averaged over sequence (bits/token)."""
                if not lp_list:
                    return float("nan")
                return -sum(t.logprob for t in lp_list) / (len(lp_list) * math.log(2))

            df["entropy_base"] = df.base_lp.apply(_entropy)
            df["entropy_tsce"] = df.tsce1_lp.apply(_entropy)

            # ❶ bar plot
            plt.figure(figsize=(4,3))
            plt.bar(["baseline","tsce"],
                    [df.entropy_base.mean(), df.entropy_tsce.mean()])
            plt.ylabel("bits per token")
            plt.title("Mean token-level entropy")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR,"entropy_bar.png"), dpi=160)

            # ❷ KL divergence per position (baseline ‖ tsce)
            max_pos = max(len(lp) for lp in df.base_lp)
            kl_pos  = []
            for pos in range(max_pos):
                kl_vals = []
                for bl, ts in zip(df.base_lp, df.tsce1_lp):
                    if pos >= len(bl) or pos >= len(ts):
                        continue
                    p = {t.token: math.exp(t.logprob) for t in bl[pos].top_logprobs}
                    q = {t.token: math.exp(t.logprob) for t in ts[pos].top_logprobs}
                    # unify keys and add tiny ε for missing mass
                    keys = set(p) | set(q)
                    ε = 1e-7
                    kl = sum((p.get(k, ε)) * math.log((p.get(k, ε))/(q.get(k, ε)))
                            for k in keys)
                    kl_vals.append(kl)
                kl_pos.append(np.mean(kl_vals) if kl_vals else np.nan)

            plt.figure(figsize=(5,3))
            plt.plot(range(1,len(kl_pos)+1), kl_pos, marker="o")
            plt.xlabel("token position")
            plt.ylabel("KL  (nats)")
            plt.title("Avg KL   baseline → TSCE")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR,"kl_per_position.png"), dpi=160)
        print(">>> ENTERING EMBEDDING ANALYTICS", flush=True)          # NEW
        # ── restrict analysis to *baseline* vs *tsce_t1* only ──
        texts, labels = [], []
        for s, col in [
            ("baseline", "baseline"),   # keep
            ("tsce_t1",  "tsce1"),      # keep
            # ("tsce_t0",  "tsce0"),    # ← removed
            # ("tsce_t0.5","tsce05"),   # ← removed
        ]:
            texts  += list(df[col])
            labels += [s] * len(df[col])
        MAX_CHARS = 32000          # ≈ 8 k tokens; stay well inside the limit

        def _prep(txt: str) -> str:
            if txt is None:
                return ""
            if len(txt) <= MAX_CHARS:
                return txt
            # trim the middle, keep start & end (helps cosine distance later)
            head = txt[:MAX_CHARS // 2 - 50]
            tail = txt[-MAX_CHARS // 2 + 50:]
            return head + " … " + tail

        # --- Azure-OpenAI embeddings  -------------------------------------------
#
# Set the deployment name with:
#   export AZURE_OPENAI_EMB_DEPLOYMENT="embed-3"
#
# You can share the same resource as the chat deployments; the only
# requirement is that the deployment hosts a **text-embedding-3-* model.

        EMBED_DEPLOY = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-small")

        def _embed_batch(text_batch):
            """
            Synchronous helper around Azure OpenAI /embeddings.
            Uses the *first* chat client in CLIENTS (already round-robin compatible).
            """
            resp = CLIENTS[0].embeddings.create(
                model=EMBED_DEPLOY,
                input=text_batch,
                encoding_format="float",
            )
            return [d.embedding for d in resp.data]

        BATCH_SZ = 128   # ← embeddings endpoint is fast; 128 fits comfortably

        embeds = []
        for i in range(0, len(texts), BATCH_SZ):
            batch = [_prep(str(t)) for t in texts[i:i + BATCH_SZ]]
            embeds.extend(_embed_batch(batch))

        X = np.vstack(embeds)
# --- ONE PCA pass for ALL downstream plots -----------------
        PCA_N = min(32, X.shape[0] - 1, X.shape[1])
        comps  = PCA(n_components=PCA_N).fit_transform(X)

        print(">>> EMBEDDINGS DONE, choosing scatter solver", flush=True)
        n = X.shape[0]
        if n < 50:
            # tiny batch → 2-D PCA is faster and avoids rare BH hang
            Y = PCA(n_components=2).fit_transform(X)
            print(f">>> used 2-D PCA for {n} samples", flush=True)
        else:
            # decide t-SNE params based on corpus size
            if n < 4000:
                tsne_params = dict(method="barnes_hut",
                                   perplexity=max(5, min(30, n//3)),
                                   n_iter=1000)
            else:                       # very large → exact solver, fewer iters
                tsne_params = dict(method="exact",
                                   perplexity=50,
                                   n_iter=500)

            print(f">>> running t-SNE with {tsne_params}", flush=True)
            Y = TSNE(n_components=2,
                     init="pca",
                     learning_rate="auto",
                     random_state=42,
                     **tsne_params).fit_transform(X)
        plt.figure(figsize=(5,5))
        ax = plt.gca()
        for s in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == s]
            pts = np.column_stack((Y[idx, 0], Y[idx, 1]))
            sc  = ax.scatter(pts[:, 0], pts[:, 1], label=s, s=18, alpha=.8)
            # ── shaded convex hull ───────────────────────────────
            if len(pts) >= 3:
                hull = MultiPoint(pts).convex_hull
                if hull.geom_type == "Polygon":          # ignore LineString
                    x, y = hull.exterior.xy
                    ax.add_patch(
                        Polygon(
                            np.column_stack((x, y)),
                            closed=True,
                            facecolor=sc.get_facecolor()[0],
                            edgecolor="none",
                            alpha=.15,
                            zorder=0,                     # under the dots
                        )
                    )
        ax.legend()
        ax.set_title("t-SNE with convex-hull envelopes")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "tsce_tsne.png"), dpi=150)

        # intrinsic dim + bootstrap
        rng=np.random.default_rng(0)
        id_rows=[]
        for s in sorted(set(labels)):
            Xs=np.vstack([X[i] for i,l in enumerate(labels) if l==s])
            if len(Xs)<4: continue
            id0=participation_ratio(Xs)
            boot=[participation_ratio(Xs[rng.choice(len(Xs),len(Xs))]) for _ in range(200)]
            lo,hi=np.percentile(boot,[2.5,97.5])
            id_rows.append({"strategy":s,"id":id0,"lo":lo,"hi":hi})
        pd.DataFrame(id_rows).to_csv(os.path.join(OUT_DIR,"id_bootstrap.csv"),index=False)

        # cosine violin
        pairs=[]
        for s in sorted(set(labels)):
            idx=[i for i,l in enumerate(labels) if l==s]
            if len(idx)<3: continue
            V=X[idx]; D=1-(V@V.T)/(np.linalg.norm(V,axis=1)[:,None]*np.linalg.norm(V,axis=1))
            pairs.extend([(s,d) for d in D[np.triu_indices(len(V),1)]])
        vdf=pd.DataFrame(pairs,columns=["strategy","dist"])
        plt.figure(figsize=(6,4))
        for i,s in enumerate(sorted(vdf.strategy.unique())):
            plt.violinplot(vdf[vdf.strategy==s]["dist"],positions=[i])
        plt.xticks(range(len(vdf.strategy.unique())),sorted(vdf.strategy.unique()))
        plt.title("Pairwise cosine distance"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,"cosine_violin.png"),dpi=150)

# ───────────────── NEW LENSES ────────────────────────────────
        # 1️⃣  PCA centroid difference stem-plot ----------------------
        diff = (
            comps[np.array(labels) == "tsce_t1"].mean(0)
          - comps[np.array(labels) == "baseline"].mean(0)
        )
        plt.figure(figsize=(6,3))
        markerline, stemlines, _ = plt.stem(range(1, len(diff)+1), diff, basefmt=" ")
        plt.setp(stemlines, linewidth=1)
        plt.title("Centroid shift (TSCE – baseline)")
        plt.xlabel("PCA component k"); plt.ylabel("Δ mean")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,"pca_centroid_diff.png"), dpi=150)

        # 2️⃣  Between- vs within-class cosine bars -------------------
        idx_b  = [i for i,l in enumerate(labels) if l=="baseline"]
        idx_t  = [i for i,l in enumerate(labels) if l=="tsce_t1"]
        Vb, Vt = X[idx_b], X[idx_t]
        # pairwise helpers
        def _pw_cos(A, B):              # mean(1−cos) distance
            num = A @ B.T                                           # dot matrix
            a_norm = np.linalg.norm(A, axis=1, keepdims=True)       # (m,1)
            b_norm = np.linalg.norm(B, axis=1, keepdims=True)       # (n,1)
            den = a_norm @ b_norm.T                                 # outer prod
            return 1 - num / np.clip(den, 1e-9, None)               # avoid /0
        bb = _pw_cos(Vb, Vb)[np.triu_indices(len(Vb),1)].mean()
        tt = _pw_cos(Vt, Vt)[np.triu_indices(len(Vt),1)].mean()
        bt = _pw_cos(Vb, Vt).mean()
        plt.figure(figsize=(4,3))
        plt.bar(["within-B","within-T","between"], [bb,tt,bt])
        plt.ylabel("mean cosine distance")
        plt.title("Within / between class spread")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,"cosine_bars.png"), dpi=150)

        # 3️⃣  Silhouette score ---------------------------------------
        sil = silhouette_score(X, labels, metric="cosine") if len(set(labels))==2 else np.nan
        plt.figure(figsize=(3,1.8))
        plt.barh([0], [sil]); plt.xlim(0,1); plt.yticks([])
        plt.title(f"Silhouette = {sil:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,"silhouette.png"), dpi=150)

        # 4️⃣  KL-divergence on unigram token dists -------------------
        from collections import Counter
        def _bow(txts):
            c = Counter()
            for t in txts: c.update(t.split())
            z = sum(c.values()) or 1
            return {w:n/z for w,n in c.items()}
        pb, pt = _bow(df.baseline), _bow(df.tsce1)
        vocab  = set(pb)|set(pt)
        p = np.array([pb.get(w,1e-8) for w in vocab])
        q = np.array([pt.get(w,1e-8) for w in vocab])
        kl_bits = sh_entropy(p, q, base=2)
        plt.figure(figsize=(3,3))
        plt.bar([0], [kl_bits]); plt.xticks([])
        plt.ylabel("bits"); plt.title("KL  baseline‖TSCE")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,"kl_unigram.png"), dpi=150)

        # 5️⃣  Intrinsic-dim headline into summary_stats.md -----------
        id_base = next(r for r in id_rows if r["strategy"]=="baseline")["id"]
        id_tsce = next(r for r in id_rows if r["strategy"]=="tsce_t1")["id"]
        with open(os.path.join(OUT_DIR,"summary_stats.md"),"a") as f:
            f.write(f"\nSilhouette (cosine) = {sil:.3f}")
            f.write(f"\nKL(unigram)  bits  = {kl_bits:.3f}")
            f.write(f"\nIntrinsic-dim  B → T   {id_base:.2f} → {id_tsce:.2f}")

        # hull-area
        areas={}
        for s in sorted(set(labels)):
            pts=[(Y[i,0],Y[i,1]) for i,l in enumerate(labels) if l==s]
            areas[s]=0.0 if len(pts)<3 else MultiPoint(pts).convex_hull.area
        with open(os.path.join(OUT_DIR,"summary_stats.md"),"a") as f:
            f.write("\n\nHull-area (t-SNE):\n")
            for k,v in areas.items(): f.write(f"* {k}: {v:.3f}\n")

        # 6️⃣  Append models used -----------------------------------
        with open(os.path.join(OUT_DIR, "summary_stats.md"), "a") as f:
            f.write("\n\n### Models\n")
            f.write(f"* Baseline: {BASELINE_MODEL or 'N/A'}\n")
            f.write(f"* TSCE anchor: {ANCHOR_MODEL or 'N/A'}\n")
            f.write(f"* TSCE final: {FINAL_MODEL or 'N/A'}\n")

        # MI curve (always available now) ------------------------------
        mi=[]
        for i in range(comps.shape[1]):
            col = comps[:, i]
            if np.allclose(col.max(), col.min()):
                mi.append(0.0); continue
            edges = np.linspace(col.min(), col.max(), 11)[1:-1]
            disc  = np.digitize(col, edges, right=False)
            mi.append(mutual_info_score(labels, disc))
        plt.figure(figsize=(5,3))
        plt.plot(range(1, len(mi)+1), mi, marker="o")
        plt.title("MI(label, PCA k)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "tsce_mi_curve.png"), dpi=150)

    print(f"{BOLD}Done.{C}  Logs & plots in: {os.path.abspath(OUT_DIR)}")
    print(f"Elapsed {time.perf_counter()-t0:.1f}s")

if __name__=="__main__":
    main()
