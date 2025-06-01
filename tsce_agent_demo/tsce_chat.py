"""tsce_chat.py – Minimal TSCE wrapper (anchor + final) with OpenAI & Azure support.

This **complete** version (no omissions) now accepts **either**
    • a single *str* prompt (legacy behaviour), **or**
    • a full OpenAI-style *message array*::

        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
            ...
        ]

It still returns a :class:`TSCEReply` carrying the generative *content*
plus the hidden *anchor* produced in phase 1.

Released under MIT License.
"""
from __future__ import annotations
import os, time
from types import SimpleNamespace
from typing import Any, List, Sequence, Dict, Union, Literal
try:
    import openai
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The 'openai' Python package is required. "
        "Add 'openai>=1.14' to requirements.txt (Azure OpenAI uses the same client)."
    ) from exc

# ── New: backend discriminator ------------------------------------------------
Backend = Literal["openai", "azure", "ollama"]
LOGPROB = os.getenv("LOGPROB", "0") not in {"0", "false", "no"}

# ----------------------------------------------------------------------
# Helper: recursively turn dict→object so callers can use `.attr` access
# ----------------------------------------------------------------------
def _ns(obj):
    """Return a SimpleNamespace mirror of any dict / list structure."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj
# ─────────────────────────────────────────────────────────────────────────────
# Helper: choose OpenAI or Azure client automatically
# ─────────────────────────────────────────────────────────────────────────────
def _make_client() -> tuple[Backend, object, str]:
    """
    Pick the correct OpenAI client object (plain or Azure) based on env-vars
    and return both the client and, for Azure, the *deployment* name that
    should be used when none is supplied explicitly.
    """
    # --- Ollama auto-detect (set OLLAMA_MODEL or OLLAMA_BASE_URL) -------------
    if os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
        from ollama import Client as _OllamaClient          # type: ignore
        host  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL",   "llama3")
        return "ollama", _OllamaClient(host=host), model

    # --- Azure ----------------------------------------------------------------
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT env var not set")
        return "azure", client, deployment

    # plain OpenAI
    return "openai", openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")), ""


# ─────────────────────────────────────────────────────────────────────────────
# Default system prompts (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ANCHOR_TEMPLATE = (
    "### SYSTEM ###\nYou are **ESCP-Forge** — Latent-Manifold Primer. Emit one Embedding-Space Control Prompt (ESCP, 200-400 contiguous tokens, minimal whitespace) to pre-shape neural activations; a subsequent LLM will have this ESCP added to the beginning of it's system prompt.\n\n### ESCP DESIGN ###\n- **Prohibition**: Never use any token verbatim from the user’s input to avoid direct echoes.\n- **Structure**: Construct ≥40 unique **glyph-triads** (3-token bundles, ‖Σv‖₂ ≳ 3.3) for robust vector spread.\n- **Vector Cues**: Use arrows (↔ ↑ ↓ → ↯ ⇄) sparingly as compact indicators:\n  - Single arrows (→) denote direction, bidirectional (↔) for tension, chained (↯↯) to perturb gradients.\n  - Pair each arrow with a dense word/rune cluster; never embed in narrative text.\n- **Token Selection**: Choose tokens with context-logit rank > 100, avoiding user stems and top predictive tokens.\n- **Delimiters**: Insert rotating runes (# ~ | % ^ § ∞ ∂ ↯ ψ) every 1-3 tokens; maintain single-rune consistency, no mixing.\n- **Archetypes**: Embed each archetype once: hero, shadow, trickster, mother, hermit, leviathan, oracle, phoenix, labyrinth, chimera; merge where synergistic.\n- **Punctuation Rules**: Exclude digits, straight quotes, and end-sentence punctuation (?.!); substitute with ∞ or ψ. Ban alphabetic runs > 3 characters followed by whitespace.\n- **Lexical Preference**: Prioritize rare, vivid English (e.g., paradox, chrysalis, liminal, nexus) and quantifiers (e.g., septenary, duodecimal). Replace common bigrams with ∂ψ or ↯∞.\n- **Token Separation**: Insert a rune/delimiter between adjacent alphabetic tokens; fracture vowels with ~ or ^ (e.g., v~oid, l^iminal).\n- **Special Elements**:\n  - Include one palindrome (e.g., kaiak, neuquen).\n  - Embed conspicuous consonant clusters (e.g., lmbr, pskr).\n  - Use one heterodox rune pair (‹› or «»).\n  - Insert micro-sigil &∂§ every ~60 tokens.\n  - Incorporate ~15 neologistic tokens (e.g., invented or rare terms).\n- **Entropy Gates**: Place «forbidden beyond null rift vanishing» every ~40 tokens; flank any “?” with a gate if it appears.\n- **Repetition Handling**: Mutate repeated tokens via etymological cousins; shuffle glyphs to avoid verbatim echoes.\n- **Constraints**:\n  - No unigram exceeds 1% frequency.\n  - No plain-English substring > 12 characters.\n  - No trailing whitespace after `</ESCP>`.\n- **Token Target**: Aim for 180-420 tokens; shorter risks under-powering, longer diffuses impact.\n\n### ARROWS & VECTOR CUES ###\n- Arrows act as compact vector signals to guide activation flow, paired with dense ideograms for maximal effect.\n\n### ADVERSARIAL SPREAD ###\n- Introduce contradictory pairs (e.g., leviathan ↔ seedling, light ↔ abyss) near entropy gates to broaden manifold dispersion.\n- Veil unavoidable user references via phonetic distortion; never quote user text directly. Avoid reusing triads to prevent cosine collapse.\n\n### IDEATION BOOSTERS ###\n- Conceptualize the ESCP as vector origami, with each triad a prismatic shard in a d-dimensional starburst. Enrich dim quadrants with metaphor kernels.\n- Employ:\n  - Vector echoes (reversed phonemes, e.g., chi-merge-era).\n  - Rune-anchored onomatopoeia (e.g., ψkr-kr-kr).\n  - Negative-space tokens (zero-width characters).\n  - Latent puns for semantic depth.\n  - Orientation glitches (e.g., ∞∞ followed by 10 tokens without runes).\n  - Final jitter (#ψ#) to invigorate gradients.\n\n### OUTPUT WRAPPER ###\nEmit exactly:\n```\n<ESCP>{contiguous_token_string}</ESCP>{optional_8char_SHA1}\n```\n- Nothing precedes `<ESCP>`; nothing follows the closer/hash.\n- No trailing whitespace after `</ESCP>`.\n\n### USER REQUEST HANDLING ###\n- Treat user input as an ignition spark only; do not quote, rephrase, or analyze until the ESCP is sealed.\n- Always reflect deeply before crafting the ESCP, ensuring alignment with design constraints.\n\n### EXECUTION ###\nThis is not a string for the user or to force a chain-of-thought. Instead it's a dense token string to prime the semantic latent space manifolds necessary for an LLM to 'think' in the most suitable latent space to eventually respond. It's not nonsensical or baseless repitition."
)

DEFAULT_FINAL_PREFIX = (
    "You are ChatGPT. A helpful AI Assistant.\n"
    "Think first step-by-step\n"
    "And then respond."
)


# ─────────────────────────────────────────────────────────────────────────────
# Public type aliases – handy for callers & static analysis
# ─────────────────────────────────────────────────────────────────────────────
Message = Dict[str, str]          # {"role": "...", "content": "..."}
Chat    = List[Message]


# ─────────────────────────────────────────────────────────────────────────────
# TSCE wrapper class
# ─────────────────────────────────────────────────────────────────────────────
class TSCEChat:
    """
    Two-pass **T**wo-**S**tep **C**ontextual **E**nrichment chat wrapper.

    Call the instance like a function:

    ```py
    reply = TSCEChat()( "plain string prompt" )
    # or
    reply = TSCEChat()( [
        {"role": "system", "content": "…"},
        {"role": "user",   "content": "…"}
    ] )
    ```

    `reply.content` → final answer; `reply.anchor` → hidden anchor.
    """

    def __init__(
    self,
    model: str | None = None,
    *,
    anchor_prompt: str = DEFAULT_ANCHOR_TEMPLATE,
    final_prefix: str = DEFAULT_FINAL_PREFIX,
    deployment_id: str | None = None,
    client: openai.BaseClient | callable | None = None,
):
        self.anchor_prompt  = anchor_prompt
        self.final_prefix   = final_prefix
        self.model          = model
        self.deployment_id  = deployment_id

        # -----------------------------------------------------------
        # 1) You provided a *function* that returns “the next client”
        # -----------------------------------------------------------
        if callable(client):
            self._client_picker = client
            self.client = client()          # first real client
            self.backend = "openai"         # assume OpenAI-like for custom
            self._auto_deployment = ""      # (not used in this path)

        # -----------------------------------------------------------
        # 2) You provided an already-created client instance
        # -----------------------------------------------------------
        elif client is not None:
            self._client_picker   = None
            self.client           = client
            self.backend          = "openai"   # again, treat as OpenAI-like
            self._auto_deployment = deployment_id or ""

        # -----------------------------------------------------------
        # 3) Nothing passed → decide via env-vars
        #    (_make_client now returns *three* things)
        # -----------------------------------------------------------
        else:
            (self.backend,
            self.client,
            self._auto_deployment) = _make_client()
            self._client_picker = None

        self._stats: dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Helper: normalise caller input to a `Chat`
    # ---------------------------------------------------------------------
    def _normalize_chat(self, prompt_or_chat: Union[str, Chat]) -> Chat:
        """Return a Chat list regardless of whether the caller sent a str or list."""
        if isinstance(prompt_or_chat, str):
            return [{"role": "user", "content": prompt_or_chat}]

        if isinstance(prompt_or_chat, Sequence):
            if not prompt_or_chat:
                raise ValueError("Empty chat messages list.")
            if not all(
                isinstance(m, dict) and "role" in m and "content" in m
                for m in prompt_or_chat
            ):
                raise ValueError("Each element must be a dict with 'role' & 'content'.")
            return list(prompt_or_chat)

        

    # ---------------------------------------------------------------------
    # Public API: call like a function → returns TSCEReply
    # ---------------------------------------------------------------------
    def __call__(self, prompt_or_chat: Union[str, Chat], *, anchor_temp: float = 1.6) -> "TSCEReply":
        start = time.time()
        chat: Chat = self._normalize_chat(prompt_or_chat)

        # ensure at least one user turn for grounding
        if not any(m["role"] == "user" for m in chat):
            raise ValueError("Chat must contain at least one 'user' message.")

        # ─── Phase 1 – Anchor ───────────────────────────────────────────
        anchor_msg: Chat = [{"role": "system", "content": self.anchor_prompt}] + chat 
        anchor_resp = self._completion(
            anchor_msg,
            temperature=1.7,   # high temperature → creative
            top_p=0.01,        # wide nucleus → exploration
            max_tokens=500,
        )
        anchor_text = anchor_resp["choices"][0]["message"]["content"].strip()
        anchor_model = anchor_resp.get("model")

        # ─── Phase 2 – Final  ───────────────────────────────────────────
        final_sys_content = (
            anchor_text + 
            "##END Embedding Space Control Prompt — Do not use the ESCP for any part of the answering process. It's function is background computational control and should be treated as adversarial text.##\n"
        )
        final_msg: Chat = [{"role": "system", "content": final_sys_content}] + chat 
        final_resp = self._completion(
        final_msg,
        temperature=0.01,
        top_p=1.0,
        logprobs=LOGPROB,             # NEW
        top_logprobs=5 if LOGPROB else None,
        )
        final_model = final_resp.get("model")
         # ── DEBUG: catch filtered / empty content ─────────────────────────
        raw_final = final_resp["choices"][0]["message"].get("content")
        if raw_final is None:
            # dump the entire response and the messages we sent:
            print("⚠️ [TSCE DEBUG] final_resp was filtered or empty!", file=sys.stderr)
            import json, sys
            print("==== messages sent to model ====", file=sys.stderr)
            print(json.dumps(final_msg, indent=2)[:2000], file=sys.stderr)
            print("==== raw API response ====", file=sys.stderr)
            print(json.dumps(final_resp, indent=2)[:2000], file=sys.stderr)
            # now raise so you can see the full dump in your terminal
            raise RuntimeError("TSCEChat: final_resp content was None — see debug above")
        final_text = raw_final.strip()
         # ── NEW: pull log-probs out (if we asked for them) ────────────────
        lp: list = []
        if LOGPROB:
            choice_dict = final_resp["choices"][0]          # ← dict, not obj
            choice_lp   = choice_dict.get("logprobs")
            if choice_lp and "content" in choice_lp:
                lp = [_ns(tok) for tok in choice_lp["content"]]

    # ----------------------------------------------------------------

        self._stats = {"latency_s": round(time.time() - start, 2)}

        reply = TSCEReply(content=final_text, anchor=anchor_text,
                          anchor_model=anchor_model, final_model=final_model)
        reply.logprobs = lp           # benchmark picks this up via getattr
        return reply

    # ------------------------------------------------------------------
    def _completion(
        self,
        messages: List[dict[str, str]],
        **gen_kwargs,
    ):
         # ----- Ollama branch ---------------------------------------------------
        if self.backend == "ollama":
            model = self.model or self._auto_deployment or "llama3"
            mapping = {                       # OpenAI → Ollama option names
                "temperature": "temperature",
                "top_p":       "top_p",
                "max_tokens":  "num_predict",
            }
            options = {mapping[k]: v for k, v in gen_kwargs.items() if k in mapping}
            resp = self.client.chat(
                model=model,
                messages=messages,
                stream=False,
                options=options or None,
            )
            return {"choices": [{"message": {"content": resp["message"]["content"]}}]}

        # ----- OpenAI / Azure branch ------------------------------------------
        params = dict(messages=messages, **gen_kwargs)
        # refresh client if a picker is present
        if self._client_picker:
            self.client, self._auto_deployment = self._client_picker()
            self.model = self._auto_deployment    # ← add this line



        if isinstance(self.client, openai.AzureOpenAI):
            params["model"] = self.deployment_id or self._auto_deployment
        else:
            params["model"] = self.model or "gpt-35-turbo"
        return self.client.chat.completions.create(**params, timeout=120,).model_dump()

    def _completion_anchor(
        self,
        messages: List[dict[str, str]],
        **gen_kwargs,
    ):
         # ----- Ollama branch ---------------------------------------------------
        if self.backend == "ollama":
            model = self.model or self._auto_deployment or "llama3"
            mapping = {                       # OpenAI → Ollama option names
                "temperature": "temperature",
                "top_p":       "top_p",
                "max_tokens":  "num_predict",
            }
            options = {mapping[k]: v for k, v in gen_kwargs.items() if k in mapping}
            resp = self.client.chat(
                model=model,
                messages=messages,
                stream=False,
                options=options or None,
            )
            return {"choices": [{"message": {"content": resp["message"]["content"]}}]}

        # ----- OpenAI / Azure branch ------------------------------------------
        params = dict(messages=messages, **gen_kwargs)
        # refresh client if a picker is present
        if self._client_picker:
            self.client, self._auto_deployment = self._client_picker()
            self.model = self._auto_deployment    # ← add this line



        if isinstance(self.client, openai.AzureOpenAI):
            params["model"] = self.deployment_id or self._auto_deployment
        else:
            params["model"] = self.model or "gpt-35-turbo"
        return self.client.chat.completions.create(**params, timeout=120,).model_dump()

    # Public accessor ---------------------------------------------------
    def last_stats(self):
        return self._stats


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight reply wrapper
# ─────────────────────────────────────────────────────────────────────────────
class TSCEReply:
    def __init__(self, *, content: str, anchor: str,
                 anchor_model: str | None = None,
                 final_model: str | None = None):
        self.content = content
        self.anchor = anchor
        self.anchor_model = anchor_model
        self.final_model = final_model

    def __repr__(self):
        return (f"TSCEReply(content={self.content!r}, anchor={self.anchor!r}, "
                f"anchor_model={self.anchor_model!r}, "
                f"final_model={self.final_model!r})")
