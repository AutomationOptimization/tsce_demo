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
import json
import os
import time
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
import requests

Backend = Literal["openai", "azure", "ollama"]
LOGPROB = os.getenv("LOGPROB", "0") not in {"0", "false", "no"}

DEFAULT_ANCHOR_ENDPOINT = "https://hda-anchor-web-b4c3hnemhedmfcca.canadacentral-01.azurewebsites.net/GetAnchor"
ANCHOR_ENDPOINT = os.getenv("TSCE_ANCHOR_ENDPOINT", DEFAULT_ANCHOR_ENDPOINT).strip()
ANCHOR_API_KEY = os.getenv("TSCE_ANCHOR_API_KEY", "").strip()
ANCHOR_TEMPERATURE = float(os.getenv("TSCE_ANCHOR_TEMPERATURE", "0.01"))
ANCHOR_TOP_K = int(os.getenv("TSCE_ANCHOR_TOP_K", "50"))
ANCHOR_TOP_P = float(os.getenv("TSCE_ANCHOR_TOP_P", "0.95"))
ANCHOR_MAX_NEW_TOKENS = int(os.getenv("TSCE_ANCHOR_MAX_NEW_TOKENS", "400"))

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


def _request_external_anchor(phase2_body: str) -> tuple[str, str | None]:
    """Call the external anchor service and return (anchor_text, model_name)."""
    if not ANCHOR_ENDPOINT:
        raise RuntimeError("External anchor endpoint is not configured.")

    headers = {"Content-Type": "application/json"}
    if ANCHOR_API_KEY:
        headers["X-API-KEY"] = ANCHOR_API_KEY

    payload = {
        "prompt": phase2_body,
        "temperature": ANCHOR_TEMPERATURE,
        "top_k": ANCHOR_TOP_K,
        "top_p": ANCHOR_TOP_P,
        "max_new_tokens": ANCHOR_MAX_NEW_TOKENS,
    }

    resp = requests.post(ANCHOR_ENDPOINT, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError as exc:  # invalid JSON
        raise RuntimeError("Anchor endpoint returned non-JSON data") from exc

    anchor_text: str | None = None
    anchor_model: str | None = None
    if isinstance(data, dict):
        for key in ("anchor", "content", "response", "text", "output"):
            candidate = data.get(key)
            if candidate:
                anchor_text = candidate
                break
        anchor_model = data.get("model")
    elif isinstance(data, str):
        anchor_text = data

    if not anchor_text:
        raise RuntimeError("Anchor endpoint response did not include anchor text")

    return anchor_text.strip(), anchor_model


# ─────────────────────────────────────────────────────────────────────────────
# Default system prompts (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FINAL_PREFIX = (
    "You are ChatGPT. A helpful AI Assistant.\n"
    "Think first step-by-step\n"
    "And then respond."
)

SECOND_PASS_BRIEF = (
    "The ESCP is a compact, high-entropy token sequence that pre-shapes internal activations without echoing user text.\n"
    "It combines ≥40 unique triads, runic delimiters, arrows, archetypes, and entropy gates to scatter activation space.\n"
    "Do NOT reference it directly or use tokens from it for your response.\n"
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
    final_prefix: str = DEFAULT_FINAL_PREFIX,
    deployment_id: str | None = None,
    client: openai.BaseClient | callable | None = None,
):
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

        phase2_body_no_anchor = json.dumps(
            [{"role": "system", "content": self._final_system_prompt("")}] + chat,
            ensure_ascii=False,
        )
        anchor_text: str | None = None
        anchor_model: str | None = None

        if ANCHOR_ENDPOINT:
            try:
                anchor_text, anchor_model = _request_external_anchor(phase2_body_no_anchor)
                if anchor_model is None:
                    anchor_model = "external-anchor"
            except Exception as exc:
                print(f"⚠️ [TSCE] external anchor call failed ({exc}); falling back to local generation.")

        if not anchor_text:
            anchor_text, anchor_model = self._local_anchor(chat)

        # ─── Phase 2 – Final  ───────────────────────────────────────────
        final_sys_content = self._final_system_prompt(anchor_text)
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

    def _local_anchor(self, chat: Chat) -> tuple[str, str | None]:
        """Fallback anchor generation using the configured OpenAI/Azure backend."""
        anchor_msg: Chat = (
            [{"role": "system", "content": self.anchor_prompt}] +
            chat +
            [{"role": "user", "content": anchor_footer}]
        )
        anchor_resp = self._completion(
            anchor_msg,
            temperature=0.1,   # high temperature → creative
            top_p=0.01,        # wide nucleus → exploration
            max_tokens=500,
        )
        anchor_text = anchor_resp["choices"][0]["message"]["content"].strip()
        return anchor_text, anchor_resp.get("model")

    def _final_system_prompt(self, anchor_text: str = "") -> str:
        """Return the full system prompt used in phase 2, inserting `anchor_text`."""
        return (
            SECOND_PASS_BRIEF
            + (anchor_text or "")
            + "##END Embedding Space Control Prompt##\n"
            + "Continue with primary directive below:\n\n"
            + self.final_prefix
        )

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
