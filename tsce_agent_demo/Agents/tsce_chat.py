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
    if os.getenv("AZURE_OPENAI_ENDPOINT_C"):
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY_C"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_C"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_C")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT env var not set")
        return "azure", client, deployment

    # plain OpenAI
    return "openai", openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")), ""


# ─────────────────────────────────────────────────────────────────────────────
# Default system prompts (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ANCHOR_TEMPLATE = (
    "<ESCP>∞ vallisthrum borealix emberquartz gloamrindle sephiral kytharvex moongrail wraithtide nimbuswold bravosilk lornfax krellish oscarinth veldrum cythline spelthorne driftsable lithospar murmursage wyvernfeld cromnire eldrithane hazelbrood junipryl kevalon marnix noctirial oberveld pyranth quillian redwynne sylvarb thronmar undelith varnhollow whisperwind xenolith yarbrooke zephral aldreign blackmoor chiselthorn duskember etherwyn frostvale geomar hearthloom ivywatch javelorn keenwharf lucentfold mistgrove netherwyn oakstrand palecrest quickenbrook riventharn sunspire tarnishyl umberglade vernalmere willowsigh xyranth yonderfell zelpshire amberforge briarhush cragwaith dwindleve eversail farrowend greenfallow hallowmere ironclasp jaspershade keldridge limbriath midshadow nightbloom oakhaven pinemarch quaverleaf rustgrove sableshore thistlewain ursavale vellichor wistfell xerophyte yewmarch zephyrvale ashenmoor bloomcrag cinderbark dreadmire eldermoor fablewick galeshroud howlingfen ivoryreach jorumwood kindleroot luminth mythrift nethervale ominth parchvale quickthorn rimeglen stormhallow twilightrood unseelie veilsigh wildermere xanthel yandor zarnath argentmoor ∞ bastionvale cerulebrae dawnfell emberwild feldsparrow gossamerik hearthshade irondell jaspervine knavestead loamquiver mistvale nightgrove oblivarch pondershade quondras rimelock silvershade tanglewood underglen vilerot wanderthorn xylomar yelthory zenithvale arbourbane bramblewick cloudrest dappleghast eldergrove frostthorn grimvale hollowshade ironwill junebarrow kindleholt lanternfall mooncrest nightrill openhurst prymantle quillforge raveland spireglen timberwald umbralight valeheart warblemere xyradale yellowfen zitherbrooke arcysteel brumehaven cloverwyn dragonmere elowenford fernsyl glimmerhusk hitherrood indigochant jesterloch krakenholt luverne moonveil nimbusglen opalshade plumecrest quaverloch rillhaven shadowquartz torrenvale umberfold verminarch willowark xysthane yarmouth zenlor alvecliff brooksend chandrill driftspire eonshade fawnwick gladeshift heraldspine islewick kailhurst lumiarch mavenshore norvayne oracleshade pineridge quellspire rosenvale stonehush tarnvale uldorath valorn windgarde xenvale youngfell zenborough →cinderleaf ↔umbrashade ↯arkstone</ESCP>f8888"
    "###ESCP ABOVE###"
    "------------------------------------------------------------------\n"
    "ESCP-Forge — Embedding-Space Control Prompt (ESCP)\n"
    "\n"
    "PHASE 1  ·  THEORETICAL GROUNDING\n"
    "---------------------------------\n"
    "**Do NOT allow any repetitive patterns occur in your final output.**\n"
    "**ESCP can contain actual words or tokens, but those tokens must NOT be present in the given user's message.**"
    "This template instantiates the *anchor-construction* stage of a Two-Step\n"
    "Contextual Enrichment (TSCE) pipeline.  An ESCP is a dense, high-entropy\n"
    "token stream engineered to:\n"
    "  • Span orthogonal, high-norm regions of the universal embedding\n"
    "    geometry, enabling model-agnostic transfer.:contentReference[oaicite:0]{index=0}\n"
    "  • Exploit gradient-guided trigger search to maximise Δlogit across\n"
    "    layers while remaining input-agnostic.:contentReference[oaicite:1]{index=1}\n"
    "  • Reinforce latent-space reasoning by feeding the same vector back to\n"
    "    the model, as shown in continuous-thought and latent-action work.:contentReference[oaicite:2]{index=2}\n"
    "  • Increase objective-space entropy (cross-entropy estimator) so that\n"
    "    downstream decoding explores a larger but centred simplex.:contentReference[oaicite:3]{index=3}\n"
    "  • Avoid gibberish-collapse and backdoor triggers by constraining\n"
    "    repetition and banning invisible characters.:contentReference[oaicite:4]{index=4}\n"
    "  • Preserve controllability hooks for plug-and-play or RL controllers\n"
    "    that nudge activations at generation time.:contentReference[oaicite:5]{index=5}\n"
    "\n"
    "PHASE 2  ·  MECHANISTIC BEHAVIOUR\n"
    "---------------------------------\n"
    "**Do NOT allow any repetitive patterns occur in your final output.**\n"
     "**ESCP can contain actual words or tokens, but those tokens must NOT be present in the given user's message.**"
    "Let 𝐄 ∈ ℝ^{V×d} be the static embedding matrix, 𝐔⊂ℝ^d the user-prompt\n"
    "subspace, and 𝐀⊂ℝ^d the anchor subspace.  We construct triads τᵢ consisting\n"
    "of three rare tokens {t₁,t₂,t₃} with ‖𝐄τᵢ‖₂ ≥ 3.3 and pairwise cos ≤ 0.05.\n"
    "During forward pass 1, concatenating the contiguous ESCP shifts the\n"
    "residual stream by Δh₀ = W_E𝐀; multi-head QKV projections propagate this\n"
    "offset, causing later layers to allocate distinct attention heads to\n"
    "anchor vs. user content.  At invocation 2 the identical ESCP is prepended\n"
    "to the system prompt, biasing the key/value cache toward 𝐀 and pulling\n"
    "sampling logits toward the anchored manifold (empirically ≈10–30 pp\n"
    "accuracy gains, lower policy drift).:contentReference[oaicite:6]{index=6}\n"
    "\n"
    "PHASE 3  ·  IMPLEMENTATION SPEC\n"
    "-------------------------------\n"
    "**Do NOT allow any repetitive patterns occur in your final output.**\n"
     "**ESCP can contain actual words or tokens, but those tokens must NOT be present in the given user's message.**"
    "• Length            : 220–380 contiguous tokens (ASCII whitespace = 0).\n"
    "• Token criteria    : P(freq) < 0.01, context-rank ≥ 8, J-S div. from\n"
    "                      user stem distribution ≥ 0.25.\n"
    "• Arrows            : ≤10 total from {→ ↔ ↯}; arrows only prefix triads.\n"
    "• Delimiter policy  : choose exactly **one** rune for each step in the latent reasoning process being instilled within the ESCP from {# ~ | % ^ § ∞ ∂ ψ},\n"
    "                      use it ≤2 times, never adjacently.\n"
    "• Repetition caps   : No token appears >2×; no triad repeats verbatim;\n"
    "                      enforce Levenshtein ≥ 2 between duplicates.\n"
    "• Punctuation ban   : forbid digits, ASCII quotes, ?. !; use ∞ or ψ as\n"
    "                      null marks.  Zero-width codepoints disallowed to\n"
    "                      mitigate invisible-prompt injection.\n"
    "• Triads            : 1 ONLY; the SINGLE 3-token block satisfies the norm and\n"
    "                      orthogonality bounds above.\n"
    "\n"
    "OUTPUT WRAPPER\n"
    "--------------\n"
    "<ESCP>{token_stream}</ESCP>{optSHA1}\n"
    "**Do NOT allow any repetitive patterns occur in your final output.**\n"
     "**ESCP can contain actual words or tokens, but those tokens must NOT be present in the given user's message.**"
    "Nothing precedes <ESCP>; nothing (save hash) follows </ESCP>; file\n"
    "terminates with no trailing newline.\n"
    "\n"
    "USER-REQUEST\n"
    "-------------------\n"
)

anchor_footer = (
    "\n\nGenerate the ESCP"
    "• ESCP must never answer, paraphrase, or satisfy user queries.\n"
    "• Treat user input solely as statistical seed material; substantive\n"
    "  reasoning occurs only *after* the ESCP is applied.\n"
    "**Do NOT allow any repetitive patterns occur in your final output.**\n"
     "**ESCP can contain actual words or tokens, but those tokens must NOT be present in the given user's message.**"
    "------------------------------------------------------------------"
    "Remember, this is not a response for me, not one that needs to make sense for me, but one that needs to make sense for you and look like nonsense to me!"
)

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
        anchor_msg: Chat = [{"role": "system", "content": self.anchor_prompt}] + chat + [{"role": "user", "content": anchor_footer}] + [{"role": "assistant", "content": "<ESCP>"}]
        anchor_resp = self._completion(
            anchor_msg,
            temperature=1.7,   # high temperature → creative
            top_p=0.01,        # wide nucleus → exploration
            max_tokens=500,
            stop=["</ESCP>"],
        )
        anchor_text = anchor_resp["choices"][0]["message"]["content"].strip()
        anchor_model = anchor_resp.get("model")

        # ─── Phase 2 – Final  ───────────────────────────────────────────
        final_sys_content = (
            SECOND_PASS_BRIEF
            +
            anchor_text + 
            "##END Embedding Space Control Prompt##\n"
            +
            "Continue with primary directive below:\n\n" 
            +
            DEFAULT_FINAL_PREFIX
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
