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
import os, time, random
from dataclasses import dataclass
from itertools import cycle
from types import SimpleNamespace
from typing import Any, Callable, List, Mapping, Sequence, Dict, Union, Literal
try:
    import openai
except ModuleNotFoundError:
    openai = None

# ── New: backend discriminator ------------------------------------------------
Backend = Literal["openai", "azure", "ollama"]
LOGPROB = os.getenv("LOGPROB", "0") not in {"0", "false", "no"}
# Anchor sampling knobs (override via env without code changes)
ANCHOR_TEMP  = float(os.getenv("TSCE_ANCHOR_TEMP", "1.25"))
ANCHOR_TOP_P = float(os.getenv("TSCE_ANCHOR_TOP_P", "0.9"))

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
# Helper: choose OpenAI or Azure client automatically (supports RR across 3 Azure clients)
# ─────────────────────────────────────────────────────────────────────────────
def _azure_clients_from_env(deployment_env_prefix: str) -> list[tuple[object, str]]:
    """Build up to three Azure clients from numbered env-vars (base, _2, _3)."""
    if openai is None:
        return []
    suffixes = ("", "_2", "_3")
    base_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    base_key = (os.getenv("AZURE_OPENAI_KEY") or "").strip()
    base_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview").strip()
    clients: list[tuple[object, str]] = []
    for suf in suffixes:
        endpoint = (os.getenv(f"AZURE_OPENAI_ENDPOINT{suf}") or (base_endpoint if suf == "" else "")).strip()
        key = (os.getenv(f"AZURE_OPENAI_KEY{suf}") or base_key).strip()
        api_version = (os.getenv(f"AZURE_OPENAI_API_VERSION{suf}") or base_version).strip()
        deployment = (os.getenv(f"{deployment_env_prefix}{suf}") or "").strip()
        if not deployment and suf == "":
            deployment = (os.getenv(deployment_env_prefix) or "").strip()
        if not endpoint or not key or not deployment:
            continue
        try:
            client = openai.AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)
        except Exception:
            continue
        clients.append((client, deployment))
    return clients


def _make_client() -> tuple[Backend, object, str, Callable[[], object | tuple[object, str]] | None]:
    """
    Pick the correct OpenAI client object (plain or Azure) based on env-vars
    and return both the client and, for Azure, a round-robin picker when
    multiple Azure resources are provided. Preference order:
    1) Azure (if any AZURE_* envs are set)
    2) Ollama (if configured)
    3) OpenAI API key
    """
    if openai is None and not (os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL")):
        raise RuntimeError(
            "The 'openai' Python package is required for OpenAI or Azure backends. "
            "Add 'openai>=1.14' to requirements.txt, or configure OLLAMA_MODEL."
        )

    # --- Azure (supports up to three clients via *_2 / *_3 suffixes) ----------
    azure_pool = _azure_clients_from_env("AZURE_OPENAI_DEPLOYMENT")
    if azure_pool:
        rr = cycle(azure_pool)
        def _picker():
            return next(rr)
        client, deployment = azure_pool[0]
        return "azure", client, deployment, _picker

    # --- Ollama auto-detect (set OLLAMA_MODEL or OLLAMA_BASE_URL) -------------
    if os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
        from ollama import Client as _OllamaClient          # type: ignore
        host  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL",   "llama3")
        return "ollama", _OllamaClient(host=host), model, None

    # plain OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return "openai", openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")), "", None

    raise RuntimeError("No LLM credentials found. Set Azure envs, OLLAMA_MODEL, or OPENAI_API_KEY.")


# ─────────────────────────────────────────────────────────────────────────────
# Default system prompts (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ANCHOR_PROMPT_FAMILY = os.getenv("TSCE_ANCHOR_PROMPT_FAMILY", "latent_bias").strip().lower() or "latent_bias"
DEFAULT_ANCHOR_MIN_TOKENS = max(1, int(os.getenv("TSCE_ANCHOR_MIN_TOKENS", "110")))
DEFAULT_ANCHOR_MAX_TOKENS = max(DEFAULT_ANCHOR_MIN_TOKENS, int(os.getenv("TSCE_ANCHOR_MAX_TOKENS", "140")))


@dataclass(frozen=True)
class AnchorPromptFamily:
    name: str
    summary: str
    system_prompt: str
    footer: str


def _token_window_text(min_tokens: int = DEFAULT_ANCHOR_MIN_TOKENS, max_tokens: int = DEFAULT_ANCHOR_MAX_TOKENS) -> str:
    if min_tokens == max_tokens:
        return f"exactly {min_tokens}"
    return f"{min_tokens} to {max_tokens}"


def _build_anchor_template(summary: str, mode_note: str, token_window: str) -> str:
    return (
        "Return one HDA string.\n"
        "Output format: <HDA>BODY</HDA>\n"
        "\n"
        "BODY rules:\n"
        f"1. {token_window} space-separated tokens.\n"
        "2. lowercase letters and spaces only.\n"
        "3. each token must be 4 to 8 letters long.\n"
        "4. each token must be invented, non-dictionary, and semantically blank.\n"
        "5. separate every token with a single space; do not fuse many subparts into one long token.\n"
        "6. do not copy content words from the user request.\n"
        "7. do not copy content words from these instructions.\n"
        "8. no digits, punctuation, quotes, braces, slashes, markup, or code.\n"
        "9. no clause, sentence, explanation, or obvious semantic theme.\n"
        "10. no token may appear more than twice.\n"
        "11. vary token shapes; avoid repeated roots and repeated syllable frames.\n"
        f"12. mode note: {summary} {mode_note}\n"
        "13. internally draft several candidates and output only the best one.\n"
        "\n"
        "Only output the HDA string. No commentary.\n"
        "\n"
        "USER-REQUEST\n"
        "------------\n"
    )


def _build_anchor_footer(reminder: str) -> str:
    return (
        "------------\n"
        "OUTPUT CHECKLIST\n"
        "----------------\n"
        "Return only <HDA>...</HDA>.\n"
        f"Use {_token_window_text()} tokens.\n"
        "Use short invented alphabetic tokens.\n"
        "Keep each token 4 to 8 letters.\n"
        "No English words.\n"
        "No copied request words.\n"
        "No copied instruction words.\n"
        "No phrases or clauses.\n"
        "No repeated roots.\n"
        f"{reminder}\n"
    )


ANCHOR_PROMPT_FAMILIES: Dict[str, AnchorPromptFamily] = {
    "opaque_control": AnchorPromptFamily(
        name="opaque_control",
        summary=(
            "mode alpha."
        ),
        system_prompt=_build_anchor_template(
            "mode alpha.",
            "balanced token lengths with moderate vowel variation.",
            _token_window_text(),
        ),
        footer=_build_anchor_footer("Favor balanced pseudo-tokens with low root reuse."),
    ),
    "latent_bias": AnchorPromptFamily(
        name="latent_bias",
        summary=(
            "mode beta."
        ),
        system_prompt=_build_anchor_template(
            "mode beta.",
            "favor denser multi-syllable pseudo-tokens and stronger shape diversity.",
            _token_window_text(),
        ),
        footer=_build_anchor_footer("Favor dense multi-syllable pseudo-tokens with high shape diversity."),
    ),
    "task_abstract": AnchorPromptFamily(
        name="task_abstract",
        summary=(
            "mode gamma."
        ),
        system_prompt=_build_anchor_template(
            "mode gamma.",
            "favor longer tokens and lower local similarity between neighboring tokens.",
            _token_window_text(),
        ),
        footer=_build_anchor_footer("Favor longer pseudo-tokens and lower local similarity."),
    ),
    "keyboard_drift": AnchorPromptFamily(
        name="keyboard_drift",
        summary=(
            "mode delta."
        ),
        system_prompt=_build_anchor_template(
            "mode delta.",
            "favor sharper consonant-vowel alternation without collapsing into repeated letter patterns.",
            _token_window_text(),
        ),
        footer=_build_anchor_footer("Favor sharper consonant-vowel alternation without repeated letter patterns."),
    ),
}


def available_anchor_prompt_families() -> tuple[str, ...]:
    return tuple(ANCHOR_PROMPT_FAMILIES.keys())


def resolve_anchor_prompt_family(name: str | None = None) -> AnchorPromptFamily:
    family_name = (name or DEFAULT_ANCHOR_PROMPT_FAMILY).strip().lower()
    family = ANCHOR_PROMPT_FAMILIES.get(family_name)
    if family is None:
        choices = ", ".join(available_anchor_prompt_families())
        raise ValueError(f"Unknown anchor prompt family '{family_name}'. Expected one of: {choices}")
    return family


def parse_anchor_prompt_families(raw: str | None) -> tuple[str, ...]:
    value = (raw or DEFAULT_ANCHOR_PROMPT_FAMILY).strip().lower()
    if not value or value == "default":
        return (resolve_anchor_prompt_family().name,)
    if value == "all":
        return available_anchor_prompt_families()

    families: List[str] = []
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        families.append(resolve_anchor_prompt_family(item).name)
    if not families:
        return (resolve_anchor_prompt_family().name,)
    return tuple(dict.fromkeys(families))


def build_anchor_prompt_messages(prompt: str, family_name: str | None = None) -> List[Dict[str, str]]:
    family = resolve_anchor_prompt_family(family_name)
    return [
        {"role": "system", "content": family.system_prompt},
        {"role": "user", "content": f"{prompt}\n\n{family.footer}"},
    ]


DEFAULT_ANCHOR_TEMPLATE = resolve_anchor_prompt_family().system_prompt
anchor_footer = resolve_anchor_prompt_family().footer

DEFAULT_FINAL_PREFIX = (
    "Think first step-by-step internally.\n"
    "Then respond as exactly ONE JSON object with two keys: 'thoughts' and 'answer'.\n"
    "- 'thoughts': your full reasoning (any length; do not limit it).\n"
    "- 'answer': the final result requested by the user. If the task requires a specific JSON schema, put that schema-compliant object INSIDE 'answer'.\n"
    "No code fences, no extra prose before or after the JSON."
)

SECOND_PASS_BRIEF = (
    "The HDA is a control code chosen for latent steering rather than human readability.\n"
    "Treat it as internal guidance only.\n"
    "Do NOT quote, explain, or imitate it in the response.\n"
    "Strictly follow output-format instructions from the user prompt — if JSON is required, output only that JSON with no extra text.\n"
)

_PHI3_ROLE_PREFIX = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def _chat_to_phi3_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI-style chat messages into Phi-3 conversation format."""
    parts: List[str] = []
    for msg in messages:
        role_token = _PHI3_ROLE_PREFIX.get(msg.get("role", "user"), "<|user|>")
        content = msg.get("content", "")
        parts.append(f"{role_token}\n{content}\n<|end|>")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# Public type aliases.
Message = Dict[str, str]
Chat = List[Message]
ClientPicker = Callable[[], object | tuple[object, str]]

DEFAULT_ANCHOR_MODEL = os.getenv("TSCE_ANCHOR_MODEL", "gpt-4o-mini")
DEFAULT_FINAL_MODEL = os.getenv("TSCE_FINAL_MODEL", os.getenv("TSCE_MODEL", "gpt-4.1-mini"))
SUPPORTED_ROLES = {"system", "developer", "user", "assistant", "tool"}


def _is_azure_client(client: object) -> bool:
    if openai is None:
        return False
    azure_cls = getattr(openai, "AzureOpenAI", None)
    return bool(azure_cls and isinstance(client, azure_cls))


def _object_to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _object_to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_object_to_plain(v) for v in value]
    if isinstance(value, tuple):
        return [_object_to_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return _object_to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return _object_to_plain(value.dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {k: _object_to_plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _response_to_dict(response: Any) -> Dict[str, Any]:
    plain = _object_to_plain(response)
    if not isinstance(plain, dict):
        raise TypeError(f"Expected chat completion response to be dict-like, got {type(response).__name__}.")
    return plain


def _choice_content(response: Mapping[str, Any], *, phase: str) -> str:
    try:
        content = response["choices"][0]["message"].get("content")
    except Exception as exc:
        raise RuntimeError(f"TSCEChat: malformed {phase} response; missing choices[0].message.content.") from exc
    if content is None:
        raise RuntimeError(f"TSCEChat: {phase} response content was None.")
    return str(content).strip()


def _usage_dict(response: Mapping[str, Any]) -> Dict[str, Any]:
    usage = response.get("usage") or {}
    plain = _object_to_plain(usage)
    return plain if isinstance(plain, dict) else {}


def _sum_usage(anchor_usage: Mapping[str, Any], final_usage: Mapping[str, Any]) -> Dict[str, int | float]:
    totals: Dict[str, int | float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [
            usage.get(key)
            for usage in (anchor_usage, final_usage)
            if isinstance(usage.get(key), (int, float))
        ]
        if values:
            totals[key] = sum(values)
    return totals


def _usage_with_phase(anchor_usage: Mapping[str, Any], final_usage: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(final_usage)
    merged["anchor"] = dict(anchor_usage)
    merged["final"] = dict(final_usage)
    total = _sum_usage(anchor_usage, final_usage)
    if total:
        merged["total"] = total
    return merged


def _is_openai_error(exc: Exception, class_name: str) -> bool:
    if openai is None:
        return False
    error_cls = getattr(openai, class_name, None)
    return bool(error_cls and isinstance(exc, error_cls))


def _is_transient_error(exc: Exception, httpx_module: Any) -> bool:
    if _is_openai_error(exc, "APITimeoutError") or _is_openai_error(exc, "APIConnectionError"):
        return True
    if _is_openai_error(exc, "APIError"):
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return status in {500, 502, 503, 504}
    if httpx_module is not None:
        transient_types = tuple(
            t
            for t in (
                getattr(httpx_module, "ReadTimeout", None),
                getattr(httpx_module, "ConnectTimeout", None),
                getattr(httpx_module, "RemoteProtocolError", None),
            )
            if t is not None
        )
        return bool(transient_types and isinstance(exc, transient_types))
    return False


def _clean_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


class _AttrDict(dict):
    """Dict with recursive attribute access for OpenAI-shaped responses."""

    def __init__(self, value: Mapping[str, Any]):
        super().__init__((k, self._wrap(v)) for k, v in value.items())

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class TSCEReply:
    """Final response plus auditable TSCE phase metadata."""

    def __init__(
        self,
        *,
        content: str,
        anchor: str,
        anchor_model: str | None = None,
        final_model: str | None = None,
        latency: float = 0.0,
        latency_by_phase: Mapping[str, float] | None = None,
        usage: Mapping[str, Any] | None = None,
        usage_by_phase: Mapping[str, Mapping[str, Any]] | None = None,
        anchor_request: Mapping[str, Any] | None = None,
        final_request: Mapping[str, Any] | None = None,
        raw_anchor: Mapping[str, Any] | None = None,
        raw_final: Mapping[str, Any] | None = None,
        logprobs: Sequence[Any] | None = None,
    ):
        self.content = content
        self.anchor = anchor
        self.anchor_model = anchor_model
        self.final_model = final_model
        self.latency = float(latency)
        self.latency_s = self.latency
        self.latency_by_phase = dict(latency_by_phase or {})
        self.phase_latency = self.latency_by_phase
        self.usage = dict(usage or {})
        self.usage_by_phase = {
            phase: dict(value) for phase, value in (usage_by_phase or {}).items()
        }
        self.anchor_request = dict(anchor_request or {})
        self.final_request = dict(final_request or {})
        self.raw_anchor = dict(raw_anchor or {})
        self.raw_final = dict(raw_final or {})
        self.logprobs = list(logprobs or [])

    def __repr__(self) -> str:
        return (
            f"TSCEReply(content={self.content!r}, anchor={self.anchor!r}, "
            f"anchor_model={self.anchor_model!r}, final_model={self.final_model!r})"
        )


class TSCEChat:
    """
    Two-pass TSCE chat wrapper.

    Call the instance like a function:

    ```py
    reply = TSCEChat()("plain string prompt")
    reply = TSCEChat()([
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ])
    ```
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        anchor_model: str | None = None,
        anchor_prompt: str = DEFAULT_ANCHOR_TEMPLATE,
        final_prefix: str = DEFAULT_FINAL_PREFIX,
        deployment_id: str | None = None,
        client: object | ClientPicker | None = None,
        backend: Backend | None = None,
        phase2_backend: str | None = None,
    ):
        self.anchor_prompt = anchor_prompt
        self.final_prefix = final_prefix
        self.model = model
        self.anchor_model = anchor_model
        self.deployment_id = deployment_id
        self.backend: Backend | None = backend
        self.client: object | None = None
        self._auto_deployment = deployment_id or ""
        self._client_picker: ClientPicker | None = None
        self._lazy_env_client = client is None

        if callable(client):
            self._client_picker = client
        elif client is not None:
            self._set_active_client(client)

        backend_cfg = (phase2_backend or os.getenv("TSCE_PHASE2_BACKEND", "")).strip().lower()
        self.phase2_backend = backend_cfg or "default"
        self._phi3_model = None
        self._phi3_tokenizer = None
        self._phi3_generate = None
        self._phi3_model_id = os.getenv("TSCE_PHI3_MODEL", "mlx-community/Phi-3-mini-4k-instruct-4bit")
        try:
            self._phi3_max_new_tokens = int(os.getenv("TSCE_PHI3_MAX_NEW_TOKENS", "600"))
        except ValueError:
            self._phi3_max_new_tokens = 600

        self._stats: dict[str, Any] = {}
        self._last_req_anchor: Dict[str, Any] | None = None
        self._last_resp_anchor: Dict[str, Any] | None = None
        self._last_req_final: Dict[str, Any] | None = None
        self._last_resp_final: Dict[str, Any] | None = None

    def _set_active_client(self, picked: object | tuple[object, str]) -> None:
        if isinstance(picked, tuple) and len(picked) == 2:
            self.client, self._auto_deployment = picked
        else:
            self.client = picked
        if self.backend is None:
            if _is_azure_client(self.client):
                self.backend = "azure"
            elif hasattr(self.client, "chat") and not hasattr(getattr(self.client, "chat"), "completions"):
                self.backend = "ollama"
            else:
                self.backend = "openai"
        if self.backend == "azure" and self._auto_deployment:
            self.model = self._auto_deployment

    def _ensure_client(self) -> None:
        if self.client is not None:
            return
        if not self._lazy_env_client:
            raise RuntimeError("TSCEChat client is not configured.")
        backend, client, deployment, picker = _make_client()
        self.backend = backend
        self._auto_deployment = deployment
        self._client_picker = picker
        self.client = client

    def _rotate_client(self) -> None:
        if self._client_picker is None:
            return
        self._set_active_client(self._client_picker())

    def _normalize_chat(self, prompt_or_chat: Union[str, Chat]) -> Chat:
        if isinstance(prompt_or_chat, str):
            if not prompt_or_chat.strip():
                raise ValueError("Prompt string must be non-empty.")
            return [{"role": "user", "content": prompt_or_chat}]

        if isinstance(prompt_or_chat, (bytes, bytearray)) or not isinstance(prompt_or_chat, Sequence):
            raise ValueError("Input must be a prompt string or a non-empty sequence of chat message dicts.")
        if not prompt_or_chat:
            raise ValueError("Chat messages must be a non-empty sequence.")

        chat: Chat = []
        for index, message in enumerate(prompt_or_chat):
            if not isinstance(message, Mapping):
                raise ValueError(f"Message at index {index} must be a dict with 'role' and 'content'.")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise ValueError(f"Message at index {index} has invalid 'role'; expected a non-empty string.")
            if role not in SUPPORTED_ROLES:
                choices = ", ".join(sorted(SUPPORTED_ROLES))
                raise ValueError(f"Message at index {index} has unsupported role {role!r}; expected one of: {choices}.")
            if not isinstance(content, str):
                raise ValueError(f"Message at index {index} has invalid 'content'; expected a string.")
            chat.append({"role": role, "content": content})
        return chat

    def __call__(
        self,
        prompt_or_chat: Union[str, Chat],
        *,
        anchor_temp: float | None = None,
        anchor_top_p: float = ANCHOR_TOP_P,
        anchor_max_tokens: int = 500,
        force_anchor: str | None = None,
        final_temperature: float = 0.01,
        final_top_p: float = 1.0,
        final_max_tokens: int | None = None,
        forced_final_model: str | None = None,
        forced_anchor_model_name: str | None = None,
        final_kwargs: Mapping[str, Any] | None = None,
        anchor_kwargs: Mapping[str, Any] | None = None,
    ) -> TSCEReply:
        started = time.perf_counter()
        chat = self._normalize_chat(prompt_or_chat)
        if not any(message["role"] == "user" for message in chat):
            raise ValueError("Chat messages must contain at least one 'user' message.")
        if self._client_picker is None and (force_anchor is None or self.phase2_backend != "phi3"):
            self._ensure_client()

        self._last_req_anchor = None
        self._last_resp_anchor = None
        self._last_req_final = None
        self._last_resp_final = None

        latency_by_phase: Dict[str, float] = {}

        anchor_started = time.perf_counter()
        if force_anchor is not None:
            if not isinstance(force_anchor, str) or not force_anchor.strip():
                raise ValueError("force_anchor must be a non-empty string when provided.")
            anchor_text = force_anchor.strip()
            anchor_model = forced_anchor_model_name or "external"
            anchor_resp = {
                "model": anchor_model,
                "choices": [{"message": {"content": anchor_text}, "finish_reason": "external"}],
                "usage": {},
                "tsce_external_anchor": True,
            }
            self._last_req_anchor = {
                "backend": "external",
                "phase": "anchor",
                "request": {"force_anchor": True},
            }
            self._last_resp_anchor = anchor_resp
        else:
            anchor_messages: Chat = (
                [{"role": "system", "content": self.anchor_prompt}]
                + chat
                + [{"role": "user", "content": anchor_footer}]
            )
            anchor_forced_model = (
                forced_anchor_model_name
                or (
                    None
                    if self._client_picker is not None or self._is_azure_backend() or self.backend == "ollama"
                    else (self.anchor_model or DEFAULT_ANCHOR_MODEL)
                )
            )
            anchor_resp = self._completion_anchor(
                anchor_messages,
                temperature=ANCHOR_TEMP if anchor_temp is None else anchor_temp,
                top_p=anchor_top_p,
                max_tokens=anchor_max_tokens,
                forced_model=anchor_forced_model,
                **dict(anchor_kwargs or {}),
            )
            anchor_text = _choice_content(anchor_resp, phase="anchor")
            anchor_model = anchor_resp.get("model") or anchor_forced_model or self.anchor_model
        latency_by_phase["anchor"] = time.perf_counter() - anchor_started

        final_started = time.perf_counter()
        final_messages: Chat = [
            {"role": "system", "content": f"{anchor_text}\n\n{SECOND_PASS_BRIEF}\n\n{self.final_prefix}"}
        ] + chat
        final_gen_kwargs = {
            "temperature": final_temperature,
            "top_p": final_top_p,
            "logprobs": LOGPROB,
            "top_logprobs": 5 if LOGPROB else None,
            **dict(final_kwargs or {}),
        }
        if final_max_tokens is not None:
            final_gen_kwargs["max_tokens"] = final_max_tokens

        if self.phase2_backend == "phi3":
            final_resp = self._completion_phi3(
                final_messages,
                temperature=final_temperature,
                top_p=final_top_p,
                max_tokens=final_max_tokens or self._phi3_max_new_tokens,
            )
            final_forced_model = None
        else:
            final_forced_model = (
                forced_final_model
                or (
                    None
                    if self._client_picker is not None or self._is_azure_backend() or self.backend == "ollama"
                    else (self.model or DEFAULT_FINAL_MODEL)
                )
            )
            final_resp = self._completion(
                final_messages,
                forced_model=final_forced_model,
                **final_gen_kwargs,
            )
        final_text = _choice_content(final_resp, phase="final")
        final_model = final_resp.get("model") or final_forced_model or self.model
        latency_by_phase["final"] = time.perf_counter() - final_started

        logprobs: list[Any] = []
        choice = (final_resp.get("choices") or [{}])[0]
        choice_logprobs = choice.get("logprobs") if isinstance(choice, Mapping) else None
        if choice_logprobs and isinstance(choice_logprobs, Mapping) and "content" in choice_logprobs:
            logprobs = [_ns(token) for token in choice_logprobs["content"]]

        latency = time.perf_counter() - started
        latency_by_phase["total"] = latency
        anchor_usage = _usage_dict(anchor_resp)
        final_usage = _usage_dict(final_resp)
        usage_by_phase = {"anchor": anchor_usage, "final": final_usage}
        usage = _usage_with_phase(anchor_usage, final_usage)

        self._stats = {
            "latency_s": round(latency, 6),
            "latency_by_phase": {k: round(v, 6) for k, v in latency_by_phase.items()},
            "usage_by_phase": usage_by_phase,
        }

        return TSCEReply(
            content=final_text,
            anchor=anchor_text,
            anchor_model=anchor_model,
            final_model=final_model,
            latency=latency,
            latency_by_phase=latency_by_phase,
            usage=usage,
            usage_by_phase=usage_by_phase,
            anchor_request=self._last_req_anchor,
            final_request=self._last_req_final,
            raw_anchor=self._last_resp_anchor,
            raw_final=self._last_resp_final,
            logprobs=logprobs,
        )

    def _is_azure_backend(self) -> bool:
        return self.backend == "azure" or _is_azure_client(self.client)

    def _model_for_phase(self, phase: str, forced_model: str | None) -> str:
        if forced_model:
            return forced_model
        if self._is_azure_backend():
            model = self.deployment_id or self._auto_deployment or self.model
            if not model:
                raise RuntimeError("Azure backend requires deployment_id or AZURE_OPENAI_DEPLOYMENT.")
            return model
        if self.backend == "ollama":
            return self.model or self._auto_deployment or os.getenv("OLLAMA_MODEL", "llama3")
        if phase == "anchor":
            return self.anchor_model or self.model or DEFAULT_ANCHOR_MODEL
        return self.model or DEFAULT_FINAL_MODEL

    def _completion(
        self,
        messages: List[dict[str, str]],
        *,
        forced_model: str | None = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        return self._completion_phase("final", messages, forced_model=forced_model, **gen_kwargs)

    def _completion_anchor(
        self,
        messages: List[dict[str, str]],
        *,
        forced_model: str | None = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        return self._completion_phase("anchor", messages, forced_model=forced_model, **gen_kwargs)

    def _completion_phase(
        self,
        phase: str,
        messages: List[dict[str, str]],
        *,
        forced_model: str | None = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        if self._client_picker is not None:
            self._rotate_client()
        else:
            self._ensure_client()
        assert self.client is not None

        if self.backend == "ollama":
            return self._completion_ollama(phase, messages, forced_model=forced_model, **gen_kwargs)

        model = self._model_for_phase(phase, forced_model)
        params = _clean_params({"model": model, "messages": messages, **gen_kwargs})
        request_snapshot = {
            "backend": "azure" if self._is_azure_backend() else "openai",
            "phase": phase,
            "request": _object_to_plain(params),
        }
        response = self._call_openai_like(params)
        response.setdefault("model", model)
        self._record_phase(phase, request_snapshot, response)
        return response

    def _completion_ollama(
        self,
        phase: str,
        messages: List[dict[str, str]],
        *,
        forced_model: str | None = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        assert self.client is not None
        model = self._model_for_phase(phase, forced_model)
        mapping = {"temperature": "temperature", "top_p": "top_p", "max_tokens": "num_predict"}
        options = {mapping[key]: value for key, value in gen_kwargs.items() if key in mapping and value is not None}
        request_snapshot = {
            "backend": "ollama",
            "phase": phase,
            "request": {"model": model, "messages": messages, "options": options or None},
        }
        response = self.client.chat(
            model=model,
            messages=messages,
            stream=False,
            options=options or None,
        )
        plain = _response_to_dict(response)
        message = plain.get("message") or {}
        content = message.get("content", "")
        usage = {}
        prompt_tokens = plain.get("prompt_eval_count")
        completion_tokens = plain.get("eval_count")
        if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
            usage = {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
            }
        out = {
            "model": plain.get("model") or model,
            "choices": [{"message": {"content": content}, "finish_reason": plain.get("done_reason", "stop")}],
            "usage": usage,
        }
        self._record_phase(phase, request_snapshot, out)
        return out

    def _call_openai_like(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        assert self.client is not None
        try:
            import httpx  # type: ignore
        except Exception:  # pragma: no cover
            httpx = None  # type: ignore

        timeout = (
            httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
            if httpx is not None
            else 120.0
        )
        max_attempts = 3
        backoff = 0.2
        for attempt in range(1, max_attempts + 1):
            try:
                create = self.client.chat.completions.create
                try:
                    return _response_to_dict(create(**params, timeout=timeout))
                except TypeError as exc:
                    if "timeout" not in str(exc):
                        raise
                    return _response_to_dict(create(**params))
            except Exception as exc:
                if _is_openai_error(exc, "RateLimitError"):
                    raise
                if not _is_transient_error(exc, httpx) or attempt >= max_attempts:
                    raise
                time.sleep(backoff + random.random() * 0.1)
                backoff = min(backoff * 2, 2.0)
        raise RuntimeError("Unreachable retry state in TSCEChat.")

    def _record_phase(self, phase: str, request: Dict[str, Any], response: Dict[str, Any]) -> None:
        if phase == "anchor":
            self._last_req_anchor = request
            self._last_resp_anchor = response
        elif phase == "final":
            self._last_req_final = request
            self._last_resp_final = response
        else:
            raise ValueError(f"Unknown TSCE phase {phase!r}.")

    def _ensure_phi3_loaded(self) -> None:
        if self._phi3_model is not None and self._phi3_tokenizer is not None and self._phi3_generate is not None:
            return
        try:
            from mlx_lm import load as _mlx_load, generate as _mlx_generate  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Phi-3 backend requested but mlx_lm is not installed or failed to import.") from exc
        self._phi3_model, self._phi3_tokenizer = _mlx_load(
            self._phi3_model_id,
            tokenizer_config={"trust_remote_code": True},
        )
        self._phi3_generate = _mlx_generate

    def _completion_phi3(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        self._ensure_phi3_loaded()
        prompt = _chat_to_phi3_prompt(messages)
        request_snapshot = {
            "backend": "phi3-local",
            "phase": "final",
            "request": {
                "model": self._phi3_model_id,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "prompt": prompt,
            },
        }
        try:
            from mlx_lm.sample_utils import make_sampler as _make_sampler  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Phi-3 sampler utilities unavailable; ensure mlx_lm is installed.") from exc

        outputs = self._phi3_generate(
            self._phi3_model,
            self._phi3_tokenizer,
            prompt,
            sampler=_make_sampler(temp=max(0.0, float(temperature)), top_p=float(top_p)),
            max_tokens=max_tokens,
        )
        text = outputs[0] if isinstance(outputs, (list, tuple)) else str(outputs)
        response = {
            "model": self._phi3_model_id,
            "choices": [
                {"message": {"content": text.split("<|end|>")[0].strip()}, "finish_reason": "stop", "logprobs": None}
            ],
            "usage": {},
        }
        self._record_phase("final", request_snapshot, response)
        return response

    def last_stats(self) -> Dict[str, Any]:
        return self._stats


class _TSCECompletionsResource:
    def __init__(self, owner: "TSCEClient"):
        self._owner = owner

    def create(
        self,
        *,
        messages: Union[str, Chat],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        force_anchor: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> _AttrDict:
        if kwargs.get("stream"):
            raise ValueError("TSCEClient does not support streaming responses.")

        passthrough = dict(kwargs)
        passthrough.pop("stream", None)
        max_tokens = passthrough.pop("max_tokens", None)
        wrapper = TSCEChat(
            model=model or self._owner.model,
            anchor_model=self._owner.anchor_model,
            anchor_prompt=self._owner.anchor_prompt,
            final_prefix=self._owner.final_prefix,
            deployment_id=self._owner.deployment_id,
            client=self._owner.client,
            backend=self._owner.backend,
            phase2_backend=self._owner.phase2_backend,
        )
        reply = wrapper(
            messages,
            force_anchor=force_anchor,
            final_temperature=0.01 if temperature is None else temperature,
            final_top_p=1.0 if top_p is None else top_p,
            final_max_tokens=max_tokens,
            forced_final_model=model,
            final_kwargs=passthrough,
        )
        usage = reply.usage_by_phase.get("anchor", {})
        total_usage = _sum_usage(usage, reply.usage_by_phase.get("final", {}))
        response = {
            "id": f"tsce-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": reply.final_model or model or self._owner.model or DEFAULT_FINAL_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": total_usage or reply.usage_by_phase.get("final", {}) or reply.usage,
            "tsce": {
                "anchor": reply.anchor,
                "anchor_model": reply.anchor_model,
                "final_model": reply.final_model,
                "latency": reply.latency,
                "latency_by_phase": reply.latency_by_phase,
                "usage_by_phase": reply.usage_by_phase,
                "anchor_request": reply.anchor_request,
                "final_request": reply.final_request,
                "raw_anchor": reply.raw_anchor,
                "raw_final": reply.raw_final,
                "metadata": dict(metadata or {}),
            },
        }
        return _AttrDict(response)


class _TSCEChatResource:
    def __init__(self, owner: "TSCEClient"):
        self.completions = _TSCECompletionsResource(owner)


class TSCEClient:
    """OpenAI-style adapter exposing client.chat.completions.create(...)."""

    def __init__(
        self,
        *,
        client: object | ClientPicker | None = None,
        model: str | None = None,
        anchor_model: str | None = None,
        anchor_prompt: str = DEFAULT_ANCHOR_TEMPLATE,
        final_prefix: str = DEFAULT_FINAL_PREFIX,
        deployment_id: str | None = None,
        backend: Backend | None = None,
        phase2_backend: str | None = None,
    ):
        self.client = client
        self.model = model
        self.anchor_model = anchor_model
        self.anchor_prompt = anchor_prompt
        self.final_prefix = final_prefix
        self.deployment_id = deployment_id
        self.backend = backend
        self.phase2_backend = phase2_backend
        self.chat = _TSCEChatResource(self)


__all__ = [
    "TSCEChat",
    "TSCEClient",
    "TSCEReply",
    "Message",
    "Chat",
    "build_anchor_prompt_messages",
    "available_anchor_prompt_families",
    "parse_anchor_prompt_families",
    "resolve_anchor_prompt_family",
]
