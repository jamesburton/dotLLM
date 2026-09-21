#!/usr/bin/env python3
"""
sdk_conformance.py — official OpenAI + Anthropic SDK conformance matrix (#453).

Drives a running dotLLM server with the *vendor* SDKs (`openai`, `anthropic`)
pointed at it via ``base_url``, so the claim "more compatible than most" becomes
a measurement rather than an opinion.

The suite is EXPECTED to start mostly red: each sibling issue (#448 Anthropic
/v1/messages, #449 count_tokens + SSE events, #450 models.retrieve /
stream_options / parallel_tool_calls, #451 /v1/embeddings, #452 error envelopes
and rate-limit headers) turns specific rows green. Rows carry the issue number
that should flip them, so the matrix doubles as the acceptance checklist.

Design notes
------------
* **The SDK does not validate responses.** ``openai`` builds models with
  ``construct()``, so a missing field silently becomes ``None`` and a broken
  response still "works". Every row therefore asserts the discriminating fields
  itself; a row must fail against the broken form to be worth anything.
* **``max_retries=0`` everywhere.** Both SDKs retry 429/5xx by default, which
  would both slow the 429 rows down and hide what they measure.
* **Four-state classification, and NOT_EXERCISED is never a pass.**
  ``NOT_IMPLEMENTED`` is reserved for a genuine 404 from the SDK
  (``NotFoundError``) — the endpoint does not exist yet. ``NOT_EXERCISED`` means
  the harness could not provoke the condition the row measures at all (no 429 is
  reachable, so "429 maps to a typed exception" was never put to the test).
  Anything else — 500, 400, wrong shape, timeout — is ``FAIL`` with the reason
  attached. ``SKIP`` is for rows the operator turned off. A row that silently
  degrades to "green because nothing happened" is the exact failure mode of
  #417/#420/#421, so the three non-pass states are kept distinct in the report.

Usage
-----
    python tests/conformance/sdk_conformance.py                 # launch + run
    python tests/conformance/sdk_conformance.py --base-url http://127.0.0.1:18080
    python tests/conformance/sdk_conformance.py --json-out m.json --md-out m.md

See README.md in this directory for setup, pins, and the CI gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import openai
    import anthropic
except ImportError as exc:  # pragma: no cover - operator error
    print(f"error: {exc}. Install the pinned SDKs:\n"
          f"    pip install -r {Path(__file__).with_name('requirements.txt')}",
          file=sys.stderr)
    raise SystemExit(2)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 18453  # distinct from scripts/test_server.py's 18080

# Fixture resolution follows CLAUDE.md's model-storage rules: environment
# override first, then the shared ~/.dotllm/models cache. Never a copy in-tree.
FIXTURE_ENV = "DOTLLM_CONFORMANCE_GGUF"
FIXTURE_CANDIDATES = [
    # (repo, quant) — a tool-calling chat template is required for the tool rows,
    # so a base model such as SmolLM-135M is deliberately NOT in this list.
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Q8_0"),
    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "Q8_0"),
    ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "q8_0"),
]

DUMMY_KEY = "dotllm-conformance"

PASS = "PASS"
FAIL = "FAIL"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
NOT_EXERCISED = "NOT_EXERCISED"
SKIP = "SKIP"


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------

class RowNotImplemented(Exception):
    """Raised (or derived from a 404) when the endpoint does not exist yet."""


class RowNotExercised(Exception):
    """Raised when the condition under test could never be provoked.

    Distinct from FAIL: the server did not misbehave, the harness simply could
    not put it in the state the row claims to measure (e.g. no 429 is reachable).
    Reporting this as a PASS would be the exact vacuous-test failure mode that
    #417/#420/#421 were about.
    """


@dataclass
class Row:
    id: str
    sdk: str
    case: str
    flipped_by: str          # sibling issue expected to turn this row green
    fn: Callable[["Ctx"], str]
    status: str = ""
    detail: str = ""
    elapsed: float = 0.0


@dataclass
class Ctx:
    base_url: str
    model: str
    oai: Any
    ant: Any
    attempt_429: bool = False
    notes: list[str] = field(default_factory=list)


def classify(exc: BaseException) -> tuple[str, str]:
    """Map an exception onto (status, detail). 404 alone means not-implemented."""
    if isinstance(exc, RowNotImplemented):
        return NOT_IMPLEMENTED, str(exc)
    if isinstance(exc, RowNotExercised):
        return NOT_EXERCISED, str(exc)
    if isinstance(exc, (openai.NotFoundError, anthropic.NotFoundError)):
        # Only an *unrouted* 404 means "not implemented". ASP.NET's unmatched-route
        # 404 carries no body; an implemented endpoint 404-ing for, say, an unknown
        # model id carries a JSON error body — that is a FAIL of the row, not a
        # missing feature. Without this split, #448/#450 landing would turn a
        # wrong-id 404 into a green-adjacent NOT-IMPL.
        if getattr(exc, "body", None) in (None, "", {}):
            return NOT_IMPLEMENTED, "404, empty body — route not registered"
        return FAIL, f"HTTP 404 with a body (route exists, request rejected): {_short(exc.body)}"
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        body = getattr(exc, "body", None)
        return FAIL, f"HTTP {exc.status_code}: {_short(body if body is not None else exc.message)}"
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return FAIL, f"connection error: {_short(exc)}"
    return FAIL, f"{type(exc).__name__}: {_short(exc)}"


def _short(v: Any, n: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# OpenAI rows
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}


def oai_chat(c: Ctx) -> str:
    r = c.oai.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=16, temperature=0, seed=0,
    )
    check(r.object == "chat.completion", f"object={r.object!r}")
    check(bool(r.id), "missing id")
    check(len(r.choices) == 1, f"choices={len(r.choices)}")
    ch = r.choices[0]
    check(ch.message.role == "assistant", f"role={ch.message.role!r}")
    check(bool(ch.message.content and ch.message.content.strip()), "empty content")
    check(ch.finish_reason in ("stop", "length"), f"finish_reason={ch.finish_reason!r}")
    return f"content={_short(ch.message.content, 40)!r} finish={ch.finish_reason}"


def oai_stream(c: Ctx) -> str:
    stream = c.oai.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "Count: one two three."}],
        max_tokens=24, temperature=0, seed=0, stream=True,
    )
    text, finish, n = "", None, 0
    for chunk in stream:
        n += 1
        check(chunk.object == "chat.completion.chunk", f"chunk.object={chunk.object!r}")
        if not chunk.choices:
            continue
        d = chunk.choices[0]
        if d.delta and d.delta.content:
            text += d.delta.content
        if d.finish_reason:
            finish = d.finish_reason
    check(n > 1, f"only {n} chunk(s)")
    check(bool(text.strip()), "no content deltas")
    check(finish is not None, "no chunk carried finish_reason")
    return f"{n} chunks, {len(text)} chars, finish={finish}"


def oai_tool_single(c: Ctx) -> str:
    r = c.oai.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=96, temperature=0, seed=0,
    )
    ch = r.choices[0]
    calls = ch.message.tool_calls or []
    check(len(calls) >= 1, "no tool_calls returned")
    tc = calls[0]
    check(bool(tc.id), "tool_call.id missing/empty")
    check(tc.type == "function", f"tool_call.type={tc.type!r}")
    check(tc.function.name == "get_weather", f"name={tc.function.name!r}")
    args = json.loads(tc.function.arguments)
    check(isinstance(args, dict), "arguments is not a JSON object")
    check(ch.finish_reason == "tool_calls", f"finish_reason={ch.finish_reason!r} (want 'tool_calls')")
    return f"{tc.function.name}({_short(tc.function.arguments, 40)}) finish={ch.finish_reason}"


def oai_tool_parallel(c: Ctx) -> str:
    r = c.oai.chat.completions.create(
        model=c.model,
        messages=[{"role": "user",
                   "content": "Look up the weather in Paris and in Tokyo. "
                              "Call the tool once per city."}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        parallel_tool_calls=True,
        max_tokens=192, temperature=0, seed=0,
    )
    calls = r.choices[0].message.tool_calls or []
    check(len(calls) >= 2, f"{len(calls)} tool call(s), want >= 2")
    check(len({t.id for t in calls}) == len(calls), "tool_call ids are not distinct")
    cities = {json.loads(t.function.arguments).get("location", "").lower() for t in calls}
    check(len(cities) >= 2, f"tool calls did not name distinct cities: {cities}")
    return f"{len(calls)} calls: {sorted(cities)}"


def oai_json_object(c: Ctx) -> str:
    r = c.oai.chat.completions.create(
        model=c.model,
        messages=[{"role": "user",
                   "content": "Return a JSON object with keys 'city' and 'country' for Paris."}],
        response_format={"type": "json_object"},
        max_tokens=64, temperature=0, seed=0,
    )
    content = r.choices[0].message.content or ""
    obj = json.loads(content)
    check(isinstance(obj, dict), "response is not a JSON object")
    return f"parsed {_short(content, 50)}"


def oai_json_schema(c: Ctx) -> str:
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
        "required": ["city", "population"],
        "additionalProperties": False,
    }
    r = c.oai.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "Describe Paris."}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "city_info", "strict": True, "schema": schema},
        },
        max_tokens=64, temperature=0, seed=0,
    )
    obj = json.loads(r.choices[0].message.content or "")
    check(isinstance(obj, dict), "not a JSON object")
    check("city" in obj and "population" in obj, f"missing required keys: {sorted(obj)}")
    check(isinstance(obj["population"], int), "population is not an integer")
    check(set(obj) <= {"city", "population"}, f"additionalProperties leaked: {sorted(obj)}")
    return _short(json.dumps(obj), 60)


def oai_logprobs(c: Ctx) -> str:
    r = c.oai.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=8, temperature=0, seed=0,
        logprobs=True, top_logprobs=3,
    )
    lp = r.choices[0].logprobs
    check(lp is not None, "choices[0].logprobs is null")
    check(bool(lp.content), "logprobs.content is empty")
    e = lp.content[0]
    check(e.token is not None, "entry.token missing")
    check(isinstance(e.logprob, float), f"entry.logprob={e.logprob!r}")
    check(e.bytes is not None, "entry.bytes missing (OpenAI sends the UTF-8 bytes)")
    check(e.top_logprobs is not None and len(e.top_logprobs) == 3,
          f"top_logprobs len={len(e.top_logprobs) if e.top_logprobs else None}, want 3")
    return f"{len(lp.content)} tokens, top_logprobs={len(e.top_logprobs)}"


def oai_usage_nonstream(c: Ctx) -> str:
    r = c.oai.chat.completions.create(
        model=c.model, messages=[{"role": "user", "content": "Hi."}],
        max_tokens=8, temperature=0, seed=0,
    )
    u = r.usage
    check(u is not None, "usage missing")
    check(u.prompt_tokens > 0, f"prompt_tokens={u.prompt_tokens}")
    check(u.completion_tokens > 0, f"completion_tokens={u.completion_tokens}")
    check(u.total_tokens == u.prompt_tokens + u.completion_tokens,
          f"total {u.total_tokens} != {u.prompt_tokens}+{u.completion_tokens}")
    return f"prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}"


def oai_usage_stream(c: Ctx) -> str:
    stream = c.oai.chat.completions.create(
        model=c.model, messages=[{"role": "user", "content": "Hi."}],
        max_tokens=8, temperature=0, seed=0,
        stream=True, stream_options={"include_usage": True},
    )
    usage_chunks = []
    for chunk in stream:
        if chunk.usage is not None:
            usage_chunks.append(chunk)
    check(len(usage_chunks) == 1,
          f"{len(usage_chunks)} chunk(s) carried usage, want exactly 1 (the final one)")
    final = usage_chunks[0]
    check(final.choices == [], "the usage chunk must have an empty choices array")
    check(final.usage.total_tokens > 0, "usage.total_tokens is 0")
    return f"final chunk usage total={final.usage.total_tokens}"


def oai_usage_stream_unrequested(c: Ctx) -> str:
    """Without stream_options.include_usage, OpenAI sends no usage on any chunk."""
    stream = c.oai.chat.completions.create(
        model=c.model, messages=[{"role": "user", "content": "Hi."}],
        max_tokens=8, temperature=0, seed=0, stream=True,
    )
    carrying = [i for i, ch in enumerate(stream) if ch.usage is not None]
    check(not carrying,
          f"{len(carrying)} chunk(s) carried usage although include_usage was not "
          f"requested (indices {carrying[:5]})")
    return "no unrequested usage"


def oai_models_list(c: Ctx) -> str:
    page = c.oai.models.list()
    models = list(page)
    check(len(models) >= 1, "empty model list")
    m = models[0]
    check(m.object == "model", f"entry.object={m.object!r} (want 'model')")
    check(bool(m.id), "entry.id empty")
    check(bool(m.owned_by), "entry.owned_by missing (required by the OpenAI Model schema)")
    check(m.created and m.created > 0, f"entry.created={m.created!r}")
    return f"{len(models)} model(s), first={m.id}"


def oai_models_retrieve(c: Ctx) -> str:
    m = c.oai.models.retrieve(c.model)
    check(m.id == c.model, f"id={m.id!r} != requested {c.model!r}")
    check(m.object == "model", f"object={m.object!r}")
    check(bool(m.owned_by), "owned_by missing")
    return f"retrieved {m.id}"


def oai_embeddings(c: Ctx) -> str:
    r = c.oai.embeddings.create(model=c.model, input="hello world")
    check(r.object == "list", f"object={r.object!r}")
    check(len(r.data) == 1, f"{len(r.data)} embedding(s)")
    e = r.data[0]
    check(e.object == "embedding", f"data[0].object={e.object!r}")
    check(isinstance(e.embedding, list) and len(e.embedding) > 0, "empty embedding vector")
    check(r.usage is not None and r.usage.prompt_tokens > 0, "usage.prompt_tokens missing")
    return f"dim={len(e.embedding)}"


def oai_error_envelope(c: Ctx) -> str:
    """A 400 must carry OpenAI's {"error": {"message", "type", ...}} envelope."""
    try:
        c.oai.chat.completions.create(
            model=c.model, messages=[{"role": "user", "content": "hi"}], max_tokens=-1,
        )
    except openai.BadRequestError as e:
        # The SDK unwraps the response's "error" member into `e.body`, so a *string*
        # here means the server sent the flat {"error": "<message>"} shape instead of
        # OpenAI's {"error": {"message", "type", "param", "code"}} envelope (#452).
        err = e.body
        check(not isinstance(err, str),
              f"flat error envelope: the server sent a bare string, "
              f"want an object with message/type/param/code — got {_short(err)}")
        if isinstance(err, dict) and isinstance(err.get("error"), dict):
            err = err["error"]
        check(isinstance(err, dict),
              f'error member is {type(err).__name__}, want an object; got {_short(e.body)}')
        check(isinstance(err.get("message"), str) and err["message"], "error.message missing")
        check(isinstance(err.get("type"), str) and err["type"], "error.type missing")
        return f"error.type={err['type']!r}"
    raise AssertionError("max_tokens=-1 was accepted; expected HTTP 400")


def oai_rate_limit(c: Ctx) -> str:
    if not c.attempt_429:
        raise RowNotExercised(
            "NOT EXERCISED — no 429 could be provoked at any configured limit: "
            "`dotllm serve` never populates ServerOptions.RateLimit and there is no config "
            "binding for it, so RateLimitMiddleware is never wired into the pipeline. "
            "Re-run with --attempt-429 against an externally hosted rate-limited server.")
    err = _burst_until_429(c.base_url, "/v1/chat/completions", {
        "model": c.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
    })
    if err is None:
        raise RowNotExercised("burst produced no 429 — server is not rate limited")
    try:
        c.oai.chat.completions.create(
            model=c.model, messages=[{"role": "user", "content": "hi"}], max_tokens=8)
    except openai.RateLimitError as e:
        errobj = e.body
        check(not isinstance(errobj, str),
              f"flat error envelope on the 429: {_short(errobj)} (#452)")
        if isinstance(errobj, dict) and isinstance(errobj.get("error"), dict):
            errobj = errobj["error"]
        check(isinstance(errobj, dict), f"error member not an object: {_short(e.body)}")
        check(isinstance(errobj.get("message"), str) and errobj["message"], "error.message missing")
        check("rate" in str(errobj.get("type", "")).lower(),
              f"error.type={errobj.get('type')!r}, want a rate-limit type")
        check("retry-after" in {k.lower() for k in e.response.headers},
              "Retry-After header missing")
        return f"RateLimitError, type={errobj.get('type')!r}"
    raise AssertionError("no RateLimitError raised despite an observed 429")


def _burst_until_429(base_url: str, path: str, body: dict, tries: int = 12) -> int | None:
    data = json.dumps(body).encode()
    for _ in range(tries):
        req = urllib.request.Request(base_url + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return 429
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Anthropic rows
# ---------------------------------------------------------------------------

ANT_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


def ant_message(c: Ctx) -> str:
    m = c.ant.messages.create(
        model=c.model, max_tokens=16,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
    )
    check(m.type == "message", f"type={m.type!r}")
    check(m.role == "assistant", f"role={m.role!r}")
    check(bool(m.content) and m.content[0].type == "text", "no text content block")
    check(bool(m.content[0].text.strip()), "empty text")
    check(m.stop_reason in ("end_turn", "max_tokens", "stop_sequence"), f"stop_reason={m.stop_reason!r}")
    check(m.usage.input_tokens > 0 and m.usage.output_tokens > 0, "usage tokens are 0")
    return f"{_short(m.content[0].text, 40)!r} stop={m.stop_reason}"


def ant_stream(c: Ctx) -> str:
    seen: list[str] = []
    text = ""
    with c.ant.messages.stream(
        model=c.model, max_tokens=24,
        messages=[{"role": "user", "content": "Count: one two three."}],
    ) as stream:
        for event in stream:
            seen.append(event.type)
            if event.type == "content_block_delta" and getattr(event.delta, "text", None):
                text += event.delta.text
    required = ["message_start", "content_block_start", "content_block_delta",
                "content_block_stop", "message_delta", "message_stop"]
    missing = [e for e in required if e not in seen]
    check(not missing, f"missing SSE event types: {missing} (saw {sorted(set(seen))})")
    check(bool(text.strip()), "no text accumulated")
    return f"{len(seen)} events, {len(text)} chars"


def ant_tool_single(c: Ctx) -> str:
    m = c.ant.messages.create(
        model=c.model, max_tokens=96,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[ANT_WEATHER_TOOL],
        tool_choice={"type": "tool", "name": "get_weather"},
    )
    uses = [b for b in m.content if b.type == "tool_use"]
    check(len(uses) >= 1, f"no tool_use block (blocks: {[b.type for b in m.content]})")
    u = uses[0]
    check(bool(u.id), "tool_use.id empty")
    check(u.name == "get_weather", f"name={u.name!r}")
    check(isinstance(u.input, dict), "tool_use.input is not an object")
    check(m.stop_reason == "tool_use", f"stop_reason={m.stop_reason!r} (want 'tool_use')")
    return f"{u.name}({_short(json.dumps(u.input), 40)})"


def ant_structured(c: Ctx) -> str:
    """Anthropic's structured output idiom: a forced tool with the target schema."""
    tool = {
        "name": "emit_city",
        "description": "Emit structured information about a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
            "required": ["city", "population"],
        },
    }
    m = c.ant.messages.create(
        model=c.model, max_tokens=96,
        messages=[{"role": "user", "content": "Describe Paris."}],
        tools=[tool], tool_choice={"type": "tool", "name": "emit_city"},
    )
    uses = [b for b in m.content if b.type == "tool_use"]
    check(len(uses) == 1, f"{len(uses)} tool_use blocks, want 1")
    data = uses[0].input
    check("city" in data and "population" in data, f"missing keys: {sorted(data)}")
    check(isinstance(data["population"], int), "population is not an integer")
    return _short(json.dumps(data), 60)


def ant_count_tokens(c: Ctx) -> str:
    r = c.ant.messages.count_tokens(
        model=c.model,
        messages=[{"role": "user", "content": "The quick brown fox jumps over the lazy dog."}],
    )
    check(r.input_tokens > 0, f"input_tokens={r.input_tokens}")
    return f"input_tokens={r.input_tokens}"


def ant_usage(c: Ctx) -> str:
    m = c.ant.messages.create(
        model=c.model, max_tokens=8,
        messages=[{"role": "user", "content": "Hi."}],
    )
    u = m.usage
    check(u is not None, "usage missing")
    check(u.input_tokens > 0, f"input_tokens={u.input_tokens}")
    check(u.output_tokens > 0, f"output_tokens={u.output_tokens}")
    return f"in={u.input_tokens} out={u.output_tokens}"


def ant_rate_limit(c: Ctx) -> str:
    if not c.attempt_429:
        raise RowNotExercised(
            "NOT EXERCISED, twice over: (1) no 429 is reachable at all (see the "
            "openai/error.429 row); (2) even once rate limiting is wired, "
            "RateLimitMiddleware.IsMeteredPath is a hardcoded allowlist of "
            "/v1/chat/completions, /v1/completions and /v1/embeddings — /v1/messages is "
            "NOT on it, so an Anthropic request can never be rate limited. This row must "
            "never be reported green until that allowlist covers /v1/messages.")
    err = _burst_until_429(c.base_url, "/v1/messages", {
        "model": c.model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
    })
    if err is None:
        raise RowNotExercised("burst produced no 429 on /v1/messages")
    try:
        c.ant.messages.create(model=c.model, max_tokens=8,
                              messages=[{"role": "user", "content": "hi"}])
    except anthropic.RateLimitError as e:
        body = e.body
        check(isinstance(body, dict), f"body is {type(body).__name__}")
        check(body.get("type") == "error", f'top-level type={body.get("type")!r}, want "error"')
        errobj = body.get("error")
        check(isinstance(errobj, dict), f'body["error"] not an object: {_short(body)}')
        check(errobj.get("type") == "rate_limit_error", f'error.type={errobj.get("type")!r}')
        check("retry-after" in {k.lower() for k in e.response.headers}, "Retry-After missing")
        return "RateLimitError with rate_limit_error envelope"
    raise AssertionError("no RateLimitError raised despite an observed 429")


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

def build_rows() -> list[Row]:
    return [
        # --- openai -------------------------------------------------------
        Row("openai/chat.completion", "openai", "plain completion", "—", oai_chat),
        Row("openai/chat.stream", "openai", "streaming", "—", oai_stream),
        Row("openai/tool.single", "openai", "tool call (single, forced)", "—", oai_tool_single),
        Row("openai/tool.parallel", "openai", "tool call (parallel)", "#450", oai_tool_parallel),
        Row("openai/json.object", "openai", "JSON output (json_object)", "—", oai_json_object),
        Row("openai/json.schema", "openai", "structured output (json_schema, strict)", "—", oai_json_schema),
        Row("openai/logprobs", "openai", "logprobs + top_logprobs", "—", oai_logprobs),
        Row("openai/usage.nonstream", "openai", "token counting via usage (non-stream)", "—", oai_usage_nonstream),
        Row("openai/usage.stream", "openai", "token counting via stream_options.include_usage", "#450", oai_usage_stream),
        Row("openai/usage.stream.unrequested", "openai", "no usage when include_usage is absent", "#450", oai_usage_stream_unrequested),
        Row("openai/models.list", "openai", "model list", "—", oai_models_list),
        Row("openai/models.retrieve", "openai", "model retrieve", "#450", oai_models_retrieve),
        Row("openai/embeddings", "openai", "embeddings", "#451", oai_embeddings),
        Row("openai/error.envelope", "openai", "400 error envelope", "#452", oai_error_envelope),
        Row("openai/error.429", "openai", "429 → typed exception", "#452", oai_rate_limit),
        # --- anthropic ----------------------------------------------------
        Row("anthropic/messages", "anthropic", "plain completion", "#448", ant_message),
        Row("anthropic/messages.stream", "anthropic", "streaming (SSE event types)", "#448/#449", ant_stream),
        Row("anthropic/tool.single", "anthropic", "tool call (single, forced)", "#448", ant_tool_single),
        Row("anthropic/structured", "anthropic", "structured output (forced tool)", "#448", ant_structured),
        Row("anthropic/count_tokens", "anthropic", "token counting via count_tokens", "#449", ant_count_tokens),
        Row("anthropic/usage", "anthropic", "usage on the message response", "#448", ant_usage),
        Row("anthropic/error.429", "anthropic", "429 → typed exception", "#452", ant_rate_limit),
    ]


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def resolve_fixture() -> tuple[str, str | None]:
    """Resolve the GGUF via DOTLLM_CONFORMANCE_GGUF, then ~/.dotllm/models."""
    override = os.environ.get(FIXTURE_ENV)
    if override:
        if not Path(override).is_file():
            raise SystemExit(f"{FIXTURE_ENV}={override} does not exist")
        return override, None
    models_dir = Path.home() / ".dotllm" / "models"
    for repo, quant in FIXTURE_CANDIDATES:
        repo_dir = models_dir / Path(repo)
        if not repo_dir.is_dir():
            continue
        hits = [f for f in repo_dir.glob("*.gguf") if quant.lower() in f.name.lower()]
        if hits:
            return repo, quant
    raise SystemExit(
        f"no conformance fixture found. Set {FIXTURE_ENV} to a GGUF path, or "
        f"`dotllm model pull` one of: " + ", ".join(r for r, _ in FIXTURE_CANDIDATES))


def cli_entrypoint() -> list[str]:
    """Locate the *built* CLI, not `dotnet run`.

    `dotnet run` spawns the app as a grandchild, so on Windows ``terminate()``
    kills only the launcher and the real server survives holding the port. The
    next run's ``wait_ready`` would then latch onto that orphan and silently
    measure a stale build.
    """
    out = REPO_ROOT / "src" / "DotLLM.Cli" / "bin" / "Release"
    for tfm in sorted(out.glob("net*"), reverse=True):
        exe = tfm / ("DotLLM.Cli.exe" if sys.platform == "win32" else "DotLLM.Cli")
        if exe.is_file():
            return [str(exe)]
        dll = tfm / "DotLLM.Cli.dll"
        if dll.is_file():
            return ["dotnet", str(dll)]
    raise SystemExit("CLI not built — run: dotnet build src/DotLLM.Cli -c Release")


def start_server(model: str, quant: str | None, port: int, device: str) -> subprocess.Popen:
    cmd = cli_entrypoint() + ["serve", model, "--port", str(port), "--device", device,
                              "--no-ui", "--no-browser"]
    if quant:
        cmd += ["--quant", quant]
    print(f"[server] {' '.join(cmd[1:])}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace")


def port_already_serving(base_url: str) -> bool:
    """True if something already answers /health — an orphan from a prior run."""
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(base_url: str, timeout: float = 300) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def discover_model_id(base_url: str) -> str:
    with urllib.request.urlopen(base_url + "/v1/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SYMBOL = {PASS: "PASS", FAIL: "FAIL", NOT_IMPLEMENTED: "NOT-IMPL",
          NOT_EXERCISED: "NOT-EXERCISED", SKIP: "SKIP"}


def render_markdown(rows: list[Row], meta: dict) -> str:
    out = ["# dotLLM SDK conformance matrix (#453)", "",
           f"- generated: {meta['generated']}",
           f"- commit: `{meta['commit']}`",
           f"- model: `{meta['model']}`  (fixture: `{meta['fixture']}`, device: {meta['device']})",
           f"- python: {meta['python']}",
           f"- openai: `{meta['openai']}`  anthropic: `{meta['anthropic']}`", "",
           "| row | sdk | case | status | flipped by | detail |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        detail = r.detail.replace("|", "\\|")
        out.append(f"| `{r.id}` | {r.sdk} | {r.case} | **{SYMBOL[r.status]}** | "
                   f"{r.flipped_by} | {detail} |")
    counts = {s: sum(1 for r in rows if r.status == s) for s in SYMBOL}
    out += ["", "**Totals:** " + ", ".join(f"{SYMBOL[s]}={counts[s]}" for s in SYMBOL if counts[s])]
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", help="Run against an already-running server (skips launch).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", help="Override the model id sent to the SDKs.")
    p.add_argument("--timeout", type=float, default=180.0, help="Per-request SDK timeout (s).")
    p.add_argument("--only", help="Comma-separated row id substrings to run.")
    p.add_argument("--attempt-429", action="store_true",
                   help="Actually drive the 429 rows (requires a rate-limited server).")
    p.add_argument("--json-out", type=Path)
    p.add_argument("--md-out", type=Path)
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero unless every row PASSes (default: exit 0 — the "
                        "baseline is expected to be red while the sibling issues land).")
    args = p.parse_args()

    proc = None
    base_url = args.base_url
    fixture = "(external server)"
    try:
        if base_url is None:
            base_url = f"http://127.0.0.1:{args.port}"
            if port_already_serving(base_url):
                print(f"error: something already answers {base_url}/health — most likely an "
                      f"orphaned server from an earlier run. Kill it (a stale build would be "
                      f"measured silently) or pass --base-url to target it deliberately.",
                      file=sys.stderr)
                return 2
            model_arg, quant = resolve_fixture()
            fixture = f"{model_arg}" + (f" [{quant}]" if quant else "")
            proc = start_server(model_arg, quant, args.port, args.device)
            print(f"[server] waiting for {base_url} ...")
            if not wait_ready(base_url):
                proc.terminate()
                out, _ = proc.communicate(timeout=10)
                print((out or "")[-2000:])
                return 2
            print("[server] ready")

        model_id = args.model or discover_model_id(base_url)
        oai = openai.OpenAI(base_url=f"{base_url}/v1", api_key=DUMMY_KEY,
                            max_retries=0, timeout=args.timeout)
        ant = anthropic.Anthropic(base_url=base_url, api_key=DUMMY_KEY,
                                  max_retries=0, timeout=args.timeout)
        ctx = Ctx(base_url=base_url, model=model_id, oai=oai, ant=ant,
                  attempt_429=args.attempt_429)

        rows = build_rows()
        if args.only:
            wanted = [s.strip() for s in args.only.split(",") if s.strip()]
            for r in rows:
                if not any(w in r.id for w in wanted):
                    r.status, r.detail = SKIP, "filtered out by --only"

        print(f"\n{'row':<28} {'status':<9} {'time':>7}  detail")
        print("-" * 110)
        for r in rows:
            if r.status == SKIP:
                continue
            t0 = time.monotonic()
            try:
                r.detail = r.fn(ctx)
                r.status = PASS
            except BaseException as exc:  # noqa: BLE001 - classification is the point
                r.status, r.detail = classify(exc)
            r.elapsed = time.monotonic() - t0
            print(f"{r.id:<28} {SYMBOL[r.status]:<9} {r.elapsed:>6.1f}s  {_short(r.detail, 60)}")
        print("-" * 110)

        meta = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": _git_head(),
            "model": model_id,
            "fixture": fixture,
            "device": args.device,
            "python": sys.version.split()[0],
            "openai": openai.__version__,
            "anthropic": anthropic.__version__,
        }
        counts = {s: sum(1 for r in rows if r.status == s) for s in SYMBOL}
        print(", ".join(f"{SYMBOL[s]}={counts[s]}" for s in SYMBOL if counts[s]))

        md = render_markdown(rows, meta)
        if args.md_out:
            args.md_out.write_text(md, encoding="utf-8")
            print(f"[out] {args.md_out}")
        if args.json_out:
            args.json_out.write_text(json.dumps(
                {"meta": meta,
                 "rows": [{"id": r.id, "sdk": r.sdk, "case": r.case, "status": r.status,
                           "flipped_by": r.flipped_by, "detail": r.detail,
                           "elapsed_s": round(r.elapsed, 3)} for r in rows]},
                indent=2), encoding="utf-8")
            print(f"[out] {args.json_out}")

        if args.strict:
            return 0 if counts[FAIL] == 0 and counts[NOT_IMPLEMENTED] == 0 and counts[NOT_EXERCISED] == 0 else 1
        return 0
    finally:
        if proc is not None:
            print("[server] shutting down")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
