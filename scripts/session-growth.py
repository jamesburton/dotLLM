"""Grows one chat session turn by turn against a running dotllm server and records per-turn TTFT, prefix reuse,
decode speed and VRAM. Each turn appends ~TURN_CHARS of wikitext as a new user message, keeps the assistant reply."""
import argparse, json, pathlib, subprocess, sys, time, urllib.error, urllib.request

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--url", default="http://127.0.0.1:18080/v1/chat/completions", help="chat completions endpoint")
ap.add_argument("--model", required=True, help="model id (GET /v1/models, data[0].id)")
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ap.add_argument("--corpus", default=str(REPO_ROOT / ".docs" / "corpora" / "wikitext2-test.raw"), help="text to feed in parts")
ap.add_argument("--turn-chars", type=int, default=34000, help="characters appended per turn (34000 is about 8k tokens)")
ap.add_argument("--turns", type=int, default=18, help="maximum number of turns")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--out", default="growth.tsv")
ap.add_argument("--pid", type=int, default=0, help="server process id: also record its working set and private bytes per turn")
ap.add_argument("--fit", type=int, default=0,
                help="window to grow to (GET /props max_sequence_length): the turn that would overshoot it is shrunk so its "
                     "prompt lands --fit-margin tokens below the window, and becomes the last turn")
ap.add_argument("--fit-margin", type=int, default=1000, help="tokens to leave below --fit on the last turn")
ap.add_argument("--keep-going", action="store_true",
                help="on an error (refusal, timeout) drop that turn's message and keep growing instead of stopping (3 at most)")
args = ap.parse_args()
URL, TURN_CHARS, MAX_TURNS, OUT = args.url, args.turn_chars, args.turns, args.out

text = open(args.corpus, encoding="utf-8").read()


def vram():
    """(used, free) memory of the first GPU in MiB, or (0, 0) when nvidia-smi is missing or fails."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                                      text=True, stderr=subprocess.DEVNULL)
        used, free = (int(x) for x in out.splitlines()[0].split(","))
        return used, free
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return 0, 0


def process_mib(pid):
    """(working set, private bytes) of a process in MiB, or (0, 0) when unavailable. Windows: PowerShell; Linux: /proc."""
    if pid <= 0:
        return 0, 0
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", f"$p = Get-Process -Id {pid}; \"$($p.WorkingSet64) $($p.PrivateMemorySize64)\""],
                text=True, stderr=subprocess.DEVNULL)
            ws, priv = (int(x) for x in out.split())
            return ws >> 20, priv >> 20
        status = pathlib.Path(f"/proc/{pid}/status").read_text()
        kib = {k: int(v.split()[0]) for k, v in (ln.split(":", 1) for ln in status.splitlines()) if k in ("VmRSS", "VmData")}
        return kib.get("VmRSS", 0) >> 10, kib.get("VmData", 0) >> 10
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0, 0


msgs = [{"role": "system", "content": "You are reading a long document in parts. Answer briefly."}]
with open(OUT, "w") as f:
    f.write("turn\tprompt_tok\tcached_tok\tnew_tok\tttft_s\tprefill_ms\tprefill_tps\tdecode_tps\tgen_tok\tvram_mib\tws_mib\tprivate_mib\tvram_free_mib\tend_time\terror\n")
    pos = 0; last_prompt = 0; last_gen = 0; last_turn = False; failures = 0; scale = 1.0
    for turn in range(1, MAX_TURNS + 1):
        take = int(TURN_CHARS * scale)
        if args.fit and last_prompt:
            # Tokens per character of the whole conversation so far (template and replies included) sizes a turn that would
            # overshoot the window, so the last prompt lands just under it.
            tpc = last_prompt / sum(len(m["content"]) for m in msgs)
            room = args.fit - args.fit_margin - last_prompt - last_gen - 40
            if room / tpc < take:
                take = int(room / tpc); last_turn = True
                if take < 1000:
                    break
        chunk = text[pos: pos + take]
        if len(chunk) < take:
            sys.exit(f"corpus {args.corpus} ran out after {turn - 1} turns ({len(text)} chars); lower --turns or --turn-chars")
        msgs.append({"role": "user", "content": f"Part {turn} of the document:\n{chunk}\n\nIn one sentence, what is part {turn} about?"})
        body = json.dumps({"model": args.model, "messages": msgs, "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
                           "stream_options": {"include_usage": True}}).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time(); ttft = None; reply = []; timings = {}; usage = {}; err = ""
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    if d.get("error"):
                        err = json.dumps(d["error"])[:200]
                    for c in d.get("choices") or []:
                        delta = c.get("delta") or {}
                        # Reasoning models stream reasoning_content first: either kind of delta is the first token.
                        if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                            ttft = time.time() - t0
                        if delta.get("content"):
                            reply.append(delta["content"])
                    timings = d.get("timings") or timings
                    usage = d.get("usage") or usage
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as e:  # noqa: BLE001 - record and stop
            err = repr(e)[:200]
        cached = timings.get("cached_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        prompt = timings.get("prompt_tokens", usage.get("prompt_tokens", 0))
        row = [turn, prompt, cached, prompt - cached, f"{ttft or 0:.1f}", f"{timings.get('prefill_time_ms', 0):.0f}",
               f"{timings.get('prefill_tokens_per_sec', 0):.0f}", f"{timings.get('decode_tokens_per_sec', 0):.1f}",
               timings.get("generated_tokens", 0)]
        used, free = vram()
        row += [used, *process_mib(args.pid), free, time.strftime("%Y-%m-%dT%H:%M:%S"), err.replace("\t", " ")]
        f.write("\t".join(map(str, row)) + "\n"); f.flush()
        print("\t".join(map(str, row)), flush=True)
        if err:
            failures += 1
            if not args.keep_going or failures >= 3:
                break
            msgs.pop()  # the session's prefix stays the last accepted request
            pos += take; scale /= 2; last_turn = False  # fresh text, half the size
            continue
        pos += take; last_prompt, last_gen = prompt, timings.get("generated_tokens", 0)
        msgs.append({"role": "assistant", "content": "".join(reply)})
        if last_turn:
            break
