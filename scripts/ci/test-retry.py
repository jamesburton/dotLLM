#!/usr/bin/env python3
"""Run `dotnet test`; if it fails, rerun ONLY the failed tests once (#2052).

    test-retry.py --label NAME [--max-retry-tests N] -- dotnet test <project> <args...>

Policy (docs/PACKAGING.md, "Flaky tests"):
  * One retry, never a loop. A test that fails twice fails the job.
  * Build failures are never retried (no TRX is produced and the output carries MSBuild errors).
  * Failed tests are rerun by METHOD (xUnit Theory cases share one method, so the whole Theory reruns: a name diff that
    keeps the `(arg: ...)` suffix would drop cases, see the 2026-09-24 note).
  * A test-host crash/abort/hang cannot be filtered by name, so the WHOLE project is rerun once.
  * A "pass" on retry must be proven: every first-attempt failed method has to appear as Passed in the retry's TRX. An
    over-narrow filter makes `dotnet test` exit 0 with "no test matches", which would otherwise be a silent false green.
  * More than --max-retry-tests failed methods is treated as a real regression (not a flake) and is not retried.
Every retry is logged, annotated, and written to $GITHUB_STEP_SUMMARY ("PASSED ON RETRY").
"""
import argparse, os, re, subprocess, sys
import xml.etree.ElementTree as ET

NS = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, tag):
    print(f"[{tag}] {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = []
    for line in p.stdout:
        sys.stdout.write(line)
        lines.append(line)
    sys.stdout.flush()
    return p.wait(), "".join(lines)


def with_logging(args, results_dir, trx_name):
    return args + ["--logger", f"trx;LogFileName={trx_name}", "--results-directory", results_dir, "--blame-crash"]


def method_key(class_name, name):
    name = name.split("(")[0].strip()
    if class_name and not name.startswith(class_name + "."):
        name = f"{class_name}.{name}"
    return name


def parse_trx(results_dir):
    """Return {method_key: [(outcome, message, testName), ...]} across every TRX in results_dir, or None if there is none."""
    out, found = {}, False
    for root_dir, _, files in os.walk(results_dir):
        for f in files:
            if not f.endswith(".trx"):
                continue
            found = True
            root = ET.parse(os.path.join(root_dir, f)).getroot()
            defs = {}
            for ut in root.iterfind(".//t:UnitTest", NS):
                tm = ut.find("t:TestMethod", NS)
                if tm is not None:
                    defs[ut.get("id")] = method_key(tm.get("className", ""), tm.get("name", ut.get("name", "")))
            for r in root.iterfind(".//t:UnitTestResult", NS):
                key = defs.get(r.get("testId")) or method_key("", r.get("testName", "?"))
                msg = r.findtext(".//t:ErrorInfo/t:Message", default="", namespaces=NS).strip()
                out.setdefault(key, []).append((r.get("outcome"), msg, r.get("testName", key)))
    return out if found else None


def load_flakes():
    rows = []
    with open(os.path.join(HERE, "known-flakes.tsv"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                pat, issue, label = line.rstrip("\n").split("\t")
                rows.append((re.compile(pat), issue, label))
    return rows


def flake_link(key, flakes):
    for rx, issue, label in flakes:
        if rx.search(key):
            return f"[#{issue}](https://github.com/jamesburton/dotLLM/issues/{issue}) ({label})"
    return "no known flake issue"


def summary(text):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def trunc(s, n=600):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-retry-tests", type=int, default=30)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    base = os.environ.get("RUNNER_TEMP") or os.path.join(os.getcwd(), "artifacts")
    d1 = os.path.join(base, "test-results", a.label, "attempt1")
    d2 = os.path.join(base, "test-results", a.label, "attempt2")

    rc, out1 = run(with_logging(cmd, d1, f"{a.label}-attempt1.trx"), f"{a.label} attempt 1")
    if rc == 0:
        return 0
    res1 = parse_trx(d1)
    crashed = bool(re.search(r"(test host process crashed|The active test run was aborted|Test Run Aborted|"
                             r"Test host process exited unexpectedly|blame-hang|Test Run Hang)", out1, re.I))
    if res1 is None and re.search(r"Build FAILED|error (CS|MSB|NETSDK|NU)\d+", out1):
        print(f"::error::[{a.label}] build/restore failure - not retried")
        return rc
    failed = sorted(k for k, v in (res1 or {}).items() if any(o == "Failed" for o, _, _ in v))
    flakes = load_flakes()

    if crashed or res1 is None or not failed:
        mode, retry_cmd = "whole project (test host crash/abort/hang, not filterable by name)", list(cmd)
        reason = "test host crashed or aborted" if crashed or res1 is None else "non-zero exit with no failed test in the TRX"
    else:
        if len(failed) > a.max_retry_tests:
            print(f"::error::[{a.label}] {len(failed)} failed test methods (> {a.max_retry_tests}): treated as a regression, not retried")
            return rc
        mode, reason = f"{len(failed)} failed test method(s)", "failed tests"
        flt = "|".join(f"FullyQualifiedName~{k}" for k in failed)
        retry_cmd, i = [], 0
        while i < len(cmd):
            if cmd[i] == "--filter" and i + 1 < len(cmd):
                flt = f"({cmd[i + 1]})&({flt})"
                i += 2
            else:
                retry_cmd.append(cmd[i])
                i += 1
        retry_cmd += ["--filter", flt]
    if "--no-build" not in retry_cmd:
        retry_cmd.append("--no-build")  # attempt 1 built it; a build is never retried

    gh_env = os.environ.get("GITHUB_ENV")
    if gh_env:  # keeps the TRX/blame artifact when a retry rescues the job (the upload step checks it)
        with open(gh_env, "a", encoding="utf-8") as fh:
            fh.write("TEST_RETRIED=true\n")

    md = [f"### Test retry: `{a.label}`", "", f"Attempt 1 failed ({reason}); retrying once: {mode}.", ""]
    print(f"::warning title=Test retry ({a.label})::attempt 1 failed ({reason}); retrying once: {mode}")
    for k in failed:
        first = next((m for o, m, _ in res1[k] if o == "Failed"), "")
        print(f"  RETRY {k} | {flake_link(k, flakes)} | first failure: {trunc(first, 300)}")
        md.append(f"- `{k}` - {flake_link(k, flakes)}\n  <details><summary>first-attempt failure</summary>\n\n  ```\n  {trunc(first)}\n  ```\n  </details>")
    if not failed:
        md.append("- no individual test names available (host crash); see the blame/crash artifact")
    if crashed or res1 is None:
        # A host crash names no test, so match the known-flake patterns against the crash output instead (e.g. #2050).
        for rx, issue, label in flakes:
            if rx.search(out1):
                link = f"[#{issue}](https://github.com/jamesburton/dotLLM/issues/{issue}) ({label})"
                print(f"  attempt-1 output matches known flake {link}")
                md.append(f"- attempt-1 output matches {link}")

    rc2, _ = run(with_logging(retry_cmd, d2, f"{a.label}-attempt2.trx"), f"{a.label} retry")
    res2 = parse_trx(d2) or {}
    ok = rc2 == 0
    if ok and failed:
        missing = [k for k in failed if not any(o == "Passed" for o, _, _ in res2.get(k, []))]
        if missing:
            ok = False
            print(f"::error::[{a.label}] retry exited 0 but did not run these as Passed (filter matched nothing?): {missing}")
            md.append(f"\n**Retry exited 0 but did not prove these passed: {', '.join(missing)}**")
    if ok:
        md.insert(0, f"## :warning: PASSED ON RETRY - `{a.label}`\n")
        md.append("\nThe job is green only because the retry passed. Investigate/track the flake; do not ignore it.")
        print(f"::warning title=PASSED ON RETRY ({a.label})::{len(failed) or 'host-crash'} retried; all passed on attempt 2")
        summary("\n".join(md))
        return 0
    still = sorted(k for k, v in res2.items() if any(o == "Failed" for o, _, _ in v))
    md.insert(0, f"## :x: FAILED AFTER RETRY - `{a.label}`\n")
    md.append("\nStill failing on the retry: " + (", ".join(f"`{k}`" for k in still) or "(see log)"))
    summary("\n".join(md))
    print(f"::error::[{a.label}] failed after the single retry")
    return rc2 or 1


if __name__ == "__main__":
    sys.exit(main())
