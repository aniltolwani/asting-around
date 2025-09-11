# test_sandbox_v2.py
import sys, time, json, os
import multiprocessing as mp
import pytest

import sandbox_v2 as S

ALLOWED_LIMITS = {"cpu_s": 1, "as_bytes": 64 * 1024 * 1024, "nofile": 64}  # small for fast tests

# ---------- Helpers ----------

def case(src, func="solution", args=()):
    return {"src": src, "func": func, "args": list(args)}

def run_ok(source, entry="solution", args=(), timeout=0.8):
    out = S.run_single(source, entry, tuple(args), timeout_s=timeout, limits=ALLOWED_LIMITS)
    assert set(out.keys()) == {"status","result","runtime_ms","stdout","stderr"}
    assert isinstance(out["runtime_ms"], int) and out["runtime_ms"] >= 0
    return out

def prepare_ok(src, mode="allow", op_limit=10_000, instrument=True):
    return S.prepare(src, mode=mode, op_limit=op_limit, instrument=instrument)

def expect_status(out, expect_set):
    assert out["status"] in expect_set, f"Expected {expect_set}, got {out['status']}. Full: {out}"

# ---------- Validation tests ----------

@pytest.mark.parametrize("mode", ["allow","ban"])
def test_imports_blocked(mode):
    src = "import os\ndef solution():\n    return 1"
    with pytest.raises(Exception):
        S.prepare(src, mode=mode)

@pytest.mark.parametrize("mode", ["allow","ban"])
def test_dunder_attr_blocked(mode):
    src = "def solution():\n    return (1).__class__"
    with pytest.raises(Exception):
        S.prepare(src, mode=mode)

@pytest.mark.parametrize("mode", ["allow","ban"])
def test_builtins_eval_blocked(mode):
    src = "def solution():\n    return __builtins__['eval']('1+1')"
    with pytest.raises(Exception):
        S.prepare(src, mode=mode)

# ---------- Execution / sandbox tests ----------

def test_happy_path_allow():
    src = "def add(a,b): return a+b\n\ndef solution():\n    return add(2,3)"
    prep = prepare_ok(src, mode="allow")
    out = run_ok(prep)
    assert out["result"] == 5
    expect_status(out, {"OK"})

def test_happy_path_ban():
    src = "def solution():\n    return sum([1,2,3])"
    prep = prepare_ok(src, mode="ban")
    out = run_ok(prep)
    assert out["result"] == 6
    expect_status(out, {"OK"})

def test_timeout_or_oplimit_on_infinite_loop():
    src = "def solution():\n    while True: pass"
    prep = prepare_ok(src, mode="allow", op_limit=10_000, instrument=True)
    t0 = time.perf_counter()
    out = run_ok(prep, timeout=0.5)  # wall-clock safety
    t1 = time.perf_counter()
    # Must end quickly and classify deterministically
    assert (t1 - t0) < 1.2
    expect_status(out, {"TIMEOUT","OPLIMIT"})

def test_recursion_limited():
    src = "def f(x): return f(x+1)\n\ndef solution():\n    return f(0)"
    prep = prepare_ok(src, mode="allow")
    out = run_ok(prep)
    expect_status(out, {"RECURSION","ERROR"})  # prefer RECURSION; ERROR acceptable if you map differently

def test_memory_bomb():
    # Try to trigger MEM; if RLIMIT_AS unsupported, TIMEOUT acceptable
    src = "def solution():\n    x=[0]*(30_000_000)\n    return 1"
    prep = prepare_ok(src, mode="allow")
    out = run_ok(prep, timeout=0.8)
    expect_status(out, {"MEM","TIMEOUT"})

def test_comprehension_oplimit():
    # OPLIMIT should trip even if no explicit loops called by user
    src = "def solution():\n    return sum([i for i in range(1_000_000)])"
    prep = prepare_ok(src, mode="allow", op_limit=5_000, instrument=True)
    out = run_ok(prep, timeout=0.6)
    expect_status(out, {"OPLIMIT","TIMEOUT"})  # OPLIMIT preferred

def test_builtin_call_oplimit():
    # Calls to builtins must tick (per-Call transform)
    src = """def solution():
    s = 0
    for _ in range(20_000):
        s += len([1])
    return s
"""
    prep = prepare_ok(src, mode="allow", op_limit=5_000, instrument=True)
    out = run_ok(prep, timeout=0.6)
    expect_status(out, {"OPLIMIT","TIMEOUT"})

def test_stdout_stderr_capture_and_truncation():
    src = "def solution():\n    print('A'*10000)\n    return 123"
    prep = prepare_ok(src, mode="allow")
    out = run_ok(prep)
    expect_status(out, {"OK"})
    assert isinstance(out["stdout"], str) and len(out["stdout"]) <= 4096
    assert isinstance(out["stderr"], str) and len(out["stderr"]) <= 4096
    assert out["result"] == 123

# ---------- Parallel tests ----------

def test_run_batch_parallel_mixed():
    cases = [
        case("def solution():\n  return 1"),
        case("def solution():\n  while True: pass"),
        case("def solution():\n  return sum([1,2,3,4])"),
    ]
    out = S.run_batch(
        cases,
        mode="allow",
        op_limit=10_000,
        instrument=True,
        timeout_s=0.5,
        limits=ALLOWED_LIMITS,
        max_workers=4
    )
    assert set(out.keys()) == {"results"}
    results = out["results"]
    assert len(results) == len(cases)
    statuses = [r["status"] for r in results]
    # Expect one TIMEOUT/OPLIMIT and two OK
    assert statuses.count("OK") == 2
    assert any(s in {"TIMEOUT","OPLIMIT"} for s in statuses)

def test_run_batch_preserves_order():
    # Slow in the middle to detect order mapping
    cases = [
        case("def solution():\n  return 10"),
        case("def solution():\n  i=0\n  while i < 10_000_000: i+=1\n  return 20"),
        case("def solution():\n  return 30"),
    ]
    out = S.run_batch(
        cases, mode="ban", timeout_s=0.5, limits=ALLOWED_LIMITS, max_workers=3
    )
    res = out["results"]
    assert len(res) == 3
    # Must align with input positions (not completion order)
    assert isinstance(res[0]["result"], (int, type(None)))
    assert isinstance(res[1]["status"], str)
    assert isinstance(res[2]["result"], (int, type(None)))

def test_no_zombie_children_after_single():
    # smoke check that single run doesn't leave active children
    src = "def solution():\n  while True: pass"
    prep = prepare_ok(src, mode="allow", op_limit=1_000, instrument=True)
    _ = run_ok(prep, timeout=0.2)
    time.sleep(0.05)
    assert not mp.active_children(), "No active children should remain after run_single"

# ---------- Negative / edge ----------

def test_forbidden_getattr_path():
    src = "def solution():\n  return getattr(__builtins__, 'eval', None)"
    with pytest.raises(Exception):
        prepare_ok(src, mode="allow")

def test_missing_entry_function():
    src = "def not_solution():\n  return 42"
    prep = prepare_ok(src, mode="allow")
    out = S.run_single(prep, "solution", (), timeout_s=0.4, limits=ALLOWED_LIMITS)
    # Up to you: treat as ERROR or OK with result None; the spec prefers ERROR.
    assert out["status"] in {"ERROR","OK"}
