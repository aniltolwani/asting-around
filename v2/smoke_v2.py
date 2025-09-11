# smoke_v2.py
import sandbox_v2 as S

CASES = [
    {"src":"def solution():\n  return 1", "func":"solution", "args":[]},
    {"src":"def solution():\n  while True: pass", "func":"solution", "args":[]},
    {"src":"def solution():\n  return sum([1,2,3])", "func":"solution", "args":[]},
]

if __name__ == "__main__":
    out = S.run_batch(CASES, mode="allow", timeout_s=0.5,
                      limits={"cpu_s":1,"as_bytes":64*1024*1024,"nofile":64},
                      op_limit=10_000, instrument=True, max_workers=3)
    from pprint import pprint
    pprint(out)
