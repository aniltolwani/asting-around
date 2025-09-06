# asting-around

# AST Runner — Tests Only Pack (v2)

Use this when you want to build **from scratch** and still have a ready acceptance suite.

## How to use
1) Build your implementation in a separate folder, exposing an `api.py` with:
   - `validate_and_instrument(src: str, mode: str = "banlist", op_limit: int = 10000, instrument: bool = True) -> str`
   - `run_one(src: str, entry_func: str, args: tuple, timeout_s=1.0, limits=None) -> dict`
   - `run_many(testcases: list[dict], timeout_s=1.0, limits=None, mode="banlist", op_limit=10000, instrument=True) -> dict`

2) Run the harness and point it at your impl dir:
   ```bash
   python harness.py --impl /path/to/your/impl --mode allowlist
   ```

3) Green twice in a row = move on to a variation.
