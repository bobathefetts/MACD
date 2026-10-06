# Contributing to MACD

Thanks for taking a look. Bug reports, results from real models, and pull requests are all welcome.

## Set up

```bash
git clone https://github.com/bobathefetts/MACD.git
cd MACD
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

To work on the Hugging Face backend, also run `pip install -e ".[hf]"`. Its tests build a tiny model
locally, so they need no downloads and no GPU.

## Before opening a pull request

```bash
ruff check . && ruff format .
pytest
```

CI runs the same checks on Python 3.10 to 3.13.

## Ground rules

- **Keep the test split untouched.** Nothing may use `test` to make a decision. There is a test that enforces this.
- **Keep runs reproducible.** Use the controller's seeded RNG or `stable_hash`; never the built-in `hash()` or an unseeded `random`.
- **No unverified claims.** If the README says something works, a test or a shared run should back it up.
- **Add a test with every behaviour change.**

## Sharing results

If you run MACD on a real model, open an issue with:

- the model and backend config
- the task config and dataset
- the run's `history.json` and `config.json`

Negative results are useful too.
