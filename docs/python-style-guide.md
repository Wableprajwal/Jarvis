---
doc_type: style-guide
date: 2025-11-03
---
# Python style guide

- Formatter and linter: `ruff format` and `ruff check`. Maximum line length is 100 characters.
- Target Python 3.12. All public functions have type hints; `mypy --strict` runs on `app/`.
- Naming: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE` for constants.
  Test functions are named `test_<unit>_<behaviour>`, for example `test_retry_stops_after_five_attempts`.
- Docstrings: one-line summary in the imperative mood. Only document non-obvious behaviour; do not restate the
  signature.
- Async code: never call blocking I/O inside `async def`. Wrap unavoidable blocking calls (file reads, model
  inference) in `asyncio.to_thread`. Never use `time.sleep` in async code; use `asyncio.sleep`.
- Errors: raise specific exceptions, never bare `except:`. Catch broad `Exception` only at a boundary (a tool
  call, a request handler) and convert it to a structured error.
- Dataclasses: prefer `@dataclass(slots=True)` for small value objects. Use pydantic models only at I/O
  boundaries.
- Imports: standard library, third party, then local, separated by blank lines. No wildcard imports.
- Logging: use the `logging` module with lazy `%s` formatting. Never log secrets, tokens or full request bodies.
- Tests: pytest with `asyncio_mode = "auto"`. Each test file must run without network access; mark tests that
  need a live service with `@pytest.mark.live`.
