# liq-evolution

Trading strategy evolution layer for the LIQ Stack.

## Design phase 0 foundations

- CLI entrypoint uses `typer`.
- Progress/output rendering uses `rich`.
- Data integration is based on `polars` `Series`/`DataFrame`.
- Indicator sources use `liq-ta`.
- Validation and domain contracts use `pydantic`.
- Structured logs use `structlog`.

## Public entry points

- `liq.evolution.config` for configuration models
- `liq.evolution.program` for program AST contract
- `liq.evolution.primitives` and `liq.evolution.adapters` for trading integration
