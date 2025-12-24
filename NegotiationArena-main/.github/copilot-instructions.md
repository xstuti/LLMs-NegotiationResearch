### Purpose
This file gives actionable, repository-specific directions to AI coding agents so they can be immediately productive in the NegotiationArena codebase.

### Big picture (what to know first)
- Main package: `negotiationarena/` — contains game engine primitives (`alternating_game.py`, `parser.py`, `agent_message.py`, `utils.py`, `constants.py`) and `agents/` implementations.
- Scenarios are implemented under `games/` (e.g. `games/buy_sell_game/`, `games/trading_game/`, `games/ultimatum/`). Each scenario defines a `game.py` and a `prompt.py` (or similar) that produce the structured prompt sent to agents.
- Runners: `runner/` contains top-level scripts that wire agents + games (e.g. `runner/buysell_main.py`, `runner/trading_main.py`). Use these to reproduce experiments.
- Data flow: Runner -> instantiate Agents -> instantiate Game -> Game.init_players() calls parser.instantiate_prompt() -> Agent.init_agent() -> Agent.chat() -> Parser extracts tagged fields via `negotiationarena.utils.extract_multiple_tags` and `parser.ExchangeGameDefaultParser`.

### Key repository conventions & patterns
- Tag-based, structured messages: agents are expected to reply using XML-like tags (e.g. `<my resources>...</my resources>`, `<player answer>...</player answer>`). See `games/*/prompt.py` and `negotiationarena/constants.py` for canonical tags and ordering.
- Agents are stateful and hold conversation history in memory. Do NOT reuse Agent instances across games/rounds — create a fresh Agent per player per game. See `negotiationarena/agents/*` for `init_agent` and `update_conversation_tracking` usage.
- Parsers expect exact tag ordering and formats. When adding features, update both `prompt.py` (to instruct the model) and `parser.py` (to parse the same tags). Look at `games/buy_sell_game/game.py` and `parser.ExchangeGameDefaultParser` for examples.
- Resource limitation: `games/buy_sell_game` currently supports exactly one tradable resource (explicit check in `BuySellGame.__init__`). If you add multi-resource support, update prompts and rendering accordingly.

### How to run / developer workflow
- Environment: runners call `load_dotenv(".env.local")` or `load_dotenv(".env")` — put secrets in `.env.local` (this file is in `.gitignore`). Required keys (examples found across runners):
  - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, optional `ANY_SCALE` for LLaMA/other backends.
- Quick run (PowerShell):
  - `python .\runner\buysell_main.py`
  - `python .\runner\trading_main.py`
  - `python .\runner\ultimatum_main.py`
- Logs: runners use `log_dir` (e.g. `./.logs/buysell`) — inspect `.logs/` for serialized game objects / chat logs.
- Formatting and linting: `pyproject.toml` contains `black` settings. Use `pre-commit` if available in CI.

### Integration points & dependencies
- External LLM clients used in `negotiationarena/agents/`:
  - OpenAI Python client (`openai`) — `ChatGPTAgent`.
  - Google `google-generativeai` — `GeminiAgent`.
  - Anthropic (`anthropic`) and other backends referenced in `pyproject.toml`.
- If you change agent API calls, keep the agent's `conversation` representation and `chat()` return type compatible with the parser (string containing tagged fields).

### Small-but-critical details (avoid common mistakes)
- Tag extraction is implemented with simple string search (`utils.get_tag_indices`). Tags must match exactly; whitespace differences may break parsing. When updating prompts or tags, update all parsers and tests.
- `AgentMessage` separates `public` and `secret` parts. Public messages are rendered and sent to the other player via `message_to_other_player()` — maintain this separation for privacy/reasoning fields.
- Proposal counting: proposals limits are enforced by `maximum_number_of_proposals=self.iterations // 2 - 1` in game instantiation. Changing iteration logic requires updating that calculation.
- Avoid changing how `player_roles` and `player_social_behaviour` are concatenated into the system/user prompts — many tests and experiments rely on predictable prompt composition.

### Adding a new game / agent
- To add a game, follow `games/buy_sell_game` pattern:
  1. Create `games/<new_game>/game.py` with a parser class that subclasses `GameParser` or `ExchangeGameDefaultParser`.
 2. Create `games/<new_game>/prompt.py` that returns a single string with the required tag structure.
 3. Add a runner in `runner/` wiring agents and settings; reuse `AgentMessage` and parser utilities.
- To add an agent backend: implement `chat()` that returns a single string with the model's response; `__deepcopy__` is used in several places — copy semantics should preserve serializability (e.g. replace client objects with class names when deep-copying).

### Files to inspect first (examples)
- `negotiationarena/constants.py` — canonical tags and tokens
- `games/buy_sell_game/prompt.py` — canonical prompt + required response order
- `games/buy_sell_game/game.py` — parser + game lifecycle
- `negotiationarena/parser.py` and `negotiationarena/utils.py` — parsing helpers
- `negotiationarena/agents/chatgpt.py`, `gemini.py`, `claude.py` — agent patterns
- `runner/*` — runnable examples to reproduce experiments

### Limitations & tests
- There are no automated tests in the repo root. Before making wide refactors, run a representative runner (e.g. `buysell_main.py`) with small iterations and inspect `./.logs/` to validate behavior.

If any of the above assumptions are incomplete or you need examples for a particular change (new game, new agent, parser change), tell me which area and I'll expand the instructions or add a small runnable example.
