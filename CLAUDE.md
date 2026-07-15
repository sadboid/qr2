# Claude Code preferences

## Model routing for rate-limit efficiency

- **Codebase scouting / exploration** (finding files, locating symbols/definitions,
  broad "where is X" searches, open-ended multi-query lookups): delegate to the
  `Explore` subagent with an explicit `model: "haiku"` override — do not run
  these inline or on the default session model.
  - Total token count may be similar or even higher than doing it inline; the
    point is that Haiku consumes far less of the rate-limit usage budget than
    Sonnet/Opus, so routing scouting there preserves headroom for the
    higher-value work (writing, judgment calls, synthesis) that needs the
    stronger model.
  - Applies whenever the task fits Explore's read-only search profile. Tasks
    that need judgment, code review, or multi-step reasoning still use the
    default/general-purpose agent on the inherited model unless told otherwise.
