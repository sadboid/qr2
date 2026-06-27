"""
Claude CLI bridge — calls the running Claude Code session as an LLM engine.

No ANTHROPIC_API_KEY required. Uses `claude -p` subprocess (non-interactive mode)
which authenticates through the existing Claude Code session.

Usage:
    from .claude_cli import call, is_available

    if is_available():
        text = call("Write 5 search queries for...", timeout=60)
        text = call("Generate section...", system="You are an academic writer...")
"""

import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_CLAUDE_BIN: Optional[str] = None


def is_available() -> bool:
    """Return True if the claude CLI is in PATH (i.e. running inside Claude Code)."""
    global _CLAUDE_BIN
    if _CLAUDE_BIN is None:
        _CLAUDE_BIN = shutil.which("claude") or ""
    return bool(_CLAUDE_BIN)


def call(
    prompt: str,
    system: str = None,
    timeout: int = 120,
) -> str:
    """
    Send a prompt to Claude via CLI subprocess and return the response text.

    Combines system + prompt into a single stdin payload so the call is fully
    self-contained and does not share context with the interactive session
    (--no-session-persistence).

    Args:
        prompt:  The user message.
        system:  Optional system-level instructions prepended to the payload.
        timeout: Seconds before the call is killed (default 120).

    Returns:
        Stripped response text from Claude.

    Raises:
        RuntimeError: CLI not found, non-zero exit code, or empty response.
    """
    bin_path = _CLAUDE_BIN or shutil.which("claude")
    if not bin_path:
        raise RuntimeError("claude CLI not found in PATH — is this running inside Claude Code?")

    payload = f"{system}\n\n---\n\n{prompt}" if system else prompt

    result = subprocess.run(
        [bin_path, "-p", "--output-format", "text", "--no-session-persistence"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        err = result.stderr.strip()[:300]
        raise RuntimeError(f"claude CLI exited {result.returncode}: {err}")

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("claude CLI returned empty response")

    logger.debug(f"[ClaudeCLI] prompt={len(payload)} chars → response={len(output)} chars")
    return output
