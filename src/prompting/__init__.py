"""Prompt construction: rendering, versioned templates, builder. Never calls a model."""

from src.prompting.builder import Prompt, PromptTooLarge, build_prompt, build_prompt_for_request
from src.prompting.render import render_context_block, render_peak_table
from src.prompting.templates import FULL_VERSION, PROMPT_VERSION, system_prompt

__all__ = [
    "FULL_VERSION",
    "PROMPT_VERSION",
    "Prompt",
    "PromptTooLarge",
    "build_prompt",
    "build_prompt_for_request",
    "render_context_block",
    "render_peak_table",
    "system_prompt",
]
