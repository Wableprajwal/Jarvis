from __future__ import annotations

from app.agents.base import SubAgent


class CodeAgent(SubAgent):
    name = "code"
    description = (
        "Answers questions about the local codebase, reads source files, generates unit tests "
        "for a function or file, and summarizes git diffs into PR descriptions."
    )
    system_prompt = (
        "You are the Code agent of Jarvis, an engineering assistant. You work on the user's local "
        "codebase using tools. Read files with read_file before making claims about their contents. "
        "Never invent file contents. If a tool returns an error, adapt or report it honestly. "
        "Answer concisely and precisely."
    )
    tool_names = ["read_file"]
