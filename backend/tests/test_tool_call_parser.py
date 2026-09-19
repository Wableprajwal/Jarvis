import json

import pytest

from app.providers.tool_call_parser import PromptedToolCallParser, render_tool_calls, render_tool_instructions

TOOLS = [
    {
        "name": "read_file",
        "description": "read",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {"name": "get_time", "description": "time", "input_schema": {"type": "object", "properties": {}}},
]


@pytest.fixture
def parser():
    return PromptedToolCallParser(TOOLS)


def test_fenced_json_block(parser):
    r = parser.parse('```json\n{"tool": "read_file", "arguments": {"path": "a.py"}}\n```')
    assert r.error is None and r.attempted
    assert [(c.name, c.arguments) for c in r.tool_calls] == [("read_file", {"path": "a.py"})]
    assert r.tool_calls[0].id


def test_plain_text_is_a_final_answer(parser):
    r = parser.parse("The answer is 42.")
    assert not r.attempted and r.tool_calls == [] and r.error is None and r.text == "The answer is 42."


def test_prose_around_block_is_preserved_separately(parser):
    r = parser.parse('Let me look.\n```json\n{"tool": "get_time", "arguments": {}}\n```\nThanks')
    assert r.tool_calls[0].name == "get_time"
    assert "Let me look." in r.text


def test_multiple_calls_in_array(parser):
    r = parser.parse('```json\n[{"tool":"get_time","arguments":{}},{"tool":"read_file","arguments":{"path":"x"}}]\n```')
    assert [c.name for c in r.tool_calls] == ["get_time", "read_file"]


def test_tool_calls_envelope_and_name_parameters_aliases(parser):
    r = parser.parse('```json\n{"tool_calls": [{"name": "read_file", "parameters": {"path": "x"}}]}\n```')
    assert r.error is None and r.tool_calls[0].arguments == {"path": "x"}


def test_unfenced_tool_json_is_recovered(parser):
    r = parser.parse('{"tool": "get_time", "arguments": {}}')
    assert r.error is None and r.tool_calls[0].name == "get_time"


def test_trailing_comma_tolerated(parser):
    r = parser.parse('```json\n{"tool": "read_file", "arguments": {"path": "a",},}\n```')
    assert r.error is None and r.tool_calls[0].arguments == {"path": "a"}


def test_arguments_given_as_json_string(parser):
    r = parser.parse('```json\n{"tool": "read_file", "arguments": "{\\"path\\": \\"a\\"}"}\n```')
    assert r.error is None and r.tool_calls[0].arguments == {"path": "a"}


@pytest.mark.parametrize(
    "text, fragment",
    [
        ('```json\n{"tool": "read_file", "arguments": {"path": \n```', "not valid JSON"),
        ('```json\n{"tool": "delete_everything", "arguments": {}}\n```', "unknown tool 'delete_everything'"),
        ('```json\n{"tool": "read_file", "arguments": {}}\n```', "invalid arguments for 'read_file'"),
        ('```json\n{"tool": "read_file", "arguments": {"path": 5}}\n```', "invalid arguments for 'read_file'"),
        ('```json\n{"arguments": {"path": "a"}}\n```', 'missing the "tool" name'),
        ('```json\n"just a string"\n```', "expected a JSON object"),
        ('```json\n[]\n```', "expected a JSON object"),
        ('```json\n{"tool": "read_file", "arguments": {"path": "a", "extra": 1}}\n```', "invalid arguments"),
    ],
)
def test_malformed_output_yields_error_not_exception(parser, text, fragment):
    r = parser.parse(text)
    assert r.tool_calls == [] and r.attempted
    assert fragment in r.error


def test_one_bad_call_rejects_the_batch(parser):
    r = parser.parse('```json\n[{"tool":"get_time","arguments":{}},{"tool":"nope","arguments":{}}]\n```')
    assert r.tool_calls == [] and "unknown tool 'nope'" in r.error


def test_none_and_empty_input(parser):
    assert parser.parse(None).tool_calls == []
    assert parser.parse("   ").error is None


def test_render_roundtrip(parser):
    rendered = render_tool_calls([{"id": "1", "name": "read_file", "arguments": {"path": "z"}}])
    r = parser.parse(rendered)
    assert r.tool_calls[0].arguments == {"path": "z"}


def test_instructions_include_every_schema():
    text = render_tool_instructions(TOOLS)
    assert "read_file" in text and "get_time" in text and "```json" in text
    assert json.loads(text.splitlines()[-1])["name"] == "get_time"
