"""Rewards for ToolACE references in Qwen's native XML tool-call syntax."""

from __future__ import annotations

import json
import math
import re
from collections import Counter


CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def parse_completion(text: str):
    if "<tool_call" not in text and "<function=" not in text:
        return [], bool(text.strip())
    matches = list(CALL_RE.finditer(text))
    if not matches or len(matches) != text.count("<tool_call>"):
        return [], False
    calls = []
    for match in matches:
        body = match.group(2)
        params = list(PARAM_RE.finditer(body))
        if len(params) != body.count("<parameter="):
            return [], False
        if PARAM_RE.sub("", body).strip():
            return [], False
        arguments = {}
        for param in params:
            name = param.group(1).strip()
            if not name or name in arguments:
                return [], False
            arguments[name] = param.group(2).strip()
        calls.append({"name": match.group(1).strip(), "arguments": arguments})
    return calls, True


def comparable(value):
    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value.casefold()
        if isinstance(parsed, str):
            return parsed.strip().casefold()
        return comparable(parsed)
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [comparable(x) for x in value]
    return value


def f1(correct: int, predicted: int, expected: int):
    if predicted == expected == 0:
        return 1.0
    return 2 * correct / (predicted + expected) if predicted + expected else 0.0


def call_reward(text: str, reference: list[dict], strict: bool = False):
    calls, valid = parse_completion(text)
    if not valid:
        return 0.0
    if not reference:
        return float(not calls and bool(text.strip()))
    if not calls:
        return 0.0
    expected_names = Counter(call["name"] for call in reference)
    actual_names = Counter(call["name"] for call in calls)
    name_matches = sum((expected_names & actual_names).values())
    name_score = f1(name_matches, len(calls), len(reference))
    unused = list(calls)
    matched_args = 0
    total_ref_args = sum(len(x["arguments"]) for x in reference)
    total_pred_args = sum(len(x["arguments"]) for x in calls)
    for expected in reference:
        options = [
            (sum(comparable(actual["arguments"].get(key, object())) == comparable(value)
                 for key, value in expected["arguments"].items()), index)
            for index, actual in enumerate(unused)
            if actual["name"] == expected["name"]
        ]
        if options:
            score, index = max(options)
            matched_args += score
            unused.pop(index)
    arg_score = f1(matched_args, total_pred_args, total_ref_args)
    exact = name_score == 1 and arg_score == 1
    if strict:
        return float(exact)
    return 1.0 if exact else 0.1 + 0.4 * name_score + 0.5 * arg_score


def dense_reward(completions, expected_calls, **_):
    return [call_reward(str(text), json.loads(ref)) for text, ref in zip(completions, expected_calls)]


def strict_reward(completions, expected_calls, **_):
    return [call_reward(str(text), json.loads(ref), strict=True) for text, ref in zip(completions, expected_calls)]


def concise_no_call_reward(completions, expected_calls, completion_ids=None, **_):
    rewards = []
    if completion_ids is None:
        completion_ids = [None] * len(completions)
    for text, reference, token_ids in zip(completions, expected_calls, completion_ids):
        expected = json.loads(reference)
        value = call_reward(str(text), expected)
        if not expected and value:
            length = len(str(text).strip())
            value = math.exp(-max(0, length - 240) / 500)
            if token_ids is not None and len(token_ids) >= 256:
                value *= 0.2
        rewards.append(value)
    return rewards
