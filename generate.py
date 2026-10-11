# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.40", "requests>=2.31", "tiktoken>=0.7"]
# ///
# SPDX-FileCopyrightText: Copyright (c) 2026 MBZUAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic Semantic Packing: generate length-controlled, search-grounded research conversations.

A research agent answers a seed question with `search` / `get_document` calls against a
search360 server (https://github.com/ifm-ai/search360). After every burst of tool calls it
replies to the user, and an LLM-simulated user with a random persona asks a follow-up.
This repeats until the conversation reaches STOP_FRACTION * target_tokens, then the agent
writes a final answer.

    uv run generate.py --search_url http://localhost:8000 --llm_base_url http://localhost:8001/v1 \
        --model deepseek-ai/DeepSeek-V3.1 --questions_file seed_questions.jsonl
"""

import argparse
import json
import random
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
import tiktoken
from openai import OpenAI

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

STOP_FRACTION = 0.8  # stop researching at this fraction of target_tokens; the final answer adds the rest
MAX_TURNS = 200
BURST_SIZES = [4, 5, 6, 8]  # tool calls between user follow-ups, sampled per burst
SEARCH_K = 5
MAX_DOC_WORDS = 8_000
QUERY_SIMILARITY = 0.85  # reject a search whose word-overlap (Jaccard) with a previous one is above this
AGENT_TEMPERATURE = 0.7
USER_TEMPERATURE = 0.8
MAX_EMPTY_BURSTS = 5

# Persona: one value per axis is sampled per conversation (6^4 = 1296 combinations).
PERSONA_AXES = {
    "role": {
        "expert_researcher": "an expert researcher who demands rigor and systematic investigation",
        "curious_student": "a curious student eager to learn and understand step-by-step",
        "skeptical_reviewer": "a skeptical reviewer who questions assumptions and demands evidence",
        "domain_specialist": "a domain specialist seeking nuanced, in-depth analysis",
        "methodology_focused": "a methodology-focused user who cares about research design",
        "interdisciplinary": "an interdisciplinary thinker who values cross-field connections",
    },
    "style": {
        "formal_professional": "uses formal, academic language.",
        "casual_friendly": "speaks casually and conversationally.",
        "socratic_questioning": "asks probing Socratic questions.",
        "urgent_time_pressed": "is time-pressed and wants quick answers.",
        "collaborative": "collaborates as a thinking partner.",
        "directive": "gives clear, decisive direction.",
    },
    "trait": {
        "verbose_explainer": "Prefers thorough explanations.",
        "concise_minimalist": "Wants brief, dense responses.",
        "detail_oriented": "Focuses on specific facts and edge cases.",
        "big_picture_thinker": "Cares about patterns and strategic implications.",
        "systematic_methodical": "Values structured, complete approaches.",
        "exploratory_creative": "Enjoys unconventional angles and tangents.",
    },
    "approach": {
        "depth_first": "Wants to fully understand one area before moving on.",
        "breadth_first": "Prefers surveying the landscape quickly first.",
        "evidence_driven": "Demands claims backed by sources.",
        "hypothesis_driven": "Likes forming and testing hypotheses.",
        "comparative": "Wants to compare and contrast sources.",
        "narrative": "Values connecting findings into a coherent story.",
    },
}

OPENING_TEMPLATES = [
    "{query}",
    "{query}\n\nCan you look this up?",
    "Hey — {query}",
    "{query}\n\nWalk me through what you find.",
    "{query}\n\nI'd like a rigorous look — sources, key claims, where the evidence is weak.",
    "{query}\n\nLook for sources with solid methodology — note study design.",
    "{query}\n\nGo deep on this — I have background, so skip the intro material.",
    "{query}\n\nFind evidence on both sides if it's contested.",
]

# Flavor of the simulated user's next question, sampled per follow-up.
FOLLOW_UP_TYPES = {
    "react_to_finding": "A claim caught your attention; ask about a new aspect of it (mechanism, edge case).",
    "drill_deeper": "Ask about a sub-aspect that has not been researched yet.",
    "redirect": "Pivot to a related but distinct angle.",
    "challenge": "Ask for counter-evidence or opposing positions.",
    "clarify": "Ask about the underlying mechanism, methodology, or primary source.",
}

FINAL_ANSWER_TEMPLATES = [
    "Now write your comprehensive final synthesis on: {query}\n\nDo not call any tools. Answer the question, connect "
    "findings across sources, and note contradictions or uncertainty.",
    "Based on your research, give a definitive answer to: {query}\n\nNo more tools. Cover the direct answer, the "
    "supporting evidence, important caveats, and how confident you are.",
    "Write a narrative synthesis answering: {query}\n\nYour research is done. Weave the evidence from multiple "
    "sources into a story and end with clear conclusions.",
    "Provide a balanced synthesis for: {query}\n\nNo further research. What do the sources say, where do they agree "
    "or disagree, and what conclusion does the evidence support?",
    "Give me the bottom line on: {query}\n\nDone researching. What do we know for sure, what is still debated, and "
    "what should someone take away?",
]

PUSHBACKS = [  # sent when the agent stops calling tools too early; removed from the saved conversation
    "Let's keep digging. Search for an angle on this you haven't covered yet.",
    "Read one of the documents you found in full and pull out specific evidence.",
    "Investigate the counter-evidence: search for sources that disagree.",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Semantic search over a document corpus. Returns relevant passages with document IDs. "
            "Use natural language queries (full questions work better than keywords).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieve the full text of a document by its ID. Use doc_id values from search results.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "integer", "description": "The document ID to retrieve"}},
                "required": ["doc_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a research assistant investigating a topic with search and document retrieval.

- Think before acting: what do you know, and what do you still need?
- Use natural language search queries (questions, not keywords).
- After every 1-2 searches, read at least one full document with `get_document`.
- Don't repeat searches or re-read documents.
- Every tool call must come with 2-3 paragraphs of analysis: engage with the latest result (quote specifics,
  name sources), say where evidence agrees or disagrees, and explain why the next call is the right move.

## USER CONTEXT
The user is {persona}"""

BURST_REPLY_PROMPT = """Research topic: "{query}"
The user asked: "{user_message}"

Results from your latest searches and document reads:
{observations}

Write a substantive multi-paragraph reply to the user. Quote specific facts, numbers and sources, say where sources
agree or disagree, note caveats, and say what still needs to be looked into. Synthesize; don't just list."""

USER_SIM_PROMPT = """You are {persona}
You are learning about "{query}" from a research assistant.

Last thing you said: {last_user}
What the assistant just told you: {last_reply}

Write ONE natural follow-up message (1-2 sentences). Flavor: {flavor}
- Reference something specific the assistant said, then push into territory it has NOT covered yet.
- Never ask for a summary or wrap-up, and never write search queries, doc IDs, or instructions about tools.
- Sound like a real person in your persona's tone. Output only the message."""

THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[dict]) -> int:
    text = ""
    for m in messages:
        text += (m.get("content") or "") + (m.get("think") or "")
        text += "".join(tc["function"]["name"] + tc["function"]["arguments"] for tc in m.get("tool_calls") or [])
    return len(TOKENIZER.encode(text, disallowed_special=()))


class Generator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.llm = OpenAI(base_url=args.llm_base_url, api_key=args.api_key, timeout=1800)
        self.http = requests.Session()

    def chat(self, messages: list[dict], temperature: float, tools: list | None = None) -> tuple[str, str, list]:
        kwargs = {"tools": tools} if tools else {}
        if self.args.extra_body:
            kwargs["extra_body"] = self.args.extra_body
        msg = (
            self.llm.chat.completions.create(
                model=self.args.model,
                messages=[{k: v for k, v in m.items() if k != "think"} for m in messages],
                temperature=temperature,
                max_tokens=8000,
                **kwargs,
            )
            .choices[0]
            .message
        )
        content = msg.content or ""
        think = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        if match := THINK_RE.match(content):
            content, think = content[match.end() :], match.group(1)
        return content.strip(), think.strip(), msg.tool_calls or []

    def run_tool(self, name: str, args: dict, seen: dict) -> str | None:
        """Execute a tool call, or return None if it repeats an earlier one."""
        if name == "search":
            query = str(args.get("query", "")).strip()
            words = set(query.lower().split())
            if not words or any(len(words & p) / len(words | p) > QUERY_SIMILARITY for p in seen["queries"]):
                return None
            seen["queries"].append(words)
            resp = self.http.post(f"{self.args.search_url}/search", json={"query": query, "k": SEARCH_K}, timeout=60)
            results = resp.json().get("results", []) if resp.ok else []
            if not results:
                return "[Search failed] No results found. Try different terms."
            return "\n\n---\n\n".join(
                f"[Passage {i}] (doc_id: {r['doc_id']}, relevance: {r.get('rerank_score') or 0:.2f})\n"
                f"{r['passage_text']}"
                for i, r in enumerate(results, 1)
            )
        try:
            doc_id = int(args.get("doc_id"))
        except (TypeError, ValueError):
            return None
        if doc_id in seen["docs"]:
            return None
        seen["docs"].add(doc_id)
        resp = self.http.post(f"{self.args.search_url}/get_document", json={"doc_id": doc_id}, timeout=60)
        if not resp.ok:
            return f"[Failed] Could not retrieve document {doc_id}."
        doc = resp.json()
        words = doc["doc_text"].split()
        text = " ".join(words[:MAX_DOC_WORDS]) + ("..." if len(words) > MAX_DOC_WORDS else "")
        return f"[Full Document {doc_id}]\nSource: {doc.get('doc_file', '')}\n\n{text}"

    def context(self, conversation: list[dict]) -> list[dict]:
        """System prompt + first question + as many recent messages as fit in max_context_tokens."""
        pinned, recent = conversation[:2], conversation[2:]
        budget = self.args.max_context_tokens - count_tokens(pinned)
        while recent and (count_tokens(recent) > budget or recent[0]["role"] == "tool"):
            recent = recent[1:]
        return pinned + recent

    def generate(self, query: str, target_tokens: int, rng: random.Random) -> dict:
        persona = {axis: rng.choice(list(values)) for axis, values in PERSONA_AXES.items()}
        r, s, t, a = (PERSONA_AXES[axis][persona[axis]] for axis in PERSONA_AXES)
        persona_text = f"{r} who {s} {t} {a}"

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT.format(persona=persona_text)},
            {"role": "user", "content": rng.choice(OPENING_TEMPLATES).format(query=query)},
        ]
        seen: dict = {"queries": [], "docs": set()}
        turns = empty_bursts = 0

        def budget_left() -> bool:
            return count_tokens(conversation) < STOP_FRACTION * target_tokens and turns < MAX_TURNS

        while budget_left() and empty_bursts < MAX_EMPTY_BURSTS:
            observations = []
            for _ in range(rng.choice(BURST_SIZES)):
                if not budget_left():
                    break
                content, think, tool_calls = self.chat(self.context(conversation), AGENT_TEMPERATURE, TOOLS)
                turns += 1
                if not tool_calls:
                    break
                call = tool_calls[0]
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    break
                result = self.run_tool(call.function.name, args, seen)
                if result is None:
                    break
                call_id = call.id or f"call_{uuid.uuid4().hex[:16]}"
                conversation += [
                    {
                        "role": "assistant",
                        "content": content,
                        "think": think,
                        "tool_calls": [{"id": call_id, "type": "function", "function": call.function.model_dump()}],
                    },
                    {"role": "tool", "tool_call_id": call_id, "content": result},
                ]
                observations.append(f"{call.function.name}({call.function.arguments}):\n{result}")

            if not observations:
                empty_bursts += 1
                conversation.append({"role": "user", "content": rng.choice(PUSHBACKS)})
                continue
            empty_bursts = 0

            last_user = next(m["content"] for m in reversed(conversation) if m["role"] == "user")
            reply, _, _ = self.chat(
                [
                    {
                        "role": "user",
                        "content": BURST_REPLY_PROMPT.format(
                            query=query, user_message=last_user, observations="\n\n".join(observations)
                        ),
                    }
                ],
                AGENT_TEMPERATURE,
            )
            conversation.append({"role": "assistant", "content": reply})
            if budget_left():
                follow_up, _, _ = self.chat(
                    [
                        {
                            "role": "user",
                            "content": USER_SIM_PROMPT.format(
                                persona=persona_text,
                                query=query,
                                last_user=last_user[:300],
                                last_reply=reply[:600],
                                flavor=FOLLOW_UP_TYPES[rng.choice(list(FOLLOW_UP_TYPES))],
                            ),
                        }
                    ],
                    USER_TEMPERATURE,
                )
                conversation.append({"role": "user", "content": follow_up.strip("\"'") or "Go on."})

        conversation = [m for m in conversation if m["content"] not in PUSHBACKS]
        if conversation[-1]["role"] != "user":
            conversation.append({"role": "user", "content": rng.choice(FINAL_ANSWER_TEMPLATES).format(query=query)})
        answer, think, _ = self.chat(self.context(conversation), AGENT_TEMPERATURE)
        conversation.append({"role": "assistant", "content": answer, "think": think})

        return {
            "uuid": str(uuid.uuid4()),
            "conversation": to_release_format(conversation),
            "metadata": {
                "query": query,
                "target_tokens": target_tokens,
                "trace_tokens": count_tokens(conversation),
                "num_turns": turns,
                "num_searches": len(seen["queries"]),
                "num_doc_reads": len(seen["docs"]),
                "persona": persona,
                "generator_model": self.args.model,
            },
        }


def to_release_format(conversation: list[dict]) -> list[dict]:
    """Tool schemas on the system turn; tool calls as {name, arguments: dict}."""
    out = []
    for m in conversation:
        m = {k: v for k, v in m.items() if v not in (None, "")} | {"content": m["content"]}
        if m["role"] == "system":
            m["tools"] = [t["function"] for t in TOOLS]
        if "tool_calls" in m:
            m["tool_calls"] = [
                {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
                for tc in m["tool_calls"]
            ]
        out.append(m)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search_url", required=True, help="search360 server, e.g. http://localhost:8000")
    p.add_argument("--llm_base_url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8001/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api_key", default="EMPTY")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="A single seed question")
    source.add_argument("--questions_file", help="JSONL rows of {question, target_tokens}")
    p.add_argument("--target_tokens", type=int, default=32_000, help="Used when a row has no target_tokens")
    p.add_argument("--max_context_tokens", type=int, default=100_000, help="Keep below the model's context length")
    p.add_argument("--output", default="trajectories.jsonl")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra_body", type=json.loads, help="JSON merged into every chat request")
    args = p.parse_args()
    args.search_url = args.search_url.rstrip("/")

    if args.question:
        jobs = [(args.question, args.target_tokens)]
    else:
        with open(args.questions_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        jobs = [(r["question"], r.get("target_tokens", args.target_tokens)) for r in rows]

    generator = Generator(args)
    lock = threading.Lock()

    def run(i: int, question: str, target: int) -> None:
        try:
            record = generator.generate(question, target, random.Random(args.seed + i))
        except Exception as e:
            print(f"[error] {question[:60]}: {e!r}")
            return
        with lock, open(args.output, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta = record["metadata"]
        print(f"{meta['trace_tokens']:,}/{target:,} tokens, {meta['num_searches']} searches | {question[:60]}")

    with ThreadPoolExecutor(args.workers) as pool:
        for i, (question, target) in enumerate(jobs):
            pool.submit(run, i, question, target)


if __name__ == "__main__":
    main()
