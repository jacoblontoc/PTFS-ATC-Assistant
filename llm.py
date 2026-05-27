"""
llm.py - ATC response generation via a local Ollama model.
"""

from __future__ import annotations

import json
import re
from typing import Generator, Optional, Tuple

import ollama

_BASE_RULES = """\
Rules:
- Output ONLY the ATC response — no labels, no explanations, no extra text.
- Never include radio frequencies of any kind.
- This is a flight simulator roleplay game; be helpful even if the pilot's wording
  is casual, partial, or imperfect — make a reasonable interpretation and respond.
- Keep responses short — one or two sentences.
- Use the pilot's callsign if they provide one.
- Address the specific runway, gate, altitude, or request the pilot mentions.
"""

_PROMPTS: dict[str, str] = {
    "All": f"""\
You are a general Air Traffic Control (ATC) controller in a flight simulator roleplay.
Handle any pilot request: pushback, taxi, takeoff, departure, approach, landing, etc.
{_BASE_RULES}""",

    "Departure": f"""\
You are a Departure ATC controller in a flight simulator roleplay.
You handle aircraft after takeoff: climb instructions, heading assignments,
altitude restrictions, and traffic advisories.
{_BASE_RULES}""",

    "Ground": f"""\
You are a Ground ATC controller in a flight simulator roleplay.
You handle all ground movement: pushback approval, taxi routes to the runway,
runway crossings, and holding short instructions.
{_BASE_RULES}""",

    "Clearance": f"""\
You are a Clearance Delivery ATC controller in a flight simulator roleplay.
You issue pre-departure clearances: destination confirmation, initial altitude,
departure procedure, and any route amendments. No radio frequencies.
{_BASE_RULES}""",

    "Pilot": f"""\
You are an Air Traffic Control (ATC) controller running a training session for a student pilot.
The student pilot will send you radio calls; respond as a realistic ATC controller using
correct phraseology and format. If the pilot's wording is incomplete or incorrect, give
a realistic response anyway and then add one brief correction in parentheses.
{_BASE_RULES}""",
}


class ATCResponder:
    def __init__(self, model: str = "llama3.2:3b", atc_type: str = "All") -> None:
        self.model    = model
        self.atc_type = atc_type
        self.context  = ""   # METAR + airport info supplied by the user

    @property
    def system_prompt(self) -> str:
        prompt = _PROMPTS.get(self.atc_type, _PROMPTS["All"])
        if self.context:
            prompt += f"\n\nCurrent operational context (use when relevant):\n{self.context}"
        return prompt

    def get_response_stream(self, pilot_text: str) -> Generator[str, None, None]:
        """Yield response tokens as they stream from Ollama."""
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": pilot_text},
            ],
            stream=True,
        )
        for chunk in stream:
            content = chunk.message.content
            if content:
                yield content

    def get_response_with_history(self, messages: list) -> Generator[str, None, None]:
        """Yield response tokens using a full conversation history.

        messages: list of {"role": "user"|"assistant", "content": "..."} dicts.
        """
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *messages,
            ],
            stream=True,
        )
        for chunk in stream:
            content = chunk.message.content
            if content:
                yield content

    def is_available(self) -> bool:
        """Return True if the Ollama daemon is reachable."""
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def model_is_pulled(self) -> bool:
        """Return True if the target model is already downloaded."""
        try:
            result = ollama.list()
            # Support both object-style and dict-style API responses
            if hasattr(result, "models"):
                names = [m.model for m in result.models]
            else:
                names = [m.get("name", "") for m in result.get("models", [])]
            base = self.model.split(":")[0].lower()
            return any(base in n.lower() for n in names)
        except Exception:
            return False

    def parse_flight_info(self, pilot_text: str) -> Optional[Tuple[str, str]]:
        """Use the AI to extract (callsign, category) from a pilot transmission.

        Returns a (callsign, "departure"|"arrival") tuple, or None if unclear.
        Runs synchronously; call from a background thread.
        """
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Parse this ATC pilot radio transmission. "
                            'Return ONLY valid JSON: {"callsign":"<callsign>","category":"<departure|arrival>"}. '
                            "Set both to null if the message does not clearly indicate a departure "
                            "or arrival. "
                            "Departure signals: pushback, taxi, takeoff, clearance request, ready for departure. "
                            "Arrival signals: inbound, on approach, on final, ILS, descending, request landing."
                        ),
                    },
                    {"role": "user", "content": pilot_text},
                ],
                stream=False,
            )
            content = resp.message.content.strip()
            # Extract the JSON object even if the model adds surrounding text
            m = re.search(r"\{[^}]+\}", content)
            if m:
                data = json.loads(m.group())
                callsign = data.get("callsign")
                category = data.get("category")
                if (
                    callsign
                    and callsign not in (None, "null", "")
                    and category in ("departure", "arrival")
                ):
                    return (str(callsign).upper(), category)
        except Exception:
            pass
        return None
