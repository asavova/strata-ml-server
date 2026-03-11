"""
Coach engine — provides natural-language teaching commentary.

Two modes:
1. Rule-based: uses knowledge_base patterns (always available)
2. LLM-powered: calls an external LLM API for richer commentary (optional)
"""

import logging
import os
import random
from typing import Optional

from . import knowledge_base as kb

logger = logging.getLogger("strata.coach")


class CoachEngine:
    """
    Generates coaching commentary for game positions.

    If an LLM API key is configured, uses the LLM for rich responses.
    Otherwise, falls back to rule-based coaching from the knowledge base.
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_api_key_env: str = "STRATA_LLM_API_KEY",
        llm_base_url: str = "https://api.openai.com/v1",
    ):
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = os.environ.get(llm_api_key_env) if llm_model else None
        self.llm_available = self.llm_model is not None and self.llm_api_key is not None

        if self.llm_available:
            logger.info(f"LLM coaching enabled: model={llm_model}")
        else:
            logger.info("Using rule-based coaching (no LLM configured)")

    def analyze(
        self,
        active_player: str,
        ply_count: int,
        white_count: int,
        black_count: int,
        pieces: list[dict],
        last_move: Optional[dict] = None,
    ) -> dict:
        """Analyze the position after a move."""
        if self.llm_available:
            return self._llm_analyze(
                active_player, ply_count, white_count, black_count, pieces, last_move
            )
        return self._rule_analyze(
            active_player, ply_count, white_count, black_count, pieces, last_move
        )

    def suggest(
        self,
        active_player: str,
        ply_count: int,
        white_count: int,
        black_count: int,
        pieces: list[dict],
    ) -> dict:
        """Suggest a strategic direction."""
        center_presence = [
            p for p in pieces
            if p.get("layer") == 3 and p.get("owner") == active_player
        ]

        if not center_presence:
            suggestion = "Advance toward layer 3. Elementals are fastest for cross-layer movement."
        elif len(center_presence) < 3:
            suggestion = "You have a foothold on layer 3. Reinforce with Guardian support."
        else:
            suggestion = "Strong center presence. Consider attacking opponent's advanced pieces."

        return {
            "commentary": f"Position at ply {ply_count}. Material: W={white_count}, B={black_count}.",
            "suggestion": suggestion,
            "teachingTip": random.choice(kb.STRATEGIC_PRINCIPLES),
            "confidence": 0.70,
        }

    def teach(
        self,
        active_player: str,
        ply_count: int,
        white_count: int,
        black_count: int,
        pieces: list[dict],
        question: Optional[str] = None,
    ) -> dict:
        """Respond to a teaching request or question."""
        if self.llm_available and question:
            return self._llm_teach(
                active_player, ply_count, white_count, black_count, pieces, question
            )
        return self._rule_teach(
            active_player, ply_count, white_count, black_count, pieces, question
        )

    # ── Rule-based methods ────────────────────────────────

    def _rule_analyze(
        self, active_player, ply_count, white_count, black_count, pieces, last_move
    ) -> dict:
        parts = [f"Turn {ply_count}. Active: {active_player}."]
        parts.append(f"Material — White: {white_count}, Black: {black_count}.")

        if last_move:
            if last_move.get("isCapture"):
                parts.append(
                    f"{last_move.get('mover')}'s {last_move.get('movedPiece')} captured "
                    f"{last_move.get('capturedPiece')}. "
                    f"The piece has reached {last_move.get('newStage', 'next')} stage."
                )
                parts.append("Captures accelerate transformation — a hidden advantage.")
            else:
                parts.append("A quiet positional move. Building for the future.")

        center_white = sum(1 for p in pieces if p.get("layer") == 3 and p.get("owner") == "WHITE")
        center_black = sum(1 for p in pieces if p.get("layer") == 3 and p.get("owner") == "BLACK")
        if center_white > 0 or center_black > 0:
            parts.append(f"Center control — White: {center_white}, Black: {center_black}.")

        tip = random.choice(kb.STRATEGIC_PRINCIPLES)
        if ply_count < 10:
            tip = random.choice(kb.OPENING_TIPS)

        return {
            "commentary": " ".join(parts),
            "suggestion": "",
            "teachingTip": tip,
            "confidence": round(random.uniform(0.65, 0.90), 2),
        }

    def _rule_teach(
        self, active_player, ply_count, white_count, black_count, pieces, question
    ) -> dict:
        q = (question or "").lower()

        if "center" in q or "layer 3" in q:
            commentary = (
                "Layer 3 is the strategic heart. The three key nodes at "
                "(3,1,1), (3,1,2), and (3,1,3) form the primary win condition. "
                "All strategy orbits around contesting these positions."
            )
        elif "transform" in q or "stage" in q:
            commentary = (
                "Each piece advances through 9 transformation stages — one per move. "
                "Reaching PHILOSOPHERS_STONE is powerful but makes the piece a high-value target. "
                "Balance advancement with protection."
            )
        elif "adept" in q:
            commentary = kb.PIECE_DESCRIPTIONS["ADEPT"]
        elif "elemental" in q:
            commentary = kb.PIECE_DESCRIPTIONS["ELEMENTAL"]
        elif "guardian" in q:
            commentary = kb.PIECE_DESCRIPTIONS["GUARDIAN"]
        else:
            if white_count > black_count + 3:
                commentary = "White has a material lead. Simplify by trading pieces."
            elif black_count > white_count + 3:
                commentary = "Black leads in material. White should create tactical chaos."
            else:
                commentary = "Material is balanced. Focus on positioning and center control."

        return {
            "commentary": commentary,
            "suggestion": "",
            "teachingTip": random.choice(kb.STRATEGIC_PRINCIPLES),
            "confidence": 0.80,
        }

    # ── LLM-powered methods ───────────────────────────────

    def _llm_analyze(
        self, active_player, ply_count, white_count, black_count, pieces, last_move
    ) -> dict:
        prompt = self._build_llm_prompt(
            "analyze",
            active_player, ply_count, white_count, black_count, pieces,
            last_move=last_move,
        )
        response_text = self._call_llm(prompt)
        return self._parse_llm_response(response_text)

    def _llm_teach(
        self, active_player, ply_count, white_count, black_count, pieces, question
    ) -> dict:
        prompt = self._build_llm_prompt(
            "teach",
            active_player, ply_count, white_count, black_count, pieces,
            question=question,
        )
        response_text = self._call_llm(prompt)
        return self._parse_llm_response(response_text)

    def _build_llm_prompt(
        self, mode, active_player, ply_count, white_count, black_count,
        pieces, last_move=None, question=None,
    ) -> str:
        context = kb.get_coaching_context()

        position = (
            f"Current position: ply {ply_count}, active player: {active_player}. "
            f"White has {white_count} pieces, Black has {black_count}."
        )

        center_pieces = [p for p in pieces if p.get("layer") == 3]
        if center_pieces:
            position += f" {len(center_pieces)} pieces are on the center layer."

        advanced = [p for p in pieces if p.get("stage") in ("ILLUMINATION", "PHILOSOPHERS_STONE")]
        if advanced:
            position += f" {len(advanced)} pieces are at advanced transformation stages."

        if mode == "analyze" and last_move:
            move_desc = (
                f"Last move: {last_move.get('mover')}'s {last_move.get('movedPiece')} "
                f"moved. {'Captured ' + str(last_move.get('capturedPiece')) + '.' if last_move.get('isCapture') else 'No capture.'}"
            )
        else:
            move_desc = ""

        if mode == "analyze":
            instruction = (
                "You are a Strata game coach. Analyze the position after this move. "
                "Explain what happened, why it matters strategically, and give a teaching tip. "
                "Be concise but insightful. Respond in plain text, 2-4 sentences."
            )
        else:
            instruction = (
                f"You are a Strata game coach. The player asks: '{question}'. "
                "Answer based on the current position. Be educational and encouraging. "
                "Respond in plain text, 2-4 sentences."
            )

        return f"{context}\n\n{position}\n{move_desc}\n\n{instruction}"

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API. Returns raw text response."""
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.llm_api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a strategic board game coach."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.7,
            }

            response = httpx.post(
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=15.0,
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.warning(f"LLM API returned {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _parse_llm_response(self, text: str) -> dict:
        """Convert raw LLM text into the structured coach response."""
        if not text:
            # Fallback to rule-based
            return {
                "commentary": "The coach is thinking...",
                "suggestion": "",
                "teachingTip": random.choice(kb.STRATEGIC_PRINCIPLES),
                "confidence": 0.5,
            }

        return {
            "commentary": text.strip(),
            "suggestion": "",
            "teachingTip": "",
            "confidence": 0.85,
        }
