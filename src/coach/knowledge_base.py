"""
Strategic knowledge base for rule-based coaching.
Also serves as context when calling an LLM.
"""

PIECE_DESCRIPTIONS = {
    "ADEPT": "Infantry unit. Moves to adjacent squares on the same layer only. "
             "Cheap but essential for holding territory.",
    "ELEMENTAL": "Diagonal striker. Slides diagonally across the board and between layers. "
                 "Excellent for cross-layer attacks and flanking.",
    "GUARDIAN": "Orthogonal fortress. Slides along rows, columns, and vertically between layers. "
                "Best for blocking enemy advances and controlling corridors.",
}

STAGE_DESCRIPTIONS = {
    "CALCINATION": "Stage 1 - raw potential, the piece has just begun its journey.",
    "DISSOLUTION": "Stage 2 - breaking down old patterns.",
    "SEPARATION": "Stage 3 - distinguishing the essential from the inessential.",
    "CONJUNCTION": "Stage 4 - recombining purified elements.",
    "FERMENTATION": "Stage 5 - introducing new life into the process.",
    "DISTILLATION": "Stage 6 - refining and purifying further.",
    "COAGULATION": "Stage 7 - solidifying the gains.",
    "ILLUMINATION": "Stage 8 - approaching the final transformation.",
    "PHILOSOPHERS_STONE": "Stage 9 - the ultimate form. Powerful but a high-value target.",
}

STRATEGIC_PRINCIPLES = [
    "Layer 3 is the center of the universe. Every strategic plan should orbit around it.",
    "The three key nodes at (3,1,1), (3,1,2), and (3,1,3) are the primary win condition.",
    "Adepts anchor your position on a layer. Don't underestimate their defensive value.",
    "Elementals are your assassins - use diagonals to strike where the opponent is weak.",
    "Guardians control corridors. A well-placed Guardian can block entire attack routes.",
    "Each move advances a piece's transformation stage. Aggressive play has a hidden upside.",
    "A piece at PHILOSOPHERS_STONE is both your most powerful asset and your biggest liability.",
    "When ahead in material, trade pieces to simplify. When behind, create complications.",
    "Mobility is life. A piece with no legal moves is a dead piece.",
    "Vertical control (owning the same row/col across layers) creates devastating attack lanes.",
    "Don't rush to the center with everything. Build a formation, then advance together.",
    "Captures near layer 3 are more valuable than captures on the flanks.",
]

OPENING_TIPS = [
    "In the opening, develop your Elementals first - they have the longest reach.",
    "Avoid moving the same piece twice in the opening unless there's a tactical reason.",
    "Try to establish a presence on layer 3 before your opponent does.",
    "Guardians are slow to develop but powerful once positioned. Be patient.",
]

ENDGAME_TIPS = [
    "In the endgame, piece stage matters enormously. Advanced pieces are harder to replace.",
    "If you control 2 of 3 key nodes, focus everything on securing the third.",
    "With few pieces left, Guardians become dominant - they control long corridors.",
]


def get_coaching_context() -> str:
    """Build a text block describing the game for LLM context."""
    lines = [
        "# Strata - 3D Layered Strategy Board Game",
        "",
        "## Board",
        "7 stacked layers, each 3x6. Total 126 positions.",
        "Coordinates: (layer, row, column). Layer 3 is the center.",
        "",
        "## Pieces (27 per player)",
        "- ADEPT (9): moves to adjacent squares on same layer only",
        "- ELEMENTAL (9): slides diagonally, can cross layers",
        "- GUARDIAN (9): slides orthogonally, can move vertically between layers",
        "",
        "## Transformation",
        "Each move advances a piece through 9 stages:",
        "CALCINATION -> DISSOLUTION -> SEPARATION -> CONJUNCTION -> FERMENTATION ->",
        "DISTILLATION -> COAGULATION -> ILLUMINATION -> PHILOSOPHERS_STONE",
        "",
        "## Win Conditions",
        "1. Control three key nodes on layer 3: (3,1,1), (3,1,2), (3,1,3)",
        "2. Eliminate all opponent pieces at PHILOSOPHERS_STONE stage",
        "3. Immobilize the opponent (no legal moves)",
    ]
    return "\n".join(lines)
