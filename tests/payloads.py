"""Exemplos publicados em https://docs.typesafe.ai/api/system-one. Não são medições."""

CHARGED_TWICE = (
    "I was charged twice for my subscription this month. Please refund the extra charge."
)

NOUL_RESPONSE = {
    "model": "jev-latest",
    "answers": {"billing": {"type": "noul", "noul": 0.98}},
    "usage": {"input_tokens": 120, "output_tokens": 12},
}

CHOICE_RESPONSE = {
    "model": "jev-latest",
    "answers": {
        "tone": {
            "type": "choice",
            "choice": "calm",
            "confidence": 0.7,
            "probabilities": {"angry": 0.15, "calm": 0.8, "excited": 0.05},
        }
    },
    "usage": {"input_tokens": 120, "output_tokens": 12},
}

SCORE_RESPONSE = {
    "model": "jev-latest",
    "answers": {
        "urgency": {
            "type": "score",
            "score": 1.7,
            "confidence": 0.55,
            "legend": {
                "0": "Can wait",
                "1": "Needs attention this week",
                "2": "Needs attention today",
            },
            "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
        }
    },
    "usage": {"input_tokens": 120, "output_tokens": 12},
}

BOX_RESPONSE = {
    "model": "jev-preview",
    "answers": {
        "red_rectangle": {
            "type": "box",
            "box": {"top": 0.2, "bottom": 0.66, "left": 0.19, "right": 0.68},
            "confidence": None,
        }
    },
    "usage": {"input_tokens": 285, "output_tokens": 20},
}
