from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str, **kwargs: str) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", value)
    return text
