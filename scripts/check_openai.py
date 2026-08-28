"""Verify the OpenAI key, pick a model, and check whether temperature is accepted.

    python scripts/check_openai.py

Run once after adding the key. Cheap-tier model names change often and the newer
models reject `temperature`, so both are worth confirming rather than assuming.
"""

from openai import OpenAI

from claricyte.rag.providers import OPENAI_MODEL, api_key


def main() -> None:
    client = OpenAI(api_key=api_key())

    names = sorted(m.id for m in client.models.list())
    chat = [n for n in names if n.startswith(("gpt", "o1", "o3", "o4"))]
    print(f"{len(names)} models available. Chat-capable:")
    for name in chat:
        marker = "   <- providers.OPENAI_MODEL" if name == OPENAI_MODEL else ""
        print(f"   {name}{marker}")
    if OPENAI_MODEL not in names:
        print(f"\nWARNING: {OPENAI_MODEL} is not in this account's list.")
        return

    probe = [{"role": "user", "content": "Reply with the single word: ok"}]
    reply = client.chat.completions.create(
        model=OPENAI_MODEL, messages=probe, max_completion_tokens=16
    )
    print(f"\ncall without temperature: {reply.choices[0].message.content!r}")
    used = reply.usage
    print(f"tokens: {used.prompt_tokens} in, {used.completion_tokens} out")

    try:
        client.chat.completions.create(
            model=OPENAI_MODEL, messages=probe, max_completion_tokens=16, temperature=0
        )
        print("temperature=0 accepted -> set it in OpenAIProvider for the eval")
    except Exception as error:
        print(f"temperature rejected ({type(error).__name__}) -> leave it unset")


if __name__ == "__main__":
    main()
