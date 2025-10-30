"""Public API Quick Start Example.

Demonstrates how to use the stable public entry points without importing
internal modules at top-level.
"""

from insightspike.public import create_agent


def main():
    agent = create_agent()  # defaults to mock provider; lightweight
    res = agent.process_question("What is geDIG in one sentence?")
    # Works for both dict-like and object-like result
    if hasattr(res, 'response'):
        print(res.response)
    else:
        print(res.get("response", "No response"))


if __name__ == "__main__":
    main()

