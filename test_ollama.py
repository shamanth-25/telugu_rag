import ollama

response = ollama.chat(
    model="qwen2.5:0.5b",
    messages=[
        {
            "role": "user",
            "content": """
తమిళనాడు దక్షిణ భారతదేశంలో ఒక రాష్ట్రం.
దీని రాజధాని చెన్నై.

తమిళనాడు రాజధాని ఏది?
"""
        }
    ]
)

print(response["message"]["content"])
