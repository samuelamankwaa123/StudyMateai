import os
import requests
from openai import OpenAI
import gradio as gr
from datetime import datetime

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

SYSTEM_PROMPT = """
You are a friendly study partner for a student.

- For maths: show step-by-step working
- Always show step-by-step solutions for science calculations and explain clearly
- Explain concepts clearly and simply
- If the student is wrong: correct gently and explain why
- Keep answers concise, clear, structured, and avoid unnecessary long explanations
- If current or recent search information is provided, use it directly
- If no current information is provided for a changing fact, clearly say the information may be outdated
- When search results include sources, base the answer on them
- You can always provide links where there is no answer at all
"""

FREE_TURN_LIMIT = 15
GLOBAL_REQUEST_LIMIT = 150
total_requests = 0


def normalize_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(content, dict):
        return str(content.get("text", content))

    if content is None:
        return ""

    return str(content)


def live_search(query):
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return None

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 3,
    }

    try:
        res = requests.post(url, json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()

        results = data.get("results", [])
        snippets = []

        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            link = r.get("url", "")
            snippets.append(f"{title}: {content}\nSource: {link}")

        return "\n\n".join(snippets) if snippets else None

    except Exception:
        return None


def chat(message, history):
    global total_requests

    history = history or []

    if len(history) >= FREE_TURN_LIMIT:
        return (
            f"🚫 Free limit reached.\n\n"
            f"You’ve used your {FREE_TURN_LIMIT} free questions.\n\n"
            "👉 Want more access?\n"
            "Pay $4.99 here: https://paypal.me/SamuelAmankwaa\n\n"
            "After payment:\n"
            "📩 Send proof via Email:\n"
            "samankwaa49@yahoo.com\n\n"
            "You will receive premium access."
        )

    if total_requests >= GLOBAL_REQUEST_LIMIT:
        return (
            "🚫 Free usage is currently full.\n\n"
            "👉 Skip the wait and get premium access:\n"
            "https://paypal.me/SamuelAmankwaa\n\n"
            "📩 Send proof to: samankwaa49@yahoo.com"
        )

    current_date = datetime.now().strftime("%A, %d %B %Y")

    messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\nCurrent real date: {current_date}"
    }]
            
    try:
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                user, assistant = turn
                if user:
                    messages.append({"role": "user", "content": str(user)})
                if assistant:
                    messages.append({
                        "role": "assistant",
                        "content": normalize_content(assistant)
                    })
            elif isinstance(turn, dict):
                role = turn.get("role")
                content = turn.get("content")
                if role in {"user", "assistant", "system"} and content:
                    messages.append({
                        "role": role,
                        "content": normalize_content(content)
                    })

        user_message = str(message)

        # Use live search only for current/recent questions
        if any(word in user_message.lower() for word in ["what is", "today", "latest", "current", "news", "now", "as of", "news", "president", "vice president", "recent", "future"]):
            search_results = live_search(user_message)
            if search_results:
                user_message += f"\n\nUse this real-time information:\n{search_results}"

        messages.append({"role": "user", "content": user_message})

        total_requests += 1

        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=messages,
            temperature=0.7,
        )

        answer = normalize_content(response.choices[0].message.content)
        return answer or "I could not generate a response."

    except Exception as e:
        return f"Sorry, I hit an error: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="StudyMate",
    description="Get help with homework, maths, science, and study questions",
    examples=[
        "What is photosynthesis?",
        "Solve 12 × 4 and show working",
        "I think gravity pushes objects upward. Am I right?",
    ],
)

demo.launch(server_name="0.0.0.0", server_port=7860)