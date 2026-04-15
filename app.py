import os
import gradio as gr
from openai import OpenAI

# System prompt (makes it a tutor)
SYSTEM_PROMPT = """
You are StudyMate, a friendly AI tutor for students.

- For maths, show step-by-step working.
- For science calculations, show every step clearly.
- Explain concepts in simple language.
- Be encouraging and clear.
- Help students understand, not just get the answer.
- Keep answers structured and easy to follow.
"""

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

def chat_fn(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=messages,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: {e}"

    history.append((message, reply))
    return history


with gr.Blocks(theme=gr.themes.Soft(), title="StudyMate AI") as demo:
    gr.Markdown("# 📚 StudyMate AI\nYour personal AI tutor")

    with gr.Row():
        gr.Button("Math").click(lambda: "Help me solve this math problem step-by-step:", None, None)
        gr.Button("Science").click(lambda: "Explain this science question step-by-step:", None, None)
        gr.Button("Homework").click(lambda: "Help me understand this homework:", None, None)

    chatbot = gr.Chatbot(
        value=[(None, "Hi 👋 I'm StudyMate. What do you need help with?")],
        height=500,
    )

    msg = gr.Textbox(placeholder="Ask a question...")
    send = gr.Button("Send")

    send.click(chat_fn, [msg, chatbot], chatbot)
    msg.submit(chat_fn, [msg, chatbot], chatbot)

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
