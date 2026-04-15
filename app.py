import os
import gradio as gr
from openai import OpenAI

SYSTEM_PROMPT = """
You are StudyMate, a friendly AI tutor.

- Explain step-by-step.
- Be clear and simple.
- Help students understand concepts.
- Be encouraging and structured.
- Be fast.
"""

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

def chat_fn(message, history):
    if history is None:
        history = []

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

    history = history + [(message, reply)]
    return history, ""

with gr.Blocks(theme=gr.themes.Soft(), title="StudyMate AI") as demo:
    gr.Markdown("# StudyMate\nGet help with homework, maths, science, and study questions")

    chatbot = gr.Chatbot(
        value=[(None, "Hi 👋 I’m StudyMate. What do you need help with today?")],
        height=500,
    )

    msg = gr.Textbox(placeholder="Type a message...", label="")
    send = gr.Button("Send")
    clear = gr.Button("Clear")

    send.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])

    clear.click(
        lambda: ([(None, "Chat cleared. What would you like help with now?")], ""),
        outputs=[chatbot, msg],
    )

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
