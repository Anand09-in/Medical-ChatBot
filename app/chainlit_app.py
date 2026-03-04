import chainlit as cl
import requests
import logging
API_URL = "http://localhost:8000/query"

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Welcome to Medical RAG Bot!\nAsk your medical question."
    ).send()


@cl.on_message
async def main(message: cl.Message):

    try:
        logging.info(f"Received message: {message.content}")
        response = requests.post(
            API_URL,
            json={"question": message.content},
            timeout=120
        )

        data = response.json()
        logging.info(f"API response: {data}")
        answer = data.get("answer", "No answer found.")
        sources = data.get("sources", [])

        if sources:
            answer += "\n\n📚 Sources:\n"
            for s in sources:
                answer += f"- {s}\n"

        await cl.Message(content=answer).send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()