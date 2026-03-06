import chainlit as cl
import httpx
import uuid


API_URL = "http://localhost:8000/query"


@cl.on_chat_start
async def start():

    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content="Hello! Ask me anything about the medical documents."
    ).send()


@cl.on_message
async def main(message: cl.Message):

    session_id = cl.user_session.get("session_id")

    msg = cl.Message(content="🤖 Thinking...")
    await msg.send()

    async with httpx.AsyncClient(timeout=120) as client:

        response = await client.post(
            API_URL,
            json={
                "question": message.content,
                "session_id": session_id
            }
        )

    data = response.json()

    answer = data.get("answer", "")
    sources = data.get("sources", [])

    if sources:
        answer += "\n\n📚 Sources:\n"
        for s in sources:
            answer += f"- {s}\n"

    msg.content = answer
    await msg.update()