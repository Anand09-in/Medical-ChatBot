import chainlit as cl
import httpx
import logging

API_URL = "http://localhost:8000/query"

@cl.on_message
async def main(message: cl.Message):

    try:
        logging.info(f"Received message: {message.content}")

        # Show loading message
        msg = cl.Message(content="🤖 Thinking...")
        await msg.send()

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                API_URL,
                json={"question": message.content}
            )

        data = response.json()
        logging.info(f"API response: {data}")

        answer = data.get("answer", "No answer found.")
        sources = data.get("sources", [])

        if sources:
            answer += "\n\n📚 Sources:\n"
            for s in sources:
                answer += f"- {s}\n"

        # Update the same message with the answer
        msg.content = answer
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()