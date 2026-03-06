from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def get_redis_memory(session_id: str):
    """
    Creates Redis-backed conversational memory.

    Each session_id gets its own chat history.
    """

    chat_history = RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379"
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True,
        output_key="answer",
        k=6  # remember last 6 messages
    )
    logging.info(f"Built memory for {session_id}: {memory}")
    return memory