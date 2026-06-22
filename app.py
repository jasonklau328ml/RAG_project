import chainlit as cl

@cl.on_chat_start
async def start_session():
    # Set up a blank history for this specific user
    cl.user_session.set("chat_history", [])
    
    # Send a welcoming message
    await cl.Message(content="Welcome! Ask me anything, and I'll remember our conversation.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Retrieve this specific user's history
    history = cl.user_session.get("chat_history")
    
    # Save the new user message to history
    history.append({"role": "user", "content": message.content})
    
    # Simulate an AI response that looks back at the history
    ai_reply = f"You've sent {len(history)} message(s) so far. Your latest was: '{message.content}'"
    
    # Save AI reply to history
    history.append({"role": "assistant", "content": ai_reply})
    cl.user_session.set("chat_history", history)
    
    await cl.Message(content=ai_reply).send()