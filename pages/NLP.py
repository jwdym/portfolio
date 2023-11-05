import os
import nltk
import openai

import streamlit as st

from utils import Chat, reflections
from streamlit_js_eval import streamlit_js_eval

# Introduction
st.title('Natural Language Processing (NLP)')
st.markdown('The science of NLP has existed for decades, in recent years computational resource advances have enabled advanced technologies such as Chat GPT.')
st.markdown("There are many ways to leverage NLP in software, we'll explore a few NLP techniques in this page including LLMs.")

st.subheader('NLP Projects')
st.markdown(
    """
    I don't have any publicly available NLP projects in my portfolio. I have worked with LLMs on AWS, summarizing customer feedback and latent topics, as well as various customer transformers.

    This page is meant to be a simple interactive overview of some lightweight NLP tasks I can host on a cheap virtual machine.
    """
)
st.divider()

st.subheader('Chat Bot')
st.markdown('The first computer chatbot [ELIZA](https://en.wikipedia.org/wiki/ELIZA) was made in 1964-1967, ELIZA held therapy like conversations with users using simple prompts and responses.')
st.markdown('Below you can choose a simple chat bot, even ELIZA, available from [NLTK](https://www.nltk.org/api/nltk.chat.html) to chat with.')

bot = st.selectbox(
    label='Select a Chat Bot',
    options=['Eliza', 'Iesha', 'Rude', 'Suntsu', 'Zen'],
    placeholder='Please select a chatbot',
    index=None
)

if bot in ['Eliza', 'Iesha', 'Rude', 'Suntsu', 'Zen']:
    # Configure chat bot
    if bot == 'Eliza':
        pairs = nltk.chat.eliza.pairs
        bot_response = bot_response = f"""
            Now chatting with {bot} chatbot. \n
            Talk to the program by typing in plain English, using normal upper and lower-case letters and punctuation.  Enter "quit" when done. \n
            Hello.  How are you feeling today?
        """
    elif bot == 'Iesha':
        pairs = nltk.chat.iesha.pairs
        bot_response = f"""
            Now chatting with {bot} chatbot. \n
            Talk to the program by typing in plain English, using normal upper and lower-case letters and punctuation.  Enter "quit" when done. \n
            hi!! i'm iesha! who r u??!
        """
    elif bot == 'Rude':
        pairs = nltk.chat.rude.pairs
        bot_response = f"""
            Now chatting with {bot} chatbot. \n
            Talk to the program by typing in plain English, using normal upper and lower-case letters and punctuation.  Enter "quit" when done. \n
            I suppose I should say hello.
        """
    elif bot == 'Suntsu':
        pairs = nltk.chat.suntsu.pairs
        bot_response = f"""
            Now chatting with {bot} chatbot. \n
            Talk to the program by typing in plain English, using normal upper and lower-case letters and punctuation.  Enter "quit" when done. \n
            You seek enlightenment?
        """
    elif bot == 'Zen':
        pairs = nltk.chat.zen.responses
        bot_response = f"""
            Now chatting with {bot} chatbot. \n
            Talk your way to truth with Zen Chatbot. Type 'quit' when you have had enough. \n
            Welcome, my child.
        """
    chat_bot = Chat(pairs, reflections)

    # Setup chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Record first message
    with st.chat_message(bot):
        st.markdown(bot_response)

    # Accept user input
    if prompt := st.chat_input("Response"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message(bot):
            message_placeholder = st.empty()
            full_response = chat_bot.converse(prompt)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": bot, "content": full_response})
        if prompt == 'quit':
            st.session_state.messages = []
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

if bot not in ['Eliza', 'Iesha', 'Rude', 'Suntsu', 'Zen']:
    st.subheader('LLMs')
    st.markdown('Large Language Models (LLMs) can handle a wide variety of tasks from converstational AI, text summaraization, text generation, answering questions and more!')
    st.markdown("One downside to using LLMs is that they are quite large, [LLAMA2](https://ai.meta.com/llama/)s large model has 70 billion parameters and won't fit on most computers. In order to use many LLMs we have to interact with a model hosted on a large remote server through an API.")
    st.markdown("The chat interface below is pulled from the Streamlit [ChatGPT](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app) like article.")

    api_key = st.text_input(
        label='Enter OpenAI key here',
        type = 'password',
        placeholder = ''
    )
    openai.api_key = api_key

    if api_key != '':

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.chat_message('OpenAI'):
            st.markdown('Starting chat session with OpenAI, enter "quit" when done.')

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            if prompt == 'quit':
                st.session_state.messages = []
                streamlit_js_eval(js_expressions="parent.window.location.reload()")