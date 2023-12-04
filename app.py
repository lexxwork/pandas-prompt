import streamlit as st
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse

import logging
import logger


def main():
    log = logger.setup_logger("app_logger", "app.log", logging.WARNING)
    st.title("Prompt-driven data analysis with PandasAI")

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = OpenAI(api_token=openai_api_key)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        agent = Agent(
            [df],
            memory_size=10,
            config={"llm": llm, "max_retries": 2, "response_parser": StreamlitResponse},
        )
        st.write(df.head(3))
        prompt = st.text_area("Enter your prompt:")

        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating response..."):
                    response = agent.chat(prompt)
                    log.info(f"Generated response for prompt: {prompt}")
                    if "temp_chart.png" in response:
                        st.image(response)
                    else:
                        st.write(response)
            else:
                st.warning("Please enter a prompt.")
        if st.button("New Conversation"):
            agent.start_new_conversation()


if __name__ == "__main__":
    main()
