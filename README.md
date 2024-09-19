# Contextualized Chatbot

## Installation

Before running the project, we need to install all dependencies. This involves two parts:

- This project requires Python 3 to run. Find the latest downloads [here](https://www.python.org/downloads/). Through the command line tool tool `pip` we install all dependencies for the Python project by using the command:
    
    ```bash
    pip install -r requirements.txt
    ```
- Download Ollama server for your respective operating system through the [Ollama website](https://ollama.com/).


## Execution

Running this chatbot requires us to first start the Ollama server. Afterwards we execute our script which connects to the server. We use the following commands:

1. Start the Ollama server by executing the following command on the command line:

    ```bash
    ollama run [model]
    ```
2. Then, after the server has been initialized, start the chat bot by executing the python script as follows:

    ```bash
    python bot.py [model]
    ```

Replace `[model]` with your preferred Ollama model. All available models can also be found on the Ollama website [here](https://ollama.com/library). For example, we can start with [`llama3.1:8b`](https://ollama.com/library/llama3.1) model which is Meta's [Llama](https://www.llama.com/) model with version 3.1 configured to take in 8 billion parameters. In what sizes the parameters are available depends on the respective model.
