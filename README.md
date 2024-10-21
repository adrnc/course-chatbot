# Contextualized Chatbot

The contextualized chatbot allows for returning answers to contextualized questions on a custom LLM.
With every question asked in the chat, the following happens inside our chatbot:

- The context for a user question is retrieved.
  In particular, this means that certain text passages that may be important are returned.
- In case this is not the first question, given the chat history,
  the question is reformulated in such a way that it can be understood without the chat history.
- The reformulated question and the context are passed to the LLM,
  which tries to formulate a meaningful anwer.

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

    The first time a Ollama is started with a model, that model needs to be downloaded,
    which, depending on the model size, can take some time.
2. Then, after the server has been initialized, start the chat bot by executing the python script as follows:

    ```bash
    python bot.py [model]
    ```

    Note that on some Linux version Python 3 is referred to by `python3`.

Replace `[model]` with your preferred Ollama model. All available models can also be found on the Ollama website [here](https://ollama.com/library).

### Example

For example, we can start with [`gemma2:9b`](https://ollama.com/library/gemma2) model which is Google's [Gemma](https://ai.google.dev/gemma) model with version 2 configured to take in 9 billion parameters. In what sizes the parameters are available depends on the respective model.

1. We first start the Ollama server via:

    ```bash
    ollama run "gemma2:9b"
    ```
2. In a similar fashion, we start the chat bot:

    ```bash
    python bot.py "gemma2:9b"
    ```

The first initialization process may take some time, because the custom content
needs to be contextualized. Once the bot is initialized, we should see a chat prompt,
where we can ask the bot questions regarding specific topics.

## Customizing the `data` folder

The top-level `data` folder contains the following files that can be customized:

- `content.md`: This is the markdown file from which the bot draws its custom data.
  Its content can be changed. The bot needs to be restarted in this case.
- `prompt.txt`: This text file contains the prompt that is passed to the bot.
  There are two placeholders in the prompt, namely `{context}` and `{question}`.
  These are replaced with their generated values. Newlines and whitespace in the prompt
  is kept as is. The answer by the chatbot should follow directly after.
