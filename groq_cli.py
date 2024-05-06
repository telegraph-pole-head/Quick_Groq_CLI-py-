import os
import rich_click as click
from groq import Groq
from rich.console import Console
from rich.markdown import Markdown
from pathlib import Path
from rich.panel import Panel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory

# Initialize the Groq client with your API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

console = Console()

click.rich_click.OPTION_GROUPS = {
    "groq_cli.py": [
        {
            "name": "Basic options",
            "options": [
                "--message",
                "--model",
                "--prompt",
            ],
        },
        {
            "name": "Advanced options",
            "options": [
                "--stream-mode",
                "--temperature",
                "--max-tokens",
                "--top-p",
            ],
        },
    ]
}

MODEL_ALIASES = {
    "l3-8": "llama3-8b-8192",
    "l3-70": "llama3-70b-8192",
    "ge": "gemma-7b-it",
    "l2-70": "llama2-70b-4096",
    "mi": "mixtral-8x7b-32768",
}

DEFAULT_MODEL = "l3-70"


def get_models():
    # This function would ideally call Groq's API to list available models
    # For the sake of example, we'll return a static list. Replace with API call if available.
    # return ["llama3-8b-8192","llama3-70b-8192", "gemma-7b-it", "llama2-70b-4096", "mixtral-8x7b-32768"]
    return list(MODEL_ALIASES.keys())


# Function to display model choices and get the actual model identifier
def choose_model():
    console.print("Available models:", style="bold blue")
    for alias, model in MODEL_ALIASES.items():
        console.print(f"[bold magenta]{alias}[/bold magenta] [pink]({model})[/pink]")
    model_alias = click.prompt(
        "Please choose a model alias", default=DEFAULT_MODEL, show_choices=False
    )
    return MODEL_ALIASES.get(model_alias, MODEL_ALIASES[DEFAULT_MODEL])


# Define a dictionary mapping aliases to paths using pathlib
PATH_ALIASES = {
    "def": Path("./prompts/prompt_default.md"),
    "cli": Path("./prompts/cli_helper.md"),
}


def get_prompts():
    return list(PATH_ALIASES.keys())


# Function to read content from the file at the given path
def read_content(file_path):
    try:
        return file_path.read_text()
    except FileNotFoundError:
        console.print(f"[bold red]File not found: {file_path}[/bold red]")
        return ""


@click.command()
@click.option("--message", "-d", default=None, help="Your message to the chatbot.")
@click.option(
    "--model",
    "-m",
    type=click.Choice(get_models(), case_sensitive=False),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Choose a model for the chatbot",
)
@click.option(
    "--prompt",
    "-p",
    type=click.Choice(get_prompts(), case_sensitive=False),
    default="def",
    show_default=True,
    help="Alias for file path to read system prompts (optional).",
)
@click.option(
    "--stream-mode",
    "-s",
    type=bool,
    default=False,
    show_default=True,
    help="Stream mode (true or false).",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=1.0,
    help="Temperature for controlling randomness (optional).",
)
@click.option(
    "--max-tokens",
    "-x",
    default=1024,
    show_default=True,
    help="Maximum number of tokens.",
)
@click.option(
    "--top-p",
    "-o",
    type=float,
    default=1.0,
    show_default=True,
    help="Top p for nucleus sampling.",
)
def chat(message, model, prompt, stream_mode, temperature, max_tokens, top_p):
    """Simple CLI chatbot using Groq API with multiple models."""
    try:
        model_name = MODEL_ALIASES[model]
        file_dir = Path(__file__).resolve().parent
        system_prompt = read_content(file_dir / PATH_ALIASES[prompt]) or ""

        if message:  # input by -d
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]
            one_chat(messages, model_name, stream_mode, temperature, max_tokens, top_p)
        else:  # multi round conversation
            multi_chat(
                system_prompt, model_name, stream_mode, temperature, max_tokens, top_p
            )
            pass

    except Exception as e:
        console.print(
            Panel(
                f"{str(e)}",
                title="[red]Error[/red]",
                border_style="red",
                title_align="left",
                width=120,
            )
        )


def one_chat(messages, model_name, stream_mode, temperature, max_tokens, top_p):
    try:
        # Send user message to Groq API and get response using the selected model
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream_mode,
            stop=None,
        )

        print("")  # add a new line

        # Use Rich to print the chatbot's response
        # get the chatbot's response
        if stream_mode:
            console.print(f"[bold magenta]🤖 Chatbot: {model_name}:[/bold magenta]")
            all_responses = ""
            for chunk in chat_completion:
                response = chunk.choices[0].delta.content or ""
                console.print(response, end="", width=120)
                all_responses += response + " "
            return all_responses
        else:
            response = chat_completion.choices[0].message.content or ""
            console.print(
                Panel(
                    Markdown(response),
                    border_style="bold magenta",
                    title=f"🤖 Chatbot: {model_name}",
                    title_align="left",
                    width=120,
                    highlight=True,
                )
            )
            return response

    except Exception as e:
        console.print(
            Panel(
                f"{str(e)}",
                title="[red]Error[/red]",
                border_style="red",
                title_align="left",
                width=120,
            )
        )


def multi_chat(system_prompt, model_name, stream_mode, temperature, max_tokens, top_p):
    global client

    flag = True
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    llm = ChatGroq(
        # client=client,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        # top_p=top_p,
    )
    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=300, return_messages=True
    )

    while flag:
        console.print(f"\n[bold magenta]🤗 User: [/bold magenta]", end="\n")
        usr_prompts = ''
        while True:
            usr_prompt = input()
            usr_prompts += usr_prompt
            if not usr_prompts:
                console.print(
                    Panel(
                        Markdown(
                            "Prompt cannot be **empty**, please enter again or type `q` to *quit*"
                        ),
                        border_style="yellow",
                        title=f"Help",
                        title_align="left",
                        width=120,
                    )
                )
            elif usr_prompt == "q":
                flag = False
                break
            elif len(usr_prompts) > 0 and usr_prompts[-1] != '\t': # enter (line breaks for tab + enter)
                mem = memory.load_memory_variables({})
                mem_str = "Here is your memory (chat history): " + str(mem["history"])
                messages.append({"role": "system", "content": mem_str})
                messages.append({"role": "user", "content": usr_prompts})
                reponse = one_chat(
                    messages, model_name, stream_mode, temperature, max_tokens, top_p
                )
                memory.save_context(
                    {"user": usr_prompts},
                    {"assistant": reponse},
                )
                break
            elif len(usr_prompts) > 0 and usr_prompts[-1] == '\t': 
                usr_prompts = usr_prompts[:-1] # remove '\t'


if __name__ == "__main__":
    chat()
