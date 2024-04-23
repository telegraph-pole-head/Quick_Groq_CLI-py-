import os
import rich_click as click
from groq import Groq
from rich.console import Console
from rich.markdown import Markdown

# Initialize the Groq client with your API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
console = Console()

MODEL_ALIASES = {
    "l3-8": "llama3-8b-8192",
    "l3-70": "llama3-70b-8192",
    "ge": "gemma-7b-it",
    "l2-70": "llama2-70b-4096",
    "mi": "mixtral-8x7b-32768"
}

DEFAULT_MODEL = "mi"

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
    model_alias = click.prompt("Please choose a model alias", default=DEFAULT_MODEL, show_choices=False)
    return MODEL_ALIASES.get(model_alias, MODEL_ALIASES[DEFAULT_MODEL])


@click.command()
@click.option('--message', prompt='You', help='Your message to the chatbot.')
@click.option('--model', type=click.Choice(get_models(), case_sensitive=False), default=DEFAULT_MODEL, show_default=True, help='Choose a model for the chatbot (optional)')
def chat(message, model):
    """Simple CLI chatbot using Groq API with multiple models."""
    try:
        model_name = MODEL_ALIASES[model]
        
        # Send user message to Groq API and get response using the selected model
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            model=model_name,
        )
        # Print the chatbot's response
        # Use Rich to print the chatbot's response
        response = chat_completion.choices[0].message.content
        console.print(f"[bold magenta]{model_name}:[/bold magenta]")
        console.print(Markdown(response))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")

if __name__ == '__main__':
    chat()