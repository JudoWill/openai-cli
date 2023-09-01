import io
import os

import click

from openai_cli.client import build_client

@click.group()
def cli():
    pass


@cli.command()
@click.argument("source", type=click.File("rt", encoding="utf-8"), nargs=-1)
@click.option("-t", "--token", default="", help="OpenAI API token")
@click.option(
    "-m", "--model", default="text-davinci-003", 
    help="OpenAI model option. (i.e. code-davinci-002)"
)
@click.option(
    "-c", "--chat", default=False, is_flag=True,
    help="Model uses new-style chat API (i.e. gpt-3.5-turbo)"
)
@click.option(
    "-s", "--string", default=None, type=str,
    help="String input to model. Placed AFTER the file input.",
)
def complete(source: io.TextIOWrapper, token: str, model: str, chat: bool, string: str) -> None:
    """Return OpenAI completion for a prompt from SOURCE."""
    
    client = build_client(get_token(token),
                          get_api_url(chat),
                          chat)
        
    prompt = [s.read() for s in source]
    if string:
        prompt.append(string)
        
    result = client.generate_response('\n'.join(prompt), model)
    click.echo(result)


@cli.command()
@click.option("-t", "--token", default="", help="OpenAI API token")
@click.option(
    "-m", "--model", default="text-davinci-003", 
    help="OpenAI model option. (i.e. code-davinci-002)"
)
@click.option(
    "-c", "--chat", default=False, is_flag=True,
    help="Model uses new-style Chat API (i.e. gpt-3.5-turbo)"
)
def repl(token: str, model: str, chat: bool) -> None:
    """Start interactive shell session for OpenAI completion API."""

    client = build_client(get_token(token), 
                          get_api_url(chat),
                          chat)

    while True:
        print(client.generate_response(input("Prompt: "), model))
        print()


def get_api_url(chat: bool) -> str:
    if chat:
        return os.environ.get("OPENAI_API_URL",
                              "https://api.openai.com/v1/chat/completions")
    
    return os.environ.get("OPENAI_API_URL", 
                          "https://api.openai.com/v1/completions")
    

def get_token(token: str) -> str:
    if not token:
        token = os.environ.get("OPENAI_API_TOKEN", "")
    if not token:
        raise click.exceptions.UsageError(
            message=(
                "Either --token option or OPENAI_API_TOKEN environment variable must be provided"
            )
        )
    return token


if __name__ == "__main__":
    cli()
