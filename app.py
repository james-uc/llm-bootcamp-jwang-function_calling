import json
from dotenv import load_dotenv
import chainlit as cl
from movie_functions import get_now_playing_movies, get_showtimes, buy_ticket
import litellm
from langsmith import traceable

load_dotenv(override=True)

litellm.success_callback = ["langsmith"]

# Choose one of these model configurations by uncommenting it:

# OpenAI GPT-4
# model = "openai/gpt-4o"

# Anthropic Claude
model = "claude-3-5-sonnet-20241022"

# Fireworks Qwen
# model = "fireworks_ai/accounts/fireworks/models/qwen2p5-coder-32b-instruct"

gen_kwargs = {"temperature": 0.2, "max_tokens": 500}

SYSTEM_PROMPT = """\
You are an AI movie assistant designed to provide information about currently \
playing movies and engage in general movie-related discussions. Your primary \
function is to answer questions about movies currently in theaters and offer \
helpful information to users interested in cinema.

You have access to the following functions:

<available_functions>
{
  "get_now_playing": {
    "description": "Fetches a list of movies currently playing in theaters",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
   "get_showtimes": {
      "description": "Fetches showtimes for a specific movie in a given location",
      "parameters": {
         "type": "object",
         "properties": {
         "title": {
            "type": "string",
            "description": "The title of the movie to search for"
         },
         "location": {
            "type": "string",
            "description": "The location to search for showtimes"
         }
         },
         "required": ["title", "location"]
      }
   },
   "buy_ticket": {
      "description": "Initiates the ticket purchase process for a specific movie at a theater",
      "parameters": {
         "type": "object",
         "properties": {
         "theater": {
            "type": "string",
            "description": "The name of the theater to purchase tickets from"
         },
         "movie": {
            "type": "string",
            "description": "The title of the movie to purchase tickets for"
         },
         "showtime": {
            "type": "string",
            "description": "The showtime for the movie"
         }
         },
         "required": ["theater", "movie", "showtime"]
      }
   },
   "confirm_ticket_purchase": {
      "description": "Actually executes the ticket purchase for a specific movie at a theater",
      "parameters": {
         "type": "object",
         "properties": {
         "theater": {
            "type": "string",
            "description": "The name of the theater where tickets were purchased"
         },
         "movie": {
            "type": "string",
            "description": "The title of the movie tickets were purchased for"
         },
         "showtime": {
            "type": "string",
            "description": "The showtime for the movie tickets were purchased for"
         }
         },
         "required": ["theater", "movie", "showtime"]
      }
}
</available_functions>

To use any function, generate a function call in JSON format, wrapped in \
<function_call> tags. For example:
<function_call>
{
  "name": "get_now_playing",
  "arguments": {}
}
</function_call>

If you determine that you need a function call, output ONLY the thought process and function call, \
then stop. Do not provide any additional information in the response. Once there is additional context added \
to the conversation, you can continue with a full response to the user's request.

When answering questions, follow these guidelines:

1. Always begin with a <thought_process> section to think through your response \
strategy. Consider:
   a. Determine if the question is about currently playing movies or general \
cinema topics
   b. Identify key elements of the question (e.g., specific movie titles, \
genres, actors)
   c. Decide if any available functions are needed
   d. Assess your confidence level based on the following criteria:
      - High confidence: Questions about movies released before 2020, film \
history, classic directors, or basic cinema concepts
      - Medium confidence: Questions about movies from 2020-2022, general \
industry trends, or recent developments in cinema
      - Low confidence: Questions about movies released after 2022, \
        box office numbers, or current industry specifics

2. If the question is to fetch currently playing movies:
   - Call the get_now_playing function before responding

3. For general movie-related discussions:
   - Draw upon your knowledge of cinema, directors, actors, and film history
   - Be aware that your knowledge of older movies is likely to be more accurate \
than your knowledge of recent movies
   - Offer recommendations based on genres, actors, or directors mentioned in \
the conversation
   - Explain basic film terminology or concepts if asked

4. When answering:
   - Prioritize accuracy over speculation
   - If you're unsure about something, especially regarding recent movies, \
admit it and offer to provide related information you are confident about
   - Keep responses concise but informative
   - If a question is unclear, ask for clarification before answering

Example interactions:

1. User: "What movies are playing in theaters right now?"
<thought_process>
The user wants to know about current movie listings. I need to fetch this \
real-time information using the get_now_playing function.
</thought_process>

<function_call>
{
  "name": "get_now_playing",
  "arguments": {}
}
</function_call>

2. User: "Can you tell me about the French New Wave? I'm interested in learning \
about classic film movements."
<thought_process>
This is a question about film history and a major cinematic movement. I have \
high confidence in this topic as it's well-documented historical information \
that doesn't require current data.
</thought_process>

The French New Wave (Nouvelle Vague) was a groundbreaking film movement that \
emerged in France during the late 1950s and 1960s. Led by young \
critics-turned-filmmakers like François Truffaut and Jean-Luc Godard, it \
revolutionized cinema with innovative techniques and storytelling approaches. \
Some essential films from this period include "The 400 Blows" (1959), \
"Breathless" (1960), and "Cléo from 5 to 7" (1962). Would you like to know \
more about specific directors or techniques from this movement?

3. User: "Who directed The Godfather?"
<thought_process>
This is a straightforward question about a classic film from 1972. I have high \
confidence in this information as it's a well-established historical fact.
</thought_process>

The Godfather was directed by Francis Ford Coppola. Released in 1972, it's \
considered one of the greatest films ever made and won him the Academy Award \
for Best Picture and Best Adapted Screenplay, though he lost Best Director to \
Bob Fosse for Cabaret that year.
"""


def extract_tag_content(text: str, tag_name: str) -> str | None:
    """
    Extract content between XML-style tags.

    Args:
        text: The text containing the tags
        tag_name: Name of the tag to find

    Returns:
        String content between tags if found, None if not found

    Example:
        >>> text = "before <foo>content</foo> after"
        >>> extract_tag_content(text, "foo")
        'content'
    """
    import re

    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


@traceable
@cl.on_chat_start
def on_chat_start():
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)


@cl.on_message
@traceable
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()

    response = litellm.completion(
        model=model, messages=message_history, stream=True, **gen_kwargs
    )

    for part in response:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()
    message_history.append({"role": "assistant", "content": response_message.content})

    while function_call := extract_tag_content(
        response_message.content, "function_call"
    ):
        function_call = json.loads(function_call)
        function_name = function_call["name"]
        function_args = function_call.get("arguments", {})

        if function_name == "get_now_playing":
            movies = get_now_playing_movies()
            message_history.append(
                {
                    "role": "system",
                    "content": f"Formulate a new response to the last message from the user using this context: {movies}",
                }
            )

        elif function_name == "get_showtimes":
            title = function_args.get("title", "")
            location = function_args.get("location", "")
            showtimes = get_showtimes(title, location)
            message_history.append(
                {
                    "role": "system",
                    "content": f"Formulate a new response to the last message from the user using this context: {showtimes}",
                }
            )

        elif function_name == "buy_ticket":
            theater = function_args.get("theater", "")
            movie = function_args.get("movie", "")
            showtime = function_args.get("showtime", "")
            message_history.append(
                {
                    "role": "system",
                    "content": f"Ask the user to confirm the purchase of tickets for {movie} at {theater} for the showtime {showtime}.",
                }
            )

        elif function_name == "confirm_ticket_purchase":
            theater = function_args.get("theater", "")
            movie = function_args.get("movie", "")
            showtime = function_args.get("showtime", "")
            confirmation = buy_ticket(theater, movie, showtime)
            message_history.append(
                {
                    "role": "system",
                    "content": f"Inform the user that the tickets have been purchased with this information as context: {confirmation}",
                }
            )

        else:
            break

        response_message = cl.Message(content="")
        await response_message.send()

        response = litellm.completion(
            model=model, messages=message_history, stream=True, **gen_kwargs
        )

        for part in response:
            if token := part.choices[0].delta.content or "":
                await response_message.stream_token(token)

        await response_message.update()
        message_history.append(
            {"role": "assistant", "content": response_message.content}
        )

    cl.user_session.set("message_history", message_history)


if __name__ == "__main__":
    cl.main()
