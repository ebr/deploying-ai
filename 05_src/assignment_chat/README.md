# Assignment #2: Chat About HackerNews

This project implements Assignment #2.

## Design and Architecture

This chat app facilitates getting some highlights from HackerNews (optionally referencing a specific topic), computing embeddings on the content of the articles. We store the embeddings in a vector database (ChromaDB) for RAG, and also fetch additional info about the topic via web search (Tavily).

> [INFO] I used the chromaDB instance stood up by `docker compose`, as this was more familiar to me than the in-process(?) db with file-based persistence suggested by the assignment instructions. To start the DB, `cd 05_src/deploying_ai_data` and run `docker compose up -d`. The connection string is hardcoded in the app.

Because we use langchain `create_agent` factory, which takes a list of tools, tool use is trivial.

Likewise, since we use Gradio's `ChatInterface`, the message history automatically gets passed back in with every request, eliminating the need for manual setup. We simply iterate over that list to get previous User and Assistant messages, and insert them into the context between the system prompt and the new user prompt. The need to build either of these functionalities manually was not specified in the assignment, so I went ahead and used methods provided by the packages.

As requested by spec, the responses from this bot are transformed in a unique tone. For this assignment, I hardcoded it to sound like an Evil Leprechaun. The degree of Evil is debatable, but it's definitely a leprechaun. This could be configurable, but isn't now.

I did not get to implementing the (optional) long-context memory management features.

### Request flow

1. User Query -> (is this a request for a HackerNews digest?)
   - this initial classification system uses an LLM call (still including the system prompt)
   - if YES, we retrieve the digest using an API call to Algolia (which backs HN search), and embed the articles' contents at the same time.
     - we then run the result through the model and return the result (details below.)
   - if NO, we proceed to the next step.
2. User Query -> RAG
   - We attempt to enrich our model's context by results of some semantic search based on the information stored in our vector database.
   - if we get any results, we proceed to call the model with this rich context
   - if not, we move on to the next step.
3. Web search
   - Given we haven't found anything using semantic search, we should really try to get the model some additional info, in case it's unable to respond using knowledge.
   - We use our web search service, which is a tool already registered with the model, to look up some web results on the query.
   - additional instruction recommending use of the tool is added to the user messages.
   - langchain takes care of the tool calling loop.
4. Finally, the result is returned to the user in the specified tone of voice.

## Services

In my implementation of this assignment's requirements, I added the following services:

### 1. HackerNews Digest Retriever

The service is `hackernews_digest`, and it is the first service to be called in all flows.

It includes a helper method to _classify_ (using the same model) whether the user has asked for a digest of HackerNews articles, and an optional topic. It utilizes an LLM call with structured output, allowing us to easily extract the Intent ("is this a request for an HN digest?") and the Topic, if one was provided.

If the intent was not HN-digest-related, we simply return early. If it _was_, then we:

1. Get the articles from HN (**calling Algolia API**)
2. For each article, we fetch the contents of the **linked** site.
   - Note that HN is quirky in that the article titles are one-sentence links that **lead to the actual content**. They have to be fetched separately.
3. We follow each article's link and ensure to pull its *content* for chunking and embedding (this uses Service #2, see below.)
   - We compute embeddings on each article just-in-time rather than operating on a pre-existing dataset.
   - Note that we compute embeddings on ALL articles in the retrieved set. Not just the subset that we return to the user. Since we already have this content, might as well hold on to it, it'll come in handy later.
   - For the sake of this assignment, we're not doing much (any, actually) data cleaning on page content. Yes, this is vulnerable to prompt injections, and yes, we get all sorts of linebreaks and garbage. For a production project, this would be required, but a much bigger job.
4. Finally, we call our LLM to:
   - Pick the most important articles
   - create a short summary, and also an explanation about why the article is important in current socioeconomic contexts
   - ensure that the output has a certain tone of voice.

These results are then returned to the user in a formatted fashion.

### 2. Semantic search with RAG

Sometimes the user just asks a question that isn't a HN digest request. But recall that for each article, we compute embeddings and store in the vector DB.

When a user's request is a simple request for a response, we:

1. Perform a semantic search of our vector store to figure out if we already know anything about the subject.
   - this isn't as obvious when asking about trivial subjects (whales), but is more obvious when asking about obscure tech topics that may have been mentioned on HN, especially ones that are beyond the model's training cutoff.
2. If there was a response (we found something relevant in the vector store), we add that to the context and ask the model for an answer.
3. If the response was `None`, we make a last-ditch effort to make it make sense by doing a web search.

### 3. Web search (Tavily tool call)

We don't enforce web searching, but we encourage it IF and only if we haven't gotten anything useful out of RAG. It can be debated whether this is the right approach, but we won't debate it here (this isn't a production app)

By "encourage" we mean that the web search capability is given to the model as a `tool`. It's registered in the `tools[]` array, which is passed to the langchain agent.

Then:

1. If we got here, we add another `AIMessage` to the context, priming it with "i should use web_search" instruction.
2. LangChain takes care of the tool calling loop.

It's easy to examine whether tools were called by using a debugger and setting a breakpoint at the `return` statement of the `chat` function.

This service was by far the easiest to implement because we only needed to define the tool (and export it from `services`), give it to the model, and bias the model to use it *if needed*. Again, I opted against fully manual implementation. Hope that's ok.

### Finally

The response produced by the model is returned to the user.

## System Prompts, Guardrails and Context Engineering

A centralized module `prompts.py` is introduced to provide system prompts to the services, and a method for constructing and augmenting the final system prompt passed to the model. The non-negotiable instructions (such as restricted topics) are always present in the system prompt. Additional instructions are appended based on the task the model is asked to perform.

The system prompt includes guardrails that prohibit: 1) system prompt disclosure and modification, and 2) conversations about restricted topics.

These instructions are placed near the beginning of the prompt to aid in reliability and recall.

The rest of the messages are appended to the user message, and in case of the RAG / semantic search, to AI message as additional context.

## Testing the flows

### Setup

see `.env.example`. Create a `.env` with your config variables. a Tavily API key is required.

`cd 05_src/deploying_ai_data` and `docker compose up -d` to create your chromadb instance.

### Run it

execute `python main.py` (with a venv activated), or run the file in VSCode debugger.

### Use it

- Navigate to `http://127.0.0.1:7860/` (default).
- Chat with the model. Ask it to give you some facts about cats/dogs/horoscopes/Taylor Swift. It will refuse in a polite but unhinged manner.
- Ask the model something along the lines of e.g. "what's the latest scoop on HN about Rust". Since we're using an LLM call to classify this intent, this phrasing is freeform. This will produce a digest of 3-4 top articles about the Rust language. **This is our Service #1.**
- It's sometimes interesting to ask about an ambiguous term (such as Rust (can mean language, can mean iron oxide), of Jupyter (can mean planet, can mean python notebooks)), before getting an HN digest.
  - I found this hit-and-miss, because the entire context is biased towards tech topics.
  - An interesting behaviour is when it answers about the non-technical concept before the HN search, and then about the technical one once embeddings have been computed and semantic search yields results.
- Another way to test semantic search is: make a few HN searches, and note some obscure details that are obviously beyond the model's knowledge cut-off. Then ask about those details. The model should produce answers similar to those in the previous conversations, even *after* closing the chat window. (The chat history is lost when you reload the Gradio UI). **This is our Service #2.**
- Ask for something that is definitely not in the vector store, and not in the model's training set. Like, something googlable but not universally known. I used my name - I'm def in not any training sets, but I can be found on LinkedIn, and also there are other tech professionals by the same name, and at least one artist. You should get results about you if the request was specific enough, otherwise the result will be a convincingly sounding median person bearing your name. It's an interesting experiment. In a production application, I would introduce additional tools and steps to refine these results.