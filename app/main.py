import asyncio
import os
import queue
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from pydantic import BaseModel

load_dotenv()

POSTGRES_URL = os.environ["POSTGRES_URL"]
OPENAI_KEY = os.environ["OPENAI_KEY"]
COLLECTION_NAME = "sec_10k_chunks"

PROMPT_TEMPLATE = """\
You are a financial analyst assistant. Answer the question using only the provided excerpts \
from the company's 10-K annual filing. If the answer is not in the excerpts, say so.

Excerpts:
{context}

Question: {question}
"""

store: PGVector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    # Use the sync psycopg driver — avoids ProactorEventLoop issues on Windows.
    # DB calls run in a thread pool via asyncio.to_thread.
    store = PGVector(
        connection=POSTGRES_URL,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_KEY),
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('section', '')} | {doc.metadata.get('doc_id', '')}]\n{doc.page_content}"
        for doc in docs
    )


def make_rag_chain(ticker: str, k: int = 5, section: str | None = None):
    filter_ = {"stock_symbol": ticker}
    if section:
        filter_["section"] = section

    retriever = store.as_retriever(search_kwargs={"k": k, "filter": filter_})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY, streaming=True)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


class QueryRequest(BaseModel):
    question: str
    ticker: str
    section: str | None = None


_SENTINEL = object()


@app.post("/query")
async def query(req: QueryRequest):
    chain = make_rag_chain(req.ticker, section=req.section or None)

    # Run the sync chain.stream() in a thread and forward chunks via a queue.
    # This gives token-by-token streaming without requiring async psycopg.
    chunk_queue: queue.Queue = queue.Queue()

    def stream_sync():
        try:
            for chunk in chain.stream(req.question):
                chunk_queue.put(chunk)
        except Exception as e:
            chunk_queue.put(e)
        finally:
            chunk_queue.put(_SENTINEL)

    async def generate():
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, stream_sync)
        while True:
            item = await asyncio.to_thread(chunk_queue.get)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    return StreamingResponse(generate(), media_type="text/plain")
