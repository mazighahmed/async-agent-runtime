import httpx
from pydantic import BaseModel, Field

from agentrt.tools import Tool


class RagIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class RagSource(BaseModel):
    filename: str
    page: int
    text: str


class RagOut(BaseModel):
    answer: str
    sources: list[RagSource]


def make_rag_tool(base_url: str, client: httpx.AsyncClient | None = None) -> Tool:
    client = client or httpx.AsyncClient(base_url=base_url, timeout=30)

    async def search(args: RagIn) -> RagOut:
        response = await client.post("/ask", json=args.model_dump())
        response.raise_for_status()
        body = response.json()
        sources = [
            RagSource(filename=s["filename"], page=s["page"], text=s["text"][:500])
            for s in body["sources"]
        ]
        return RagOut(answer=body["answer"], sources=sources)

    return Tool(
        "search_documents",
        "Search the indexed PDF documents and get an answer with file and page sources.",
        RagIn,
        search,
    )
