import json

import httpx

from agentrt.rag_tool import RagIn, make_rag_tool


async def test_rag_tool_calls_ask_endpoint():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answer": "Twice a year [1]",
                "sources": [
                    {
                        "doc_id": "d",
                        "filename": "solar.pdf",
                        "page": 3,
                        "chunk": 2,
                        "score": 0.6,
                        "text": "x" * 2000,
                    }
                ],
            },
        )

    client = httpx.AsyncClient(base_url="http://rag.test", transport=httpx.MockTransport(handler))
    tool = make_rag_tool("http://rag.test", client)
    out = await tool.fn(RagIn(question="How often to clean panels?"))
    assert seen == {"path": "/ask", "body": {"question": "How often to clean panels?", "top_k": 3}}
    assert out.answer == "Twice a year [1]"
    assert (out.sources[0].filename, out.sources[0].page) == ("solar.pdf", 3)
    assert len(out.sources[0].text) == 500
