import json

import httpx
import numpy as np
from openai import OpenAI

from rag.providers import OpenAIEmbeddings, OpenAIModel


def test_openai_embedding_batching_and_response_order():
    requests = []

    def handler(request):
        data = json.loads(request.content)
        requests.append(data)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(i + 1), 1.0], "object": "embedding"}
                    for i in reversed(range(len(data["input"])))
                ],
                "model": "test",
                "object": "list",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = OpenAI(
        api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    embeddings = OpenAIEmbeddings(client=client)
    vectors = embeddings.embed(["text"] * 65)
    assert [len(r["input"]) for r in requests] == [64, 1]
    assert vectors.shape == (65, 2)
    np.testing.assert_array_equal(vectors[:2, 0], [1, 2])


def test_openai_tool_request_and_final_response_shape():
    requests = []

    def handler(request):
        data = json.loads(request.content)
        requests.append(data)
        message = {"role": "assistant", "content": "Grounded [S1]"}
        if "tools" in data:
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "arguments": '{"query":"Atlas owner"}',
                    },
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 0,
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    client = OpenAI(
        api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    model = OpenAIModel(client=client)
    turn = model.complete([{"role": "user", "content": "Atlas?"}], tools=True)
    assert turn.calls[0].name == "search_documents"
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["tools"][0]["function"]["strict"] is True
    assert (
        model.complete([{"role": "user", "content": "Atlas?"}], tools=False).text == "Grounded [S1]"
    )
    assert "tools" not in requests[1]
