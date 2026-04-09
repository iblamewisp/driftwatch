import litserve as ls
from sentence_transformers import SentenceTransformer


class EmbeddingAPI(ls.LitAPI):

    def setup(self, device: str) -> None:
        # Loaded once per server process — shared across all batches
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def decode_request(self, request: dict) -> str:
        return request["text"]

    def predict(self, texts: list[str]) -> list[list[float]]:
        # texts is a list when max_batch_size > 1 — one forward pass for the whole batch
        return self.model.encode(texts).tolist()

    def encode_response(self, output: list[list[float]]) -> dict:
        return {"embedding": output}


if __name__ == "__main__":
    server = ls.LitServer(
        EmbeddingAPI(),
        accelerator="auto",       # GPU if available, CPU otherwise
        max_batch_size=64,
        batch_timeout=0.2,        # flush after 200ms regardless of batch size
        workers_per_device=1,
    )
    server.run(port=8001, num_api_servers=1)
