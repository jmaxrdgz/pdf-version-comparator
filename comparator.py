import torch

def cosine_similarity(embedding1, embedding2):
    embedding1 = torch.tensor(embedding1) if not isinstance(embedding1, torch.Tensor) else embedding1
    embedding2 = torch.tensor(embedding2) if not isinstance(embedding2, torch.Tensor) else embedding2

    dot_product = torch.dot(embedding1, embedding2)

    norm1 = torch.norm(embedding1)
    norm2 = torch.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("One of the embeddings is a zero vector.")

    return dot_product / (norm1 * norm2)