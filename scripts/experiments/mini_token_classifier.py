import torch
import torch.nn as nn


class MiniTokenClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_labels: int,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.classifier = nn.Linear(
            in_features=embedding_dim,
            out_features=num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        embeddings = self.embedding(input_ids)

        logits = self.classifier(embeddings)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
        }


if __name__ == "__main__":

    model = MiniTokenClassifier(
        vocab_size=100,
        embedding_dim=8,
        num_labels=3,
    )

    input_ids = torch.tensor([
        [12, 45, 93],
        [4, 18, 72],
    ])

    labels = torch.tensor([
        [0, 0, 2],
        [1, 0, 2],
    ])

    outputs = model(
        input_ids=input_ids,
        labels=labels,
    )

    print("Logits :", outputs["logits"].shape)
    print("Loss :", outputs["loss"])