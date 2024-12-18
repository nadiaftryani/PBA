import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiGRU(nn.Module):
    """
    Bidirectional GRU for text classification.

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    rnn_size : int
        Size of Bi-GRU

    rnn_layers : int
        Number of layers in Bi-GRU

    dropout : float
        Dropout
    """
    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        embeddings: torch.Tensor,
        emb_size: int,
        fine_tune: bool,
        rnn_size: int,
        rnn_layers: int,
        dropout: float
    ) -> None:
        super(BiGRU, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)

        # bidirectional GRU
        self.BiGRU = nn.GRU(
            emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=(0 if rnn_layers == 1 else dropout),
            batch_first=True
        )

        # fully connected layer
        self.fc = nn.Linear(2 * rnn_size, n_classes)

        self.dropout = nn.Dropout(dropout)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool = True) -> None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data

        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        # word embedding, apply dropout
        embeddings = self.dropout(self.embeddings(text))  # (batch_size, word_pad_len, emb_size)

        # pack sequences (remove word-pads, SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            embeddings,
            lengths=words_per_sentence.tolist(),
            batch_first=True,
            enforce_sorted=False
        )  # a PackedSequence object

        # run through bidirectional GRU
        rnn_out, _ = self.BiGRU(packed_words)  # (n_words, 2 * rnn_size)

        # unpack sequences (re-pad with 0s, WORDS -> SENTENCES)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)  # (batch_size, word_pad_len, 2 * rnn_size)

        # average pooling over time (word_pad_len axis)
        H = torch.mean(rnn_out, dim=1)  # (batch_size, 2 * rnn_size)

        # fully connected layer
        scores = self.fc(self.dropout(H))  # (batch_size, n_classes)

        return scores
