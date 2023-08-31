from typing import Optional

import torch


class GRUDecoder(torch.nn.Module):
    """
    Class implementing a single layer GRU decoder.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # Generate a 1-layer GRU:
        self.rnn = torch.nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.tensor, state: Optional[torch.tensor] = None):
        """
        Run input sequence through the decoder.

        Args:
            x: [nb,seq_len,input_size]
        Returns:
            hidden: [nb,seq_len,hidden_size]
        """
        if state is not None:
            hidden, next_state = self.rnn(x, state)
        else:
            hidden, next_state = self.rnn(x)
        return (hidden, next_state)
