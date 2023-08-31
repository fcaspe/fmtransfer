import pytest
import torch

from fmtransfer.models.decoders import GRUDecoder


@pytest.fixture
def gru_decoder():
    return GRUDecoder(input_size=2, hidden_size=512)


def test_decoder_can_forward(gru_decoder):
    output = gru_decoder(torch.rand(16, 1000, 2))
    assert output[0].shape == (16, 1000, 512)
    assert output[0].requires_grad
