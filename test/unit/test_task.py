import torch

from fmtransfer.tasks import FMTransfer


def test_fmtransfer_can_be_instantiated(mocker):
    decoder = mocker.stub("decoder")
    loss_fn = mocker.stub("loss_fn")

    model = FMTransfer(decoder=decoder, optimizer=torch.optim.Adam, loss_fn=loss_fn)
    assert model is not None
    assert model.decoder == decoder
    assert model.loss_fn == loss_fn


def test_fmtransfer_forwards_all_no_relay(mocker):
    class FakeModule(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, *args):
            return self.output

    batch_size = 7
    input_size = 5
    hidden_size = 512
    instance_len = 1024
    out_size = 12

    loss_fn = mocker.stub("loss_fn")

    # Fake Encoder
    expected_encoder_output = torch.rand(batch_size, instance_len, hidden_size)
    encoder = FakeModule(expected_encoder_output)
    encoder_spy = mocker.spy(encoder, "forward")

    expected_decoder_output = torch.rand(batch_size, instance_len, hidden_size)
    decoder = FakeModule(expected_decoder_output)
    decoder_spy = mocker.spy(decoder, "forward")

    expected_projection_output = torch.rand(batch_size, instance_len, out_size)
    projection = FakeModule(expected_projection_output)
    projection_spy = mocker.spy(projection, "forward")

    conditioning = torch.rand(batch_size, instance_len, input_size)

    model = FMTransfer(
        loss_fn=loss_fn,
        encoder=encoder,
        optimizer=torch.optim.Adam,
        decoder=decoder,
        projection=projection,
        relay_conditioning_to_projection=False,
    )

    y, next_state = model(conditioning)

    encoder_spy.assert_called_once_with(conditioning)
    decoder_spy.assert_called_once_with(expected_encoder_output)
    projection_spy.assert_called_once_with(expected_decoder_output)

    assert torch.all(y == expected_projection_output)
    assert next_state.shape == (batch_size, instance_len, hidden_size)


def test_fmtransfer_forwards_all_with_relay(mocker):
    class FakeModule(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, *args):
            return self.output

    batch_size = 7
    input_size = 5
    hidden_size = 512
    instance_len = 1024
    out_size = 12

    loss_fn = mocker.stub("loss_fn")

    # Fake Input
    conditioning = torch.rand(batch_size, instance_len, input_size)

    # Fake Encoder
    expected_encoder_output = torch.rand(batch_size, instance_len, hidden_size)
    encoder = FakeModule(expected_encoder_output)
    encoder_spy = mocker.spy(encoder, "forward")

    expected_decoder_output = (
        torch.rand(batch_size, instance_len, hidden_size),
        torch.rand(batch_size, hidden_size),
    )
    decoder = FakeModule(expected_decoder_output)
    decoder_spy = mocker.spy(decoder, "forward")

    expected_projection_output = torch.rand(batch_size, instance_len, out_size)
    projection = FakeModule(expected_projection_output)
    projection_spy = mocker.spy(projection, "forward")

    model = FMTransfer(
        loss_fn=loss_fn,
        encoder=encoder,
        optimizer=torch.optim.Adam,
        decoder=decoder,
        projection=projection,
        relay_conditioning_to_projection=True,
    )

    y, next_state = model(conditioning)

    expected_projection_input = torch.cat(
        [expected_decoder_output, conditioning], dim=-1
    )

    encoder_spy.assert_called_once_with(conditioning)
    decoder_spy.assert_called_once_with(expected_encoder_output)
    projection_spy.assert_called_once_with(expected_projection_input)

    assert torch.all(y == expected_projection_output)
