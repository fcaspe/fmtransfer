# FM Tone Transfer with Envelope Learning

## Install

To install with development dependencies:

```bash
$ pip install -e ".[dev]"
```

Install pre-commit hooks if developing and contributing:

```bach
$ pre-commit install
```

## Run

Code in this repo is accessed through the PyTorch Lightning CLI, which is available through the `fmtransfer` console script. To see help:

```bash
$ fmtransfer --help
```

To run an experiment, pass the appropriate config file to the `fit` subcommand:

```bash
$ fmtransfer fit -c cfg/paper_runs.yaml
```

To replicate results, please run:

```bash
$ source schedule/test/paper_runs.sh
```

