# Smart-IOT-AI-Agent

Code repository linked from the paper **"Dependency-Aware Reliable Orchestration for Smart-IoT Control with Reasoning-Aligned LLMs"** (2026 6th International Conference on Electrical, Computer and Energy Technologies, ICECET; [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11632566)).

## What is in this repository

One file: `Fine-tuned LLM Evaluation.py`, a notebook exported to a single Python script (about 3,700 lines, 70 sections). It walks through building, fine-tuning and scoring a GPT-2 style language model in PyTorch:

| Sections | Content |
|---|---|
| 1 to 9 | Tokenization, byte pair encoding with `tiktoken`, data loader, token and positional embeddings |
| 10 to 21 | Self attention, causal attention, multi-head attention |
| 22 to 33 | GPT model built from scratch, text generation |
| 34 to 48 | Loss and perplexity, training loop, temperature and top-k decoding, saving and loading weights |
| 49 | Loading pretrained GPT-2 weights from OpenAI |
| 50 to 57 | Fine-tuning for classification (SMS spam dataset from the UCI repository) |
| 58 to 69 | Instruction fine-tuning of GPT-2 medium (355M) on `instruction-data.json` in Alpaca prompt format |
| 70 | Scoring the fine-tuned model's responses from 0 to 100 with `llama3` served by a local Ollama instance |

The script does not contain the IoT orchestration pipeline described in the paper.

## Requirements

The repository has no requirements file. The script imports:

```bash
pip install torch tiktoken numpy matplotlib pandas tensorflow tqdm psutil
```

It also needs:

- `the-verdict.txt` next to the script, or passed with `--text-file`. The script reads it in Section 2.
- A `gpt_download3` module that provides `download_and_load_gpt2`. It is not in this repository.
- A running Ollama server with the `llama3` model for Section 70 (`http://localhost:11434/api/chat`).

The script downloads the SMS spam dataset and `instruction-data.json` on first use.

## Usage

```bash
python "Fine-tuned LLM Evaluation.py" --text-file the-verdict.txt
```

**Known issue:** the script does not run as committed. It still contains notebook leftovers that are not valid Python (a stray character on line 2021 and a `pip install` line on line 2279), and the `gpt_download3` module is missing. Read it as a reference, or run it section by section in a notebook after fixing those lines.

## Citation

```bibtex
@INPROCEEDINGS{11632566,
  author={Sarowar, Md Selim and Islam, Md Tanvir and Nuruzzaman, Md. and Islam Mozumder, Md Ariful},
  booktitle={2026 6th International Conference on Electrical, Computer and Energy Technologies (ICECET)},
  title={Dependency-Aware Reliable Orchestration for Smart-IoT Control with Reasoning-Aligned LLMs},
  year={2026},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICECET65726.2026.11632566}}
```

## License

This repository has no LICENSE file.
