# TouchRL

A blind fingertip learns to find and name a hidden shape by touch alone.

The world is a 16x16 grid with one hidden shape. The agent feels only the 3x3 cells under its finger and decides where to touch next. Built from scratch with PyTorch and trained with REINFORCE on a laptop CPU.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
