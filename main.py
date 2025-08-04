from data.maze import MazeDataset
from model.tokenizers.maze import MazeTokenizer, InputTokens, OutputTokens
from model.hrm import HRM, HRMConfig
from model import train_loop

if __name__ == "__main__":
    train_data = MazeDataset("./cache", train=True)
    test_data = MazeDataset("./cache", train=False)

    d_model = 64
    max_seq = 256

    tokenizer = MazeTokenizer()

    hrm = HRM(
        HRMConfig(len(InputTokens), len(OutputTokens), d_model, max_seq_len=max_seq)
    )

    print(f"hrm params: {sum(p.numel() for p in hrm.parameters()):,}")

    train_loop(hrm, tokenizer, train_data, test_data, M=4)
