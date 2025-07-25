from data.maze import MazeDataset
from data.utils import save_image
from model.tokenizers.maze import MazeTokenizer, InputTokens, OutputTokens
from model import HRM, HRMConfig

if __name__ == "__main__":
    d_train = MazeDataset("./cache", train=True)
    d_test = MazeDataset("./cache", train=False)
    d_model = 64
    max_seq = 256

    tokenizer = MazeTokenizer()

    config = HRMConfig(
        len(InputTokens), len(OutputTokens), d_model, max_seq_len=max_seq
    )
    hrm = HRM(config)

    print(f"hrm params: {sum(p.numel() for p in hrm.parameters()):,}")

    x, y = d_train[0]
    save_image("test-data/train.png", x)

    x.unsqueeze_(0)  # NOTE(Nic): add batch dim for testing.
    y.unsqueeze_(0)  # NOTE(Nic): add batch dim for testing.

    x_seq, y_seq = tokenizer(x, y)

    x, y = d_test[0]
    save_image("test-data/test.png", x)

    x.unsqueeze_(0)  # NOTE(Nic): add batch dim for testing.
    y.unsqueeze_(0)  # NOTE(Nic): add batch dim for testing.

    x_seq, y_seq = tokenizer(x, y)
    pred_logits = hrm(x_seq, y_seq.size(1))

    loss = hrm.loss(y_seq, pred_logits)
    print(loss)

    print("inp", x_seq.shape, x_seq.dtype)
    print("tar", y_seq.shape, y_seq.dtype)
    print("pred", pred_logits.shape, pred_logits.dtype)

    # batch_size, seq_len, d_model = x_prime.shape
