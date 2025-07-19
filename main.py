from data.maze import MazeDataset
from data.utils import save_image
from model.tokenizers.maze import MazeTokenizer, InputTokens
from model.input import InputNetwork

if __name__ == "__main__":
    d_train = MazeDataset("./cache", train=True)
    d_test = MazeDataset("./cache", train=False)

    tokenizer = MazeTokenizer()
    f_input = InputNetwork(len(InputTokens), 256, 256)

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

    out = f_input(x_seq)
