from data.maze import MazeDataset
from data.utils import save_image
from model.tokenizers.maze import MazeTokenizer

if __name__ == "__main__":
    d_train = MazeDataset("./cache", train=True)
    d_test = MazeDataset("./cache", train=False)

    tokenizer = MazeTokenizer()

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

    print(x_seq)
