from data.maze import MazeDataset
from data.utils import save_image
from model.tokenizers.maze import MazeTokenizer

if __name__ == "__main__":
    d = MazeDataset("./cache")
    tokenizer = MazeTokenizer()
    x, y = d[0]

    x.unsqueeze_(0)  # NOTE(Nic): add batch dim for testing.
    y.unsqueeze_(0)  # NOTE(Nic): add batch dim for testing.

    x_seq, y_seq = tokenizer(x, y)
