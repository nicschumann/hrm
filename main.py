from data.maze import MazeDataset
from data.utils import save_image

if __name__ == "__main__":
    d = MazeDataset("./cache")
    x, y = d[0]

    print(x.shape, x.dtype)
    print(y.shape, y.dtype)

    save_image("test3.png", x)
