from data.maze import MazeDataset

if __name__ == "__main__":
    d = MazeDataset("./cache")
    x, y = d[0]

    print(x.shape)
    print(y.shape)
