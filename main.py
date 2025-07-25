from data.maze import MazeDataset
from data.utils import save_image
from model.tokenizers.maze import MazeTokenizer, InputTokens, OutputTokens
from model.input import InputNetwork
from model.recurrence import RecurrentModule, get_dummy_vars
from model.output import OutputNetwork

if __name__ == "__main__":
    d_train = MazeDataset("./cache", train=True)
    d_test = MazeDataset("./cache", train=False)
    d_model = 64
    max_seq = 256

    tokenizer = MazeTokenizer()

    f_input = InputNetwork(len(InputTokens), d_model, max_seq)
    f_rec = RecurrentModule(d_model, 4, d_model)
    f_output = OutputNetwork(d_model, len(OutputTokens), max_seq)

    print(f"input params: {sum(p.numel() for p in f_input.parameters()):,}")
    print(f"rec params: {sum(p.numel() for p in f_rec.parameters()):,}")
    print(f"out params: {sum(p.numel() for p in f_output.parameters()):,}")

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
    x_prime = f_input(x_seq)

    z_h, z_l = get_dummy_vars(d_model)

    print("encoded_seq", x_prime.shape)
    print("z_h", z_h.shape)
    print("z_l", z_l.shape)

    z_out = f_rec(z_l, z_h, x_prime)
    pred_logits = f_output(z_out, y_seq.shape[1])

    print("z_out", z_out.shape)
    print("pred", pred_logits.shape)
    print(pred_logits.argmax(dim=-1).shape)

    # batch_size, seq_len, d_model = x_prime.shape
