def print_tensor(tensor):
    print(f"(shape: {tensor.shape}):")
    n_br, _, _ = tensor.shape
    for i in range(n_br):
        print(tensor[i, :, :].int())
    print()
