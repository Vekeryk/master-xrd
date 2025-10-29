import torch


def fparam(param=None, index=None, arr=None):
    if arr is not None:
        if len(arr) != 7:
            raise ValueError("Array must have exactly 7 elements.")

        return f"Dmax1={arr[0]:.4f}, D01={arr[1]:.4f}, L1={arr[2]*1e8:.0f} " +\
               f"Rp1={arr[3]*1e8:.0f}, D02={arr[4]:.4f}, L2={arr[5]*1e8:.0f}, Rp2={arr[6]*1e8:.0f}"

    if param is not None and index is not None:
        if index in [0, 1, 4]:
            return f"{param:.4f}"
        else:
            return f"{param:.2e}"

    raise ValueError("Invalid parameters.")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    print("No GPU available, using CPU")
    return torch.device("cpu")
