import torch

def maxpool2d(A, kernel_size, stride=1, padding=0, return_indices=False):
    input = torch.from_numpy(A).unsqueeze(0).type(torch.float32)
    if return_indices:
        with torch.no_grad():
            output, indices = torch.nn.functional.max_pool2d(input, kernel_size, stride=stride, padding=padding,
                                                             dilation=1, ceil_mode=False, return_indices=True)
        output = output.squeeze().numpy()
        indices = indices.squeeze().numpy()
        return output, indices
    else:
        with torch.no_grad():
            output = torch.nn.functional.max_pool2d(input, kernel_size, stride=stride, padding=padding, dilation=1,
                                                    ceil_mode=False, return_indices=False)

        output = output.squeeze().numpy()
        return output