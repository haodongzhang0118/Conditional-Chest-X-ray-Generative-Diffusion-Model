from einops import rearrange
def window_partition(x, window_size):
    # This is the Patchify function that can divide the image into smaller patch
    # The output size with default value of reshape_seq is (B, H // window_size, W // window_size, window_size, window_size, C)
    # Batch size, number of window in height, number of window in width, window size, window size, channel

    # reshape_seq = True: Reshape the 2D patch into a 1D sequence
    # The output size with reshape_seq = True is (B, window_size * window_size, C) (H, W, C)
    # Batch size, Patch Sequence, Channal

    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1)
    windows = rearrange(windows, 'b h w p1 p2 c -> b (h w) (p1 p2) c')
    return windows # (B N P^2 C)

def window_reverse(windows, window_size, H, W):
    # Reverse the patchified image back to the original image
    # (B, N, P^2, C)
    windows = rearrange(windows, 'b (h w) (p1 p2) c -> b h w p1 p2 c', p1=window_size, p2=window_size, h= H // window_size, w = W // window_size)
    windows = rearrange(windows, 'b h w p1 p2 c -> b c (h p1) (w p2)')
    return windows # (B C H W)