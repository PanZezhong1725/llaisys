import llaisys

import torch
from test_utils import *
import argparse


def test_tensor():
    torch_tensor = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
    llaisys_tensor = llaisys.Tensor(
        (3, 4, 5), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )

    # Test load
    print("===Test load===")
    llaisys_tensor.load(torch_tensor.data_ptr())
    llaisys_tensor.debug()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor, torch_tensor)

    # Test view
    print("===Test view===")
    torch_tensor_view = torch_tensor.view(6, 10)
    llaisys_tensor_view = llaisys_tensor.view(6, 10)
    llaisys_tensor_view.debug()
    assert llaisys_tensor_view.shape() == torch_tensor_view.shape
    assert llaisys_tensor_view.strides() == torch_tensor_view.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_view, torch_tensor_view)

    # Test permute
    print("===Test permute===")
    torch_tensor_perm = torch_tensor.permute(2, 0, 1)
    llaisys_tensor_perm = llaisys_tensor.permute(2, 0, 1)
    llaisys_tensor_perm.debug()
    assert llaisys_tensor_perm.shape() == torch_tensor_perm.shape
    assert llaisys_tensor_perm.strides() == torch_tensor_perm.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_perm, torch_tensor_perm)

    # Test slice
    print("===Test slice===")
    torch_tensor_slice = torch_tensor[:, :, 1:4]
    llaisys_tensor_slice = llaisys_tensor.slice(2, 1, 4)
    llaisys_tensor_slice.debug()
    assert llaisys_tensor_slice.shape() == torch_tensor_slice.shape
    assert llaisys_tensor_slice.strides() == torch_tensor_slice.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_slice, torch_tensor_slice)


def test_stride_compatible_view():
    """A view may reshape within stride-compatible storage chunks."""
    torch_base = torch.arange(60, dtype=torch_dtype("i64")).reshape(2, 3, 10)
    llaisys_base = llaisys.Tensor(
        (2, 3, 10), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )
    llaisys_base.load(torch_base.data_ptr())

    # The slice has shape (2, 3, 5), strides (30, 10, 1), and gaps between
    # rows. Its outer two dimensions form one chunk, while the final dimension
    # is a separate contiguous chunk.
    torch_slice = torch_base[:, :, :5]
    llaisys_slice = llaisys_base.slice(2, 0, 5)
    assert llaisys_slice.shape() == torch_slice.shape
    assert llaisys_slice.strides() == torch_slice.stride()

    torch_view = torch_slice.view(6, 5)
    llaisys_view = llaisys_slice.view(6, 5)
    assert llaisys_view.strides() == torch_view.stride()
    assert check_equal(llaisys_view, torch_view)

    # Merging across the storage gap would require moving data, so view must
    # reject it. reshape/contiguous are the copying alternatives in C++.
    try:
        llaisys_slice.view(2, 15)
    except ValueError:
        pass
    else:
        raise AssertionError("Incompatible non-contiguous view was accepted")


def test_scalar_and_empty_tensor_metadata():
    scalar = llaisys.Tensor(
        (), dtype=llaisys_dtype("f32"), device=llaisys_device("cpu")
    )
    assert scalar.shape() == ()
    assert scalar.strides() == ()
    assert scalar.is_contiguous()

    empty = llaisys.Tensor(
        (2, 0, 3), dtype=llaisys_dtype("f32"), device=llaisys_device("cpu")
    )
    assert empty.shape() == (2, 0, 3)
    assert empty.is_contiguous()
    assert empty.view(0, 6).shape() == (0, 6)


if __name__ == "__main__":
    test_tensor()
    test_stride_compatible_view()
    test_scalar_and_empty_tensor_metadata()

    print("\n\033[92mTest passed!\033[0m\n")
