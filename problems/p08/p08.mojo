from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.testing import assert_equal

from std.time import perf_counter_ns


# ANCHOR: add_10_shared
comptime TPB = 4
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (2, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


def add_10_shared(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    # Allocate shared memory using stack_allocation
    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB]())

    # Computer thread's glbal index
    # block_dim.x = nummber of threads per block ie. 4
    # block_idx.x = which block it belongs to ie. 0 or 1
    # thread_idx.x = threads position inside the block.
    # so you get 0, 1, 2, ...5, 6, 7
    var global_i = block_dim.x * block_idx.x + thread_idx.x

    # Index shared block's memory
    # so values are 0, 1, 2, 3
    var local_i = thread_idx.x

    if global_i < size:
        shared[local_i] = a[global_i]

    # wait until all threads in block have finished writing to shared memory.
    # syncs threads inside the same block. doesn't sync different blocks with each other
    # not needed here cuz each thread reads only same slot it wrote.
    # becomes important when threads access data written by another thread.
    # in that casebarrier must happen after shared writes and before reads
    # eg. matrix multiplication
    barrier()

    if global_i < size:
        output[global_i] = shared[local_i] + 10

# ANCHOR_END: add_10_shared


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(1)

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)

        var start = perf_counter_ns()
        ctx.enqueue_function[add_10_shared, add_10_shared](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.synchronize()

        var elapsed = (perf_counter_ns() - start) / 1_000_000
        print("Kernel time:", elapsed, "ms")

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(11)
        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 08 complete ✅")
