from std.memory import UnsafePointer
from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.testing import assert_equal

# ANCHOR: add_10
comptime SIZE = 1 # size of input and output buffers
comptime BLOCKS_PER_GRID = 1 # number of thread blocks in the grid
comptime THREADS_PER_BLOCK = SIZE # number of threads in each block
comptime dtype = DType.int8 # data type for the elements in the buffers.

# UnsafePointer: pulls address to memory and allows you to store and retrieve address in that memory
# Used to access data buffers. can point to any type of value with value type as parameter.

def add_10(output: UnsafePointer[Scalar[dtype], MutAnyOrigin], a: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
    i = thread_idx.x # get each thread's unique position exclusively on the x axis
    # FILL ME IN (roughly 1 line)
    output[i] = a[i] + 10

# ANCHOR_END: add_10


def main() raises:
    with DeviceContext() as ctx:
        # GPU context is setup
        # first 2 buffers are created "out" (output) and "a" (input buffer) and initialize with 0
        # enqueue - GPU doesn't execute commands immediately, you need to submit a task which gets added to queue
        var out = ctx.enqueue_create_buffer[dtype](SIZE) 
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        
        # Fill input buffer, a with values 0 - 4
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i)

        # enqueue kernel for execution. kernel = function
        ctx.enqueue_function[add_10, add_10](
            out,
            a,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)
        ctx.synchronize()

        for i in range(SIZE):
            expected[i] = Scalar[dtype](i + 10)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 01 complete ✅")
