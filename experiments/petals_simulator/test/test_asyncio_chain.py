import time
import random
import asyncio


""" This is a simple example of a task chain consisting of two stages.
    A task can be thought of as an asynchronous pipeline executing in a single thread.
"""

async def stage1(task_id: int, arg: str) -> str:
    i = random.randint(0, 10)
    print(f"executing stage1(task={task_id}, arg=\"{arg}\"): {i} seconds")
    await asyncio.sleep(i)
    result = f"result{task_id}-1 derived from {arg}"
    print(f"stage1({arg}) -> {result}")
    return result


async def stage2(task_id: int, arg: str) -> str:
    i = random.randint(0, 10)
    print(f"executing stage2(task_id={task_id}, arg=\"{arg}\"): {i} seconds")
    await asyncio.sleep(i)
    result = f"result{task_id}-2 derived from {arg}"
    print(f"stage2({arg}) -> {result}")
    return result


async def task1(arg: str) -> None:
    TASK_ID = 1
    start = time.perf_counter()
    p1 = await stage1(TASK_ID, arg)
    p2 = await stage2(TASK_ID, p1)
    end = time.perf_counter() - start
    print(f"task{TASK_ID}({arg}) -> {p2} (took {end:0.2f} seconds)")


async def main(*args):
    await asyncio.gather(*(task1(n) for n in args))


if __name__ == "__main__":
    import sys
    random.seed(42)
    args = [1, 2, 3] if len(sys.argv) == 1 else map(int, sys.argv[1:])
    start = time.perf_counter()
    asyncio.run(main(*args))
    end = time.perf_counter() - start
    print(f"program finished in {end:0.2f} seconds.")
