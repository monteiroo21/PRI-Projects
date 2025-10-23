import multiprocessing
import time

from sapien.core.limit_memory import start_memory_monitor


def worker(worker_id):
    """Allocate ~600MB in each worker using Python built-ins"""

    # Alternatively, use bytearray for more predictable memory usage
    _ = list(range(20_000_000))

    print(f"\nWorker {worker_id} allocated memory, sleeping...")
    time.sleep(10)


if __name__ == "__main__":
    start_memory_monitor(show_memory_updates=True)

    processes = []
    for i in range(8):
        p = multiprocessing.Process(target=worker, args=(i,))
        p.start()
        processes.append(p)
        time.sleep(0.5)  # Stagger starts to see memory grow

    for p in processes:
        p.join()

    print("\nAll processes completed!")
