"""Heterogenous memory monitor for Python. It should work in all OS."""
import logging
import os
import threading
import time
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryMonitorConfig:
    limit_mb: int
    interval_sec: float
    memory_updates: bool = True

    @property
    def limit_mb_mp(self) -> int:
        # -- We allow for an additional 500MB for MP solutions
        # -- This should account for most of the python overheads
        return self.limit_mb + 500


def start_memory_monitor(show_memory_updates: bool = False):
    """Initializes and starts a background thread to monitor memory usage.

    The thread is set as a "daemon" so it will exit automatically when the
    main program finishes.
    """

    memory_config = MemoryMonitorConfig(
        limit_mb=20000, interval_sec=0.1, memory_updates=show_memory_updates
    )
    monitor_thread = threading.Thread(
        target=memory_monitor_worker, args=(memory_config,), daemon=True
    )
    logger.info(
        f"[MemoryGuard] Monitoring started."
        f" Crashing if TOTAL usage exceeds {memory_config.limit_mb} MB."
    )
    monitor_thread.start()


def get_current_memory_usage_in_mb() -> int:
    """Get the current memory usage in MB.

    NOTE: This reading is only correct for the parent process.
    """

    memory_usage, _ = get_memory_usage_in_mb_and_child_processes(None)
    return memory_usage


def get_memory_usage_in_mb_and_child_processes(
    process: psutil.Process | None = None,
) -> tuple[int, list[psutil.Process]]:
    """Get the current memory usage in MB.

    To account for multiprocessing scenarios we will add the memory usage of all child processes.

    NOTE: This reading is only correct for the parent process.
    """

    # -- Get the memory usage of the parent process --
    current_process = psutil.Process(os.getpid()) if process is None else process
    rss_memory_bytes = current_process.memory_info().rss

    all_child_processes = current_process.children(recursive=True)
    for child in all_child_processes:
        try:
            rss_memory_bytes += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Child might have terminated or is inaccessible, skip it
            all_child_processes.remove(child)

    return int(rss_memory_bytes / (1024 * 1024)), all_child_processes


def memory_monitor_worker(memory_config: MemoryMonitorConfig):
    """The worker function that runs in the background.

    It periodically checks the current process's memory usage AND all child processes.
    """

    # -- Get the current process object --
    current_process = psutil.Process(os.getpid())

    while True:
        try:
            memory_usage, child_processes = get_memory_usage_in_mb_and_child_processes(
                current_process
            )
            process_count = len(child_processes) + 1

            if memory_config.memory_updates:
                print(
                    f"\r[MemoryGuard] Total usage across {process_count}"
                    f" process(es): {memory_usage:.2f} MB",
                    end="",
                )

            memory_limit = (
                memory_config.limit_mb_mp if len(child_processes) > 0 else memory_config.limit_mb
            )

            if memory_usage > memory_limit:
                logger.error(
                    f"\n\n[MemoryGuard] CRITICAL: Total memory usage ({memory_usage:.2f} MB) "
                    f"across {process_count} process(es) exceeded the limit of {memory_limit} MB. "
                    f"Terminating program."
                )

                # -- Kill all child processes first --
                for child in child_processes:
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # -- Use os._exit(1) for an immediate, forceful exit.
                # -- This is more abrupt than sys.exit() and suitable for a "crash".
                os._exit(1)

        except psutil.NoSuchProcess:
            # -- The process might have already exited, so we can stop the thread. --
            break
        except Exception as e:
            logger.error(f"\n[MemoryGuard] Error in memory monitor: {e}")
            break

        time.sleep(memory_config.interval_sec)
