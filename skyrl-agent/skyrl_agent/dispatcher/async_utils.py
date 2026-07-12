import asyncio
import queue
import threading
from concurrent import futures
from typing import Callable, Coroutine, Iterable, List

GENERAL_TIMEOUT: int = 15


class _AsyncLoopThread:
    """Own one event loop for coroutines called from synchronous agent code."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop = None
        self._thread = None
        self._submissions = queue.Queue()

    def submit(self, coro: Coroutine):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._ready.clear()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        self._ready.wait()
        result = futures.Future()
        self._submissions.put((coro, result))
        return result

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        pending = set()
        while True:
            if not pending:
                coro, result = self._submissions.get()
                self._create_task(loop, pending, coro, result)
            while True:
                try:
                    coro, result = self._submissions.get_nowait()
                except queue.Empty:
                    break
                self._create_task(loop, pending, coro, result)
            loop.run_until_complete(asyncio.sleep(0.01))

    @staticmethod
    def _create_task(loop, pending, coro, result):
        task = loop.create_task(coro)
        pending.add(task)

        def complete(completed):
            pending.remove(completed)
            if result.cancelled():
                return
            if completed.cancelled():
                result.cancel()
            elif completed.exception() is not None:
                result.set_exception(completed.exception())
            else:
                result.set_result(completed.result())

        task.add_done_callback(complete)


ASYNC_LOOP_THREAD = _AsyncLoopThread()


async def call_sync_from_async(fn: Callable, *args, **kwargs):
    """
    Shorthand for running a function in the default background thread pool executor
    and awaiting the result. The nature of synchronous code is that the future
    returned by this function is not cancellable
    """
    loop = asyncio.get_event_loop()
    coro = loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    result = await coro
    return result


def call_async_from_sync(corofn: Callable, timeout: float = GENERAL_TIMEOUT, *args, **kwargs):
    """
    Shorthand for running a coroutine in the default background thread pool executor
    and awaiting the result
    """

    if corofn is None:
        raise ValueError("corofn is None")
    if not asyncio.iscoroutinefunction(corofn):
        raise ValueError("corofn is not a coroutine function")

    async def arun():
        coro = corofn(*args, **kwargs)
        result = await coro
        return result

    future = ASYNC_LOOP_THREAD.submit(arun())
    futures.wait([future], timeout=timeout or None)
    return future.result()


async def call_coro_in_bg_thread(corofn: Callable, timeout: float = GENERAL_TIMEOUT, *args, **kwargs):
    """Function for running a coroutine in a background thread."""
    await call_sync_from_async(call_async_from_sync, corofn, timeout, *args, **kwargs)


async def wait_all(iterable: Iterable[Coroutine], timeout: int = GENERAL_TIMEOUT) -> List:
    """
    Shorthand for waiting for all the coroutines in the iterable given in parallel. Creates
    a task for each coroutine.
    Returns a list of results in the original order. If any single task raised an exception, this is raised.
    If multiple tasks raised exceptions, an AsyncException is raised containing all exceptions.
    """
    tasks = [asyncio.create_task(c) for c in iterable]
    if not tasks:
        return []
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        for task in pending:
            task.cancel()
        raise asyncio.TimeoutError()
    results = []
    errors = []
    for task in tasks:
        try:
            results.append(task.result())
        except Exception as e:
            errors.append(e)
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise AsyncException(errors)
    return [task.result() for task in tasks]


class AsyncException(Exception):
    def __init__(self, exceptions):
        self.exceptions = exceptions

    def __str__(self):
        return "\n".join(str(e) for e in self.exceptions)
