import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import modal_app
from modal_worker.worker_base import WorkerBase


class _JoinObservedThread(threading.Thread):
    def __init__(self, *, target):
        super().__init__(target=target)
        self.join_called = threading.Event()

    def join(self, timeout=None):
        self.join_called.set()
        super().join(timeout=1)


class _ClosingLoop:
    def is_closed(self):
        return False

    def call_soon_threadsafe(self, callback):
        raise RuntimeError("Event loop is closed")


class _Engine:
    def __init__(self):
        self.stop_called = False

    def stop(self):
        self.stop_called = True


class _Handler:
    def __init__(self, engine):
        self.batched_engine = engine


class ModalWorkerLifecycleTests(unittest.TestCase):
    @staticmethod
    def _initialize_shutdown_controls(worker):
        worker._serve_controls_lock = threading.Lock()
        worker._serve_shutdown_requested = False

    def test_cleanup_before_control_publication_is_consumed_after_publication(self):
        worker = WorkerBase()
        self._initialize_shutdown_controls(worker)
        serve_started = threading.Event()
        publish_controls = threading.Event()
        thread_errors = []

        def run_serve_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            serve_started.set()
            publish_controls.wait()
            try:
                worker._publish_serve_controls(loop)
                loop.run_until_complete(worker._serve_stop_event.wait())
            except BaseException as exc:
                thread_errors.append(exc)
            finally:
                loop.close()

        serve_thread = _JoinObservedThread(target=run_serve_loop)
        worker._serve_thread = serve_thread
        serve_thread.start()
        self.assertTrue(serve_started.wait(timeout=1))

        cleanup = WorkerBase.cleanup._get_raw_f()
        cleanup_finished = threading.Event()

        def run_cleanup():
            try:
                cleanup(worker)
            finally:
                cleanup_finished.set()

        cleanup_thread = threading.Thread(target=run_cleanup)
        cleanup_thread.start()
        self.assertTrue(serve_thread.join_called.wait(timeout=1))
        publish_controls.set()
        self.assertTrue(cleanup_finished.wait(timeout=2))

        try:
            self.assertFalse(serve_thread.is_alive())
            self.assertTrue(worker._serve_stop_event.is_set())
            if thread_errors:
                raise thread_errors[0]
        finally:
            if serve_thread.is_alive():
                worker._serve_loop.call_soon_threadsafe(worker._serve_stop_event.set)
            cleanup_thread.join(timeout=1)
            serve_thread.join(timeout=1)

    def test_cleanup_completes_when_loop_closes_before_threadsafe_signal(self):
        worker = WorkerBase()
        self._initialize_shutdown_controls(worker)
        worker._serve_loop = _ClosingLoop()
        worker._serve_stop_event = threading.Event()

        cleanup = WorkerBase.cleanup._get_raw_f()
        cleanup(worker)

        self.assertTrue(worker._serve_shutdown_requested)

    def test_partial_startup_does_not_stop_engine_while_serve_thread_is_alive(self):
        worker = WorkerBase()
        self._initialize_shutdown_controls(worker)
        keep_running = threading.Event()
        serve_started = threading.Event()
        engine = _Engine()
        worker._handler = _Handler(engine)

        def run_partial_startup():
            serve_started.set()
            keep_running.wait()

        serve_thread = _JoinObservedThread(target=run_partial_startup)
        worker._serve_thread = serve_thread
        serve_thread.start()
        self.assertTrue(serve_started.wait(timeout=1))

        cleanup = WorkerBase.cleanup._get_raw_f()
        cleanup(worker)

        try:
            self.assertTrue(serve_thread.is_alive())
            self.assertFalse(engine.stop_called)
        finally:
            keep_running.set()
            serve_thread.join(timeout=1)

    def test_cleanup_signals_serve_controls_published_during_startup(self):
        worker = WorkerBase()
        self._initialize_shutdown_controls(worker)
        controls_published = threading.Event()
        thread_errors = []

        def run_serve_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                worker._publish_serve_controls(loop)
                controls_published.set()
                loop.run_until_complete(worker._serve_stop_event.wait())
            except BaseException as exc:
                thread_errors.append(exc)
                controls_published.set()
            finally:
                loop.close()

        serve_thread = threading.Thread(target=run_serve_loop)
        worker._serve_thread = serve_thread
        serve_thread.start()
        self.assertTrue(controls_published.wait(timeout=1))
        if thread_errors:
            raise thread_errors[0]

        cleanup = WorkerBase.cleanup._get_raw_f()
        cleanup(worker)

        self.assertFalse(serve_thread.is_alive())
        self.assertTrue(worker._serve_stop_event.is_set())


class ShutdownRuntimeSweepTests(unittest.IsolatedAsyncioTestCase):
    def _make_worker_with_handler(self):
        worker = WorkerBase()
        handler = SimpleNamespace(
            batched_engine=_Engine(),
            request_active_session_shutdown=AsyncMock(return_value=0),
            drain_finalization_tasks=AsyncMock(return_value=True),
            _active_sessions={},
            ws_server=SimpleNamespace(is_running=False),
        )
        worker._handler = handler
        worker._site = None
        worker._batched_warmup_task = None
        worker._batched_warmup_stop = None
        worker._runner = None
        return worker

    async def test_shutdown_runtime_cancels_orphaned_reconnect_expiry_tasks(self):
        worker = self._make_worker_with_handler()

        orphan_expiry = asyncio.create_task(
            asyncio.sleep(3600), name="reconnect-expiry-orphan-1"
        )
        unrelated = asyncio.create_task(asyncio.sleep(3600), name="unrelated-watchdog")

        await worker._shutdown_runtime()

        self.assertTrue(orphan_expiry.done())
        self.assertTrue(orphan_expiry.cancelled())
        self.assertFalse(unrelated.done())
        unrelated.cancel()

    async def test_shutdown_runtime_sweep_is_noop_without_orphans(self):
        worker = self._make_worker_with_handler()

        await worker._shutdown_runtime()

        self.assertEqual(worker._handler.request_active_session_shutdown.await_count, 1)
        self.assertTrue(worker._handler.drain_finalization_tasks.await_count >= 1)


if __name__ == "__main__":
    unittest.main()
