"""Tests for wait_ready.sh -- the readiness gates that replace run_stack.sh's
fixed sleeps.

These exercise the real shell functions against real files/sockets; there is
no stack and no mocking. The gates are the risky part of dropping the sleeps,
so they are tested directly rather than only through a stack bring-up.
"""
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAIT_READY = os.path.join(APP_DIR, "wait_ready.sh")

# The simulator sizes /tmp/fpga_* to 0x12000 + 0x1000 (configs/src/ipc/mod.rs),
# and gc mmaps len 0x1000 at offset 0x1000. Anything shorter is a SIGBUS.
FPGA_SIZE = 0x13000


def call(func, *args):
    """Run one wait_ready.sh function; return (returncode, stderr)."""
    script = 'source "%s"; %s "$@"' % (WAIT_READY, func)
    p = subprocess.run(["bash", "-c", script, "_"] + [str(a) for a in args],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stderr.decode()


class TestWaitFileSize(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "fpga_alice")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def test_returns_at_once_when_already_big_enough(self):
        with open(self.path, "wb") as f:
            f.truncate(FPGA_SIZE)
        t0 = time.time()
        rc, _ = call("wait_file_size", self.path, FPGA_SIZE, 2.0)
        self.assertEqual(rc, 0)
        self.assertLess(time.time() - t0, 0.5, "should not have polled at all")

    def test_rejects_a_file_that_exists_but_is_empty(self):
        # The actual race: the simulator creates the file with O_TRUNC and only
        # then set_len()s it. A gate that accepts mere existence lets gc mmap a
        # 0-byte file and take a SIGBUS. This is the whole reason for the size.
        with open(self.path, "wb"):
            pass
        rc, err = call("wait_file_size", self.path, FPGA_SIZE, 0.3)
        self.assertNotEqual(rc, 0)
        self.assertIn("fpga_alice", err)

    def test_waits_for_a_file_that_is_sized_late(self):
        def grow():
            time.sleep(0.2)
            with open(self.path, "wb") as f:
                f.truncate(FPGA_SIZE)

        with open(self.path, "wb"):
            pass
        t = threading.Thread(target=grow)
        t.start()
        try:
            rc, err = call("wait_file_size", self.path, FPGA_SIZE, 3.0)
        finally:
            t.join()
        self.assertEqual(rc, 0, err)

    def test_times_out_when_the_file_never_appears(self):
        rc, err = call("wait_file_size", self.path, FPGA_SIZE, 0.3)
        self.assertNotEqual(rc, 0)
        self.assertIn("timeout", err.lower())


class TestWaitExists(unittest.TestCase):
    """FIFOs and unix sockets are always 0 bytes; existence is the gate."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "gc_alice_angle.fifo")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def test_accepts_a_fifo(self):
        os.mkfifo(self.path)
        rc, err = call("wait_exists", self.path, 2.0)
        self.assertEqual(rc, 0, err)

    def test_times_out_when_missing(self):
        rc, err = call("wait_exists", self.path, 0.3)
        self.assertNotEqual(rc, 0)
        self.assertIn("timeout", err.lower())


class TestWaitTcpListen(unittest.TestCase):
    def test_accepts_a_real_listener(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            rc, err = call("wait_tcp_listen", port, 2.0)
        finally:
            srv.close()
        self.assertEqual(rc, 0, err)

    def test_times_out_when_nobody_listens(self):
        # Bind and close to get a port that is almost certainly free.
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        rc, err = call("wait_tcp_listen", port, 0.3)
        self.assertNotEqual(rc, 0)
        self.assertIn("timeout", err.lower())


if __name__ == "__main__":
    unittest.main()
