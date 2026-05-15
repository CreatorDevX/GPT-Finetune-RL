import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def capture_io():
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdin_block = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stdout_capture):
        with contextlib.redirect_stderr(stderr_capture):
            with redirect_stdin(stdin_block):
                yield stdout_capture, stderr_capture


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    if platform.uname().system != "Darwin":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.kill = None
    _os.system = None
    _os.putenv = None
    _os.remove = None
    _os.removedirs = None
    _os.rmdir = None
    _os.fchdir = None
    _os.setuid = None
    _os.fork = None
    _os.forkpty = None
    _os.killpg = None
    _os.rename = None
    _os.renames = None
    _os.truncate = None
    _os.replace = None
    _os.unlink = None
    _os.fchmod = None
    _os.fchown = None
    _os.chmod = None
    _os.chown = None
    _os.chroot = None
    _os.lchflags = None
    _os.lchmod = None
    _os.lchown = None
    _os.getcwd = None
    _os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None

    __builtins__["help"] = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _unsafe_execute(code: str, timeout: float, maximum_memory_bytes: Optional[int], result_dict):
    with create_tempdir():
        import os as _os
        import shutil as _shutil
        rmtree = _shutil.rmtree
        rmdir = _os.rmdir
        chdir_fn = _os.chdir
        unlink = _os.unlink

        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        result_dict.update({
            "success": False,
            "stdout": "",
            "stderr": "",
            "timeout": False,
            "memory_exceeded": False,
            "error": None,
        })

        try:
            exec_globals = {}
            with capture_io() as (stdout_capture, stderr_capture):
                with time_limit(timeout):
                    exec(code, exec_globals)
            result_dict.update({
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            })
        except TimeoutException:
            result_dict.update({"timeout": True, "error": "Execution timed out"})
        except MemoryError as e:
            result_dict.update({"memory_exceeded": True, "error": f"Memory limit exceeded: {e}"})
        except BaseException as e:
            result_dict.update({"error": f"{type(e).__name__}: {e}"})

        _shutil.rmtree = rmtree
        _os.rmdir = rmdir
        _os.chdir = chdir_fn
        _os.unlink = unlink


def execute_code(code: str, timeout: float = 5.0, maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024) -> ExecutionResult:
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    p = multiprocessing.Process(target=_unsafe_execute, args=(code, timeout, maximum_memory_bytes, result_dict))
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        return ExecutionResult(success=False, stdout="", stderr="", error="Execution timed out (process killed)", timeout=True)

    if not result_dict:
        return ExecutionResult(success=False, stdout="", stderr="", error="Execution failed (no result returned)", timeout=True)

    return ExecutionResult(
        success=result_dict["success"],
        stdout=result_dict["stdout"],
        stderr=result_dict["stderr"],
        error=result_dict["error"],
        timeout=result_dict["timeout"],
        memory_exceeded=result_dict["memory_exceeded"],
    )
