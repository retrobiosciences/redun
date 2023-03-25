"""
Redun Executor class for running tasks and scripts in a local conda environment.
"""
from configparser import SectionProxy
import os
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor
from shlex import quote
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any, Dict, List, Optional, Tuple

from redun.config import create_config_section
from redun.executors.base import Executor, register_executor
from redun.scheduler import Job, Scheduler, Traceback
from redun.executors.command import get_oneshot_command
from redun.scripting import ScriptError, get_task_command
from redun.utils import get_import_paths, pickle_dump


@register_executor("conda")
class CondaExecutor(Executor):
    """
    Executor that runs tasks and scripts in a local conda environment.
    """

    def __init__(
        self,
        name: str,
        scheduler: Optional[Scheduler] = None,
        config: Optional[SectionProxy] = None,
    ):
        super().__init__(name, scheduler=scheduler)

        # Parse config.
        if not config:
            config = create_config_section()

        self.max_workers = config.getint("max_workers", 20)
        self.default_env = config.get("conda_environment")
        self._scratch_prefix_rel = config.get("scratch", ".scratch_redun")
        self._scratch_prefix_abs: Optional[str] = None

        self._thread_executor: Optional[ThreadPoolExecutor] = None

    @property
    def _scratch_prefix(self) -> str:
        if not self._scratch_prefix_abs:
            if os.path.isabs(self._scratch_prefix_rel):
                self._scratch_prefix_abs = self._scratch_prefix_rel
            else:
                # TODO: Is there a better way to find the path of the current
                # config dir?
                try:
                    assert self._scheduler
                    base_dir = os.path.abspath(
                        self._scheduler.config["repos"]["default"]["config_dir"]
                    )
                except KeyError:
                    # Use current working directory as base_dir if default
                    # config_dir cannot be found.
                    base_dir = os.getcwd()

                self._scratch_prefix_abs = os.path.normpath(
                    os.path.join(base_dir, self._scratch_prefix_rel)
                )
        assert self._scratch_prefix_abs
        return self._scratch_prefix_abs

    def _start(self) -> None:
        """
        Start pool on first Job submission.
        """
        os.makedirs(self._scratch_prefix, exist_ok=True)
        if not self._thread_executor:
            self._thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def stop(self) -> None:
        """
        Stop Executor pools.
        """
        if self._thread_executor:
            self._thread_executor.shutdown()
            self._thread_executor = None

    def _submit(self, job: Job) -> None:
        # Ensure pool are started.
        self._start()

        # Run job in a new thread or process.
        assert self._thread_executor
        assert job.task

        def on_done(future):
            success, result = future.result()
            if success:
                self._scheduler.done_job(job, result)
            else:
                error, traceback = result
                self._scheduler.reject_job(job, error, traceback)

        assert job.args
        args, kwargs = job.args
        self._thread_executor.submit(
            execute, self._scratch_prefix, job, self.get_job_env(job), job.task.fullname, args, kwargs
        ).add_done_callback(on_done)

    def submit(self, job: Job) -> None:
        assert job.task
        assert not job.task.script
        self._submit(job)

    def submit_script(self, job: Job) -> None:
        assert job.task
        assert job.task.script
        self._submit(job)

    def get_job_env(self, job: Job) -> str:
        """
        Return the conda environment name for the given job.
        """
        result = job.get_option("conda", self.default_env)
        if result is None:
            raise RuntimeError("No conda environment name or default value provided.")
        return result


def execute(
    scratch_path: str,
    job: Job,
    env_name: str,
    task_fullname: str,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
) -> Tuple[bool, Any]:
    """
    Run a job in a local conda environment.
    """
    assert job.task

    if kwargs is None:
        kwargs = {}
    task_files_dir = mkdtemp(prefix=f"{task_fullname}_", dir=scratch_path)
    input_path = os.path.join(task_files_dir, "input")
    output_path = os.path.join(task_files_dir, "output")
    error_path = os.path.join(task_files_dir, "error")

    command_path = os.path.join(task_files_dir, "command")
    command_output_path = os.path.join(task_files_dir, "command_output")
    command_error_path = os.path.join(task_files_dir, "command_error")

    if job.task.script:
        # Careful. This will execute the body of the task in the host environment.
        # The rest of redun all works like this so I'm not changing it here.
        inner_command = [get_task_command(job.task, args, kwargs)]
    else:
        # # Serialize arguments to input file.
        # with open(input_path, "wb") as f:
        #     pickle_dump([args, kwargs], f)
        inner_command = get_oneshot_command(
            scratch_path, job, job.task, args, kwargs, input_path=input_path, output_path=output_path, error_path=error_path
        )

    command = wrap_command(
        inner_command, env_name, command_path, command_output_path, command_error_path
    )
    cmd_result = subprocess.run(command, check=False, capture_output=False)

    if not job.task.script:
        return handle_oneshot_output(output_path, error_path, command_error_path)
    else:
        return handle_script_output(
            cmd_result.returncode, command_output_path, command_error_path
        )


def wrap_command(
    command: List[str],
    env_name: str,
    command_path: str,
    command_output_path: str,
    command_error_path: str,
) -> List[str]:
    """
    Given a bash command:
    1. Wrap it in a `conda run` command.
    2. Write it to a file.
    3. Generate and return a command to execute the file, teeing stdout and stderr to files.
    """
    conda_command = ["conda", "run", "--no-capture-output", "-n", env_name, *command]
    inner_cmd_str = " ".join(quote(token) for token in conda_command)
    with open(command_path, "wt") as cmd_f:
        cmd_f.write(inner_cmd_str)

    wrapped_command = [
        "bash",
        "-c",
        "-o",
        "pipefail",
        """
chmod +x {command_path}
. {command_path} 2> >(tee {command_error_path} >&2) | tee {command_output_path}
""".format(
            command_path=quote(command_path),
            command_output_path=quote(command_output_path),
            command_error_path=quote(command_error_path),
        ),
    ]
    return wrapped_command


def handle_oneshot_output(
    output_path: str, error_path: str, command_error_path: str
) -> Tuple[bool, Any]:
    """
    Handle output of a oneshot command.
    """
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            result = pickle.load(f)
        return True, result
    elif os.path.exists(error_path):
        with open(error_path, "rb") as f:
            error, error_traceback = pickle.load(f)
        return False, (error, error_traceback)
    else:
        # Cover the case where the oneshot command was not entered or it failed to start.
        # For conda this could happen if the specified environment name doesn't exist.
        with open(command_error_path, "rb") as f:
            error = ScriptError(f.read())
            error_traceback = Traceback.from_error(error)
        return False, (error, error_traceback)


def handle_script_output(
    return_code: int, command_output_path: str, command_error_path: str
) -> Tuple[bool, Any]:
    """
    Handle output of a script command.
    """
    if return_code == 0:
        with open(command_output_path, "rb") as f:
            result = f.read()
        return True, result
    else:
        with open(command_error_path, "rb") as f:
            error = ScriptError(f.read())
            error_traceback = Traceback.from_error(error)
        return False, (error, error_traceback)
