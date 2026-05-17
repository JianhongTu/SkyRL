"""Custom SWE/R2E task wrapper for Docker's default runtime.

The upstream SkyRL-Agent SWEBenchTask requests the Sysbox runtime class for
OpenHands remote sandboxes. Our runtime host does not have Sysbox installed, so
this wrapper keeps the upstream task behavior but leaves the remote runtime
class unset. The OpenHands remote runtime server then uses Docker's default
runtime, which is runc on this host.
"""

from skyrl_agent.tasks.swebench.utils import SWEBenchTask


class SWEBenchDockerRuntimeTask(SWEBenchTask):
    """SWEBenchTask variant that uses Docker's default runtime instead of Sysbox."""

    @classmethod
    def get_config(cls, instance, data_source, agent_config=None, max_iterations=None):
        app_config = super().get_config(instance, data_source, agent_config, max_iterations)
        app_config.sandbox.remote_runtime_class = None
        return app_config
