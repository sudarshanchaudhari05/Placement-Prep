"""
CoppeliaSim ZeroMQ Remote API Client Abstraction
================================================
Encapsulates communication with CoppeliaSim via the modern ZeroMQ Remote API:
- Library: coppeliasim_zmqremoteapi_client (or zmqRemoteApi)
- Default Port: 23000 (ZeroMQ REQ/REP)
- Manages connection lifecycle, simulation stepping, scene loading, and object handles.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("AGVPathTracking.CoppeliaSim")


class CoppeliaSimClient:
    """
    Client interface for CoppeliaSim ZeroMQ Remote API.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 23000,
        timeout_s: float = 2.0,
    ):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

        self._client = None
        self._sim = None
        self._is_connected = False
        self._api_package_installed = False

        self._check_api_installation()

    def _check_api_installation(self) -> bool:
        """Check if coppeliasim_zmqremoteapi_client is installed."""
        try:
            import coppeliasim_zmqremoteapi_client
            self._api_package_installed = True
            return True
        except ImportError:
            self._api_package_installed = False
            return False

    @property
    def is_api_installed(self) -> bool:
        return self._api_package_installed

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> bool:
        """
        Attempt to connect to running CoppeliaSim ZeroMQ server.
        """
        if not self._api_package_installed:
            logger.warning(
                "CoppeliaSim ZeroMQ Remote API client is NOT installed. "
                "Install via: pip install coppeliasim-zmqremoteapi-client"
            )
            self._is_connected = False
            return False

        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            self._client = RemoteAPIClient(host=self.host, port=self.port)
            self._sim = self._client.require("sim")
            # Ping by getting simulation state
            _ = self._sim.getSimulationState()
            self._is_connected = True
            logger.info(f"Connected to CoppeliaSim at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to CoppeliaSim server at {self.host}:{self.port} ({e})")
            self._is_connected = False
            self._client = None
            self._sim = None
            return False

    def disconnect(self) -> None:
        """Disconnect and release handles."""
        self._is_connected = False
        self._sim = None
        self._client = None

    def start_simulation(self) -> bool:
        if not self._is_connected or self._sim is None:
            return False
        try:
            self._sim.startSimulation()
            return True
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            return False

    def stop_simulation(self) -> bool:
        if not self._is_connected or self._sim is None:
            return False
        try:
            self._sim.stopSimulation()
            return True
        except Exception as e:
            logger.error(f"Failed to stop simulation: {e}")
            return False

    def step(self) -> None:
        if self._is_connected and self._client is not None:
            try:
                self._client.step()
            except Exception as e:
                logger.error(f"Simulation step failed: {e}")

    def get_sim_handle(self):
        """Direct handle to CoppeliaSim 'sim' namespace."""
        return self._sim
