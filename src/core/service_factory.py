"""
Service Factory for Dependency Injection

Provides centralized service creation and dependency management.
Implements the Factory and Dependency Injection patterns.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from enum import Enum

from ..config.app_config import get_config, ApplicationConfig
from ..domain.interfaces.tracking_interfaces import ITracker, ITrackingService
from ..domain.interfaces.calibration_interfaces import ICalibrationService
from ..domain.interfaces.diameter_measurement_interfaces import (
    IDiameterMeasurement,
    DiameterMethod,
    DiameterMeasurementConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceScope(Enum):
    """Service lifetime scopes"""

    SINGLETON = "singleton"  # Single instance for entire application
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # Single instance per scope/request


class ServiceRegistry:
    """Registry for service implementations"""

    def __init__(self):
        self._services: Dict[Type, Dict[str, Any]] = {}
        self._instances: Dict[Type, Any] = {}  # For singletons

    def register(
        self,
        interface: Type[T],
        implementation: Type[T] = None,
        factory: Callable[[], T] = None,
        scope: ServiceScope = ServiceScope.SINGLETON,
        name: str = "default",
    ) -> None:
        """
        Register a service implementation.

        Args:
            interface: Interface/protocol type
            implementation: Concrete implementation class
            factory: Factory function to create instances
            scope: Service lifetime scope
            name: Registration name (for multiple implementations)
        """
        if interface not in self._services:
            self._services[interface] = {}

        self._services[interface][name] = {
            "implementation": implementation,
            "factory": factory,
            "scope": scope,
        }

        logger.debug(f"Registered {implementation or factory} for {interface} as '{name}'")

    def resolve(self, interface: Type[T], name: str = "default") -> Optional[T]:
        """
        Resolve a service instance.

        Args:
            interface: Interface type to resolve
            name: Registration name

        Returns:
            Service instance or None
        """
        if interface not in self._services:
            logger.error(f"No registration found for {interface}")
            return None

        if name not in self._services[interface]:
            logger.error(f"No registration found for {interface} with name '{name}'")
            return None

        registration = self._services[interface][name]
        scope = registration["scope"]

        # Handle singleton scope
        if scope == ServiceScope.SINGLETON:
            cache_key = (interface, name)
            if cache_key in self._instances:
                return self._instances[cache_key]

            instance = self._create_instance(registration)
            if instance:
                self._instances[cache_key] = instance
            return instance

        # Handle transient scope
        elif scope == ServiceScope.TRANSIENT:
            return self._create_instance(registration)

        # Handle scoped (not implemented yet)
        else:
            return self._create_instance(registration)

    def _create_instance(self, registration: Dict[str, Any]) -> Optional[Any]:
        """Create a service instance from registration"""
        try:
            if registration["factory"]:
                return registration["factory"]()
            elif registration["implementation"]:
                return registration["implementation"]()
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to create service instance: {e}")
            return None

    def get_all(self, interface: Type[T]) -> Dict[str, T]:
        """Get all registered implementations for an interface"""
        if interface not in self._services:
            return {}

        result = {}
        for name in self._services[interface]:
            instance = self.resolve(interface, name)
            if instance:
                result[name] = instance

        return result


class ServiceFactory:
    """
    Main service factory for the application.

    Provides centralized service creation with dependency injection.
    """

    _instance: Optional["ServiceFactory"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config = get_config()
        self._registry = ServiceRegistry()
        self._initialize_services()
        self._initialized = True

    def _initialize_services(self):
        """Initialize and register all services"""

        # Register tracking services
        self._register_tracking_services()

        # Register calibration services
        self._register_calibration_services()

        # Register diameter measurement services
        self._register_diameter_services()

        # Register QCA services
        self._register_qca_services()

        # Register RWS services
        self._register_rws_services()

    def _register_tracking_services(self):
        """Register tracking service implementations"""
        try:
            # Import implementations
            from ..services.tracking.tracking_service import TrackingService
            from ..core.simple_tracker import SimpleTracker

            # Register main tracking service
            self._registry.register(
                ITrackingService, implementation=TrackingService, scope=ServiceScope.SINGLETON
            )

            # Register tracker implementations
            self._registry.register(
                ITracker, implementation=SimpleTracker, scope=ServiceScope.TRANSIENT, name="simple"
            )

            logger.info("Tracking services registered")

        except ImportError as e:
            logger.warning(f"Could not register tracking services: {e}")

    def _register_calibration_services(self):
        """Register calibration service implementations"""
        try:
            from ..services.calibration.calibration_service import CalibrationService
            from ..services.calibration.measurement_strategies import (
                CatheterSegmentation,
                CatheterMeasurement,
            )

            # Register calibration service with dependencies
            def calibration_factory():
                segmentation = CatheterSegmentation()
                measurement = CatheterMeasurement()
                return CalibrationService(segmentation, measurement)

            self._registry.register(
                ICalibrationService, factory=calibration_factory, scope=ServiceScope.SINGLETON
            )

            logger.info("Calibration services registered")

        except ImportError as e:
            logger.warning(f"Could not register calibration services: {e}")

    def _register_diameter_services(self):
        """Register diameter measurement implementations"""
        try:
            from ..analysis.vessel_diameter_accurate import measure_vessel_diameter_accurate
            from ..analysis.gradient_diameter_measurement import measure_gradient_diameter
            from ..domain.interfaces.diameter_measurement_interfaces import (
                DiameterMeasurementAdapter,
            )

            # Register gradient-based measurement
            gradient_adapter = DiameterMeasurementAdapter(
                measure_gradient_diameter, DiameterMethod.GRADIENT, requires_segmentation=False
            )

            self._registry.register(
                IDiameterMeasurement,
                factory=lambda: gradient_adapter,
                scope=ServiceScope.SINGLETON,
                name="gradient",
            )

            # Register segmentation-based measurement
            segmentation_adapter = DiameterMeasurementAdapter(
                measure_vessel_diameter_accurate,
                DiameterMethod.SEGMENTATION,
                requires_segmentation=True,
            )

            self._registry.register(
                IDiameterMeasurement,
                factory=lambda: segmentation_adapter,
                scope=ServiceScope.SINGLETON,
                name="segmentation",
            )

            logger.info("Diameter measurement services registered")

        except ImportError as e:
            logger.warning(f"Could not register diameter services: {e}")

    def _register_qca_services(self):
        """Register QCA analysis services"""
        try:
            from ..services.qca_analysis_service import QCAAnalysisService

            self._registry.register(
                QCAAnalysisService, implementation=QCAAnalysisService, scope=ServiceScope.SINGLETON
            )

            logger.info("QCA services registered")

        except ImportError as e:
            logger.warning(f"Could not register QCA services: {e}")

    def _register_rws_services(self):
        """Register RWS analysis services"""
        try:
            from ..services.rws_analysis_service import RWSAnalysisService

            self._registry.register(
                RWSAnalysisService, implementation=RWSAnalysisService, scope=ServiceScope.SINGLETON
            )

            logger.info("RWS services registered")

        except ImportError as e:
            logger.warning(f"Could not register RWS services: {e}")

    # Public methods for service resolution

    def get_tracking_service(self) -> Optional[ITrackingService]:
        """Get the tracking service instance"""
        return self._registry.resolve(ITrackingService)

    def get_tracker(self, method: str = "simple") -> Optional[ITracker]:
        """Get a tracker implementation"""
        return self._registry.resolve(ITracker, name=method)

    def get_calibration_service(self) -> Optional[ICalibrationService]:
        """Get the calibration service instance"""
        return self._registry.resolve(ICalibrationService)

    def get_diameter_measurement(
        self, method: DiameterMethod = DiameterMethod.GRADIENT
    ) -> Optional[IDiameterMeasurement]:
        """Get a diameter measurement implementation"""
        return self._registry.resolve(IDiameterMeasurement, name=method.value)

    def get_all_diameter_methods(self) -> Dict[str, IDiameterMeasurement]:
        """Get all available diameter measurement methods"""
        return self._registry.get_all(IDiameterMeasurement)

    def create_diameter_config(
        self, method: DiameterMethod = DiameterMethod.GRADIENT
    ) -> DiameterMeasurementConfig:
        """Create default configuration for a diameter method"""
        config_map = {
            DiameterMethod.GRADIENT: {
                "max_search_distance": self._config.qca.max_search_distance_pixels,
                "smoothing_window": self._config.qca.diameter_smoothing_window,
                "edge_threshold": 0.5,
                "use_subpixel": True,
            },
            DiameterMethod.SEGMENTATION: {
                "max_search_distance": self._config.qca.max_search_distance_pixels,
                "smoothing_window": 3,
                "edge_threshold": 0.5,
                "use_subpixel": True,
            },
        }

        params = config_map.get(method, config_map[DiameterMethod.GRADIENT])

        return DiameterMeasurementConfig(method=method, **params)

    @property
    def config(self) -> ApplicationConfig:
        """Get application configuration"""
        return self._config

    @property
    def registry(self) -> ServiceRegistry:
        """Get service registry for custom registrations"""
        return self._registry


# Convenience functions


def get_service_factory() -> ServiceFactory:
    """Get the global service factory instance"""
    return ServiceFactory()


def get_service(service_type: Type[T], name: str = "default") -> Optional[T]:
    """
    Convenience function to get a service.

    Args:
        service_type: Service interface type
        name: Registration name

    Returns:
        Service instance or None
    """
    factory = get_service_factory()
    return factory.registry.resolve(service_type, name)


def register_service(
    interface: Type[T],
    implementation: Type[T] = None,
    factory: Callable[[], T] = None,
    scope: ServiceScope = ServiceScope.SINGLETON,
    name: str = "default",
) -> None:
    """
    Convenience function to register a service.

    Args:
        interface: Interface type
        implementation: Implementation class
        factory: Factory function
        scope: Service scope
        name: Registration name
    """
    service_factory = get_service_factory()
    service_factory.registry.register(interface, implementation, factory, scope, name)
