"""
Environment-aware configuration management for all services.
All configuration values can be overridden via environment variables.
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """
    Centralized configuration class that reads from environment variables.
    All values have sensible defaults but can be overridden via ENV.
    """
    
    # ==================== Base Directories ====================
    @staticmethod
    def get_base_dir() -> Path:
        """Get the base directory (can be overridden via BASE_DIR env)"""
        base_dir = os.getenv("BASE_DIR")
        if base_dir:
            return Path(base_dir).resolve()
        return Path(__file__).resolve().parent.parent
    
    @staticmethod
    def get_models_dir() -> Path:
        """Get models directory (can be overridden via MODELS_DIR env)"""
        models_dir = os.getenv("MODELS_DIR")
        if models_dir:
            return Path(models_dir).resolve()
        return Config.get_base_dir() / "models"
    
    @staticmethod
    def get_screenshots_dir() -> Path:
        """Get screenshots directory (can be overridden via SCREENSHOTS_DIR env)"""
        screenshots_dir = os.getenv("SCREENSHOTS_DIR")
        if screenshots_dir:
            return Path(screenshots_dir).resolve()
        return Config.get_base_dir() / "screenshots"
    
    # ==================== Camera Settings ====================
    @staticmethod
    def get_default_fps() -> int:
        """Default FPS for camera streams"""
        return int(os.getenv("DEFAULT_FPS", "5"))
    
    @staticmethod
    def get_max_fps() -> int:
        """Maximum allowed FPS"""
        return int(os.getenv("MAX_FPS", "30"))
    
    @staticmethod
    def get_min_fps() -> int:
        """Minimum allowed FPS"""
        return int(os.getenv("MIN_FPS", "1"))
    
    @staticmethod
    def get_max_cameras_single_mode() -> int:
        """Maximum cameras in single-stream mode"""
        return int(os.getenv("MAX_CAMERAS_SINGLE_MODE", "50"))
    
    @staticmethod
    def get_default_camera_timeout() -> int:
        """Default camera connection timeout in seconds"""
        return int(os.getenv("DEFAULT_CAMERA_TIMEOUT", "30"))
    
    # ==================== Detection Settings ====================
    @staticmethod
    def get_default_confidence_threshold() -> float:
        """Default confidence threshold for object detection"""
        return float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.5"))
    
    @staticmethod
    def get_default_iou_threshold() -> float:
        """Default IOU threshold for NMS"""
        return float(os.getenv("DEFAULT_IOU_THRESHOLD", "0.45"))
    
    @staticmethod
    def get_yolo_model_path() -> str:
        """Path to YOLO model file"""
        default_path = str(Config.get_models_dir() / "yolo11n.pt")
        return os.getenv("YOLO_MODEL_PATH", default_path)
    
    # ==================== Device Settings ====================
    @staticmethod
    def use_gpu() -> bool:
        """Whether to use GPU for inference"""
        return os.getenv("USE_GPU", "true").lower() in ("true", "1", "yes")
    
    # ==================== API Settings ====================
    @staticmethod
    def get_api_host() -> str:
        """API host address"""
        return os.getenv("API_HOST", "0.0.0.0")
    
    @staticmethod
    def get_api_port() -> int:
        """API port number"""
        return int(os.getenv("API_PORT", "8000"))
    
    @staticmethod
    def get_api_reload() -> bool:
        """Whether to enable API auto-reload (development only)"""
        return os.getenv("API_RELOAD", "false").lower() in ("true", "1", "yes")
    
    # ==================== Logging Settings ====================
    @staticmethod
    def get_log_level() -> str:
        """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @staticmethod
    def get_log_format() -> str:
        """Log message format"""
        return os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    @staticmethod
    def get_log_file() -> str:
        """Log file path"""
        return os.getenv("LOG_FILE", "app.log")
    
    # ==================== ROI Settings ====================
    @staticmethod
    def get_roi_color() -> tuple:
        """ROI color in BGR format (can be overridden as comma-separated: R,G,B)"""
        color_str = os.getenv("ROI_COLOR", "0,255,255")  # Yellow
        try:
            return tuple(map(int, color_str.split(",")))
        except:
            return (0, 255, 255)
    
    @staticmethod
    def get_roi_thickness() -> int:
        """ROI line thickness"""
        return int(os.getenv("ROI_THICKNESS", "2"))
    
    @staticmethod
    def get_roi_fill_alpha() -> float:
        """ROI fill transparency (0.0 to 1.0)"""
        return float(os.getenv("ROI_FILL_ALPHA", "0.3"))
    
    # ==================== Frame Processing ====================
    @staticmethod
    def get_max_frame_width() -> int:
        """Maximum frame width for processing"""
        return int(os.getenv("MAX_FRAME_WIDTH", "1920"))
    
    @staticmethod
    def get_max_frame_height() -> int:
        """Maximum frame height for processing"""
        return int(os.getenv("MAX_FRAME_HEIGHT", "1080"))
    
    @staticmethod
    def get_jpeg_quality() -> int:
        """JPEG compression quality (0-100)"""
        return int(os.getenv("JPEG_QUALITY", "85"))
    
    # ==================== Multi-Stream Settings (DeepStream) ====================
    @staticmethod
    def get_multi_stream_batch_size() -> int:
        """Batch size for multi-stream processing"""
        return int(os.getenv("MULTI_STREAM_BATCH_SIZE", "4"))
    
    @staticmethod
    def get_multi_stream_width() -> int:
        """Width for multi-stream processing"""
        return int(os.getenv("MULTI_STREAM_WIDTH", "1280"))
    
    @staticmethod
    def get_multi_stream_height() -> int:
        """Height for multi-stream processing"""
        return int(os.getenv("MULTI_STREAM_HEIGHT", "720"))
    
    @staticmethod
    def get_deepstream_path() -> Optional[str]:
        """Path to DeepStream installation (only needed for camera-detection service)"""
        return os.getenv("DEEPSTREAM_PATH")
    
    # ==================== Database Settings ====================
    @staticmethod
    def get_database_url() -> str:
        """
        Database connection URL.
        Format: postgresql://user:password@host:port/database
        """
        return os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/goec"
        )
    
    # ==================== Service URLs (for orchestration) ====================
    @staticmethod
    def get_camera_detection_url() -> str:
        """URL of camera-detection service"""
        return os.getenv("CAMERA_DETECTION_URL", "http://localhost:8000")
    
    @staticmethod
    def get_usecase_service_url() -> str:
        """URL of usecase service"""
        return os.getenv("USECASE_SERVICE_URL", "http://localhost:8001")
    
    @staticmethod
    def get_alert_service_url() -> str:
        """URL of alert service"""
        return os.getenv("ALERT_SERVICE_URL", "http://localhost:8002")
    
    @staticmethod
    def get_analytics_service_url() -> str:
        """URL of analytics service"""
        return os.getenv("ANALYTICS_SERVICE_URL", "http://localhost:8003")
    
    # ==================== Utility Methods ====================
    @classmethod
    def print_config(cls):
        """Print all configuration values (useful for debugging)"""
        print("=" * 60)
        print("Configuration Settings:")
        print("=" * 60)
        print(f"Base Directory: {cls.get_base_dir()}")
        print(f"Models Directory: {cls.get_models_dir()}")
        print(f"Screenshots Directory: {cls.get_screenshots_dir()}")
        print(f"API: {cls.get_api_host()}:{cls.get_api_port()}")
        print(f"Log Level: {cls.get_log_level()}")
        print(f"Database URL: {cls.get_database_url()}")
        print(f"Use GPU: {cls.use_gpu()}")
        print("=" * 60)
    
    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist and create them if needed"""
        dirs_to_check = [
            cls.get_models_dir(),
            cls.get_screenshots_dir(),
        ]
        
        for directory in dirs_to_check:
            directory.mkdir(parents=True, exist_ok=True)
        
        return True


# For backward compatibility, you can also use these as module-level variables
# But it's recommended to use Config.get_*() methods for better testability

# Legacy support (will be removed in future versions)
LOG_LEVEL = Config.get_log_level()
LOG_FORMAT = Config.get_log_format()
LOG_FILE = Config.get_log_file()
