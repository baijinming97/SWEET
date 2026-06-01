import os
import sys


_dll_directory_handles = []


def configure_qt_plugins():
    """Point PyQt5 at the bundled Qt plugins when running from the portable venv."""
    if not sys.platform.startswith("win"):
        return

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    qt_root = os.path.join(
        project_root,
        "python",
        "Lib",
        "site-packages",
        "PyQt5",
        "Qt5",
    )
    plugin_root = os.path.join(qt_root, "plugins")
    platform_root = os.path.join(plugin_root, "platforms")

    if not os.path.exists(os.path.join(platform_root, "qwindows.dll")):
        return

    os.environ["QT_PLUGIN_PATH"] = plugin_root
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_root

    qt_bin = os.path.join(qt_root, "bin")
    path_parts = [os.path.normcase(path) for path in os.environ.get("PATH", "").split(os.pathsep)]
    if os.path.normcase(qt_bin) not in path_parts:
        os.environ["PATH"] = qt_bin + os.pathsep + os.environ.get("PATH", "")

    if hasattr(os, "add_dll_directory"):
        _dll_directory_handles.append(os.add_dll_directory(qt_bin))
