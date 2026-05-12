#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky - Installer Update Dialogs

下载安装包 + 触发安装这一对 dialogs：
- InstallerProgressDialog: 进度条 + 取消 + 错误处理
- InstallerReadyDialog: 下载完成后的安装确认
"""
from __future__ import annotations

import time
import webbrowser
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from tools.installer_updater import (
    InstallerAsset,
    InstallerUpdateError,
    download_installer,
)
from ui.styles import COLORS


class _DownloadWorker(QObject):
    """后台下载 worker，运行在 QThread 中。"""

    progress = Signal(int, int)  # downloaded, total (bytes)
    finished = Signal(object)  # path: Path
    failed = Signal(str)  # error message

    def __init__(self, asset: InstallerAsset, dest_dir: Optional[Path] = None):
        super().__init__()
        self._asset = asset
        self._dest_dir = dest_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            path = download_installer(
                self._asset,
                dest_dir=self._dest_dir,
                on_progress=lambda d, t: self.progress.emit(d, t),
                should_cancel=lambda: self._cancelled,
            )
            self.finished.emit(path)
        except InstallerUpdateError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class InstallerProgressDialog(QDialog):
    """安装包下载进度对话框。

    使用模式:
        dlg = InstallerProgressDialog(asset, i18n, parent=...)
        dlg.start()
        code = dlg.exec()  # noqa: Qt API, not Python exec
        if code == QDialog.Accepted:
            path = dlg.installer_path
        else:
            err = dlg.error_message  # None 表示用户取消
    """

    def __init__(
        self,
        asset: InstallerAsset,
        i18n,
        parent=None,
        dest_dir: Optional[Path] = None,
    ):
        super().__init__(parent)
        self._asset = asset
        self._dest_dir = dest_dir
        self.i18n = i18n
        self.installer_path: Optional[Path] = None
        self.error_message: Optional[str] = None
        self._start_time = 0.0
        self._worker: Optional[_DownloadWorker] = None
        self._thread: Optional[QThread] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle(self.i18n.t("update.downloading_title"))
        self.setMinimumWidth(440)
        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {COLORS['bg_primary']}; }}
            QLabel  {{ color: {COLORS['text_primary']}; font-size: 13px; }}
            QProgressBar {{
                background-color: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                border-radius: 3px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(10)

        title = QLabel(self.i18n.t("update.downloading_title"))
        title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 16px; font-weight: 600;"
        )
        layout.addWidget(title)

        name_label = QLabel(self._asset.name)
        name_label.setStyleSheet(
            f"color: {COLORS['text_tertiary']}; font-size: 12px;"
        )
        layout.addWidget(name_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("0.0 MB / 0.0 MB (0%)")
        self.status_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        layout.addWidget(self.status_label)

        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet(
            f"color: {COLORS['text_tertiary']}; font-size: 11px;"
        )
        layout.addWidget(self.eta_label)

        layout.addSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton(self.i18n.t("update.cancel"))
        self.cancel_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                color: {COLORS['text_secondary']};
                border-radius: 6px;
                padding: 8px 24px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                border-color: {COLORS['text_muted']};
                color: {COLORS['text_primary']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_muted']};
            }}
            """
        )
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

    def start(self) -> None:
        """启动后台下载线程。"""
        self._start_time = time.monotonic()
        self._thread = QThread(self)
        self._worker = _DownloadWorker(self._asset, self._dest_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, downloaded: int, total: int) -> None:
        if total <= 0:
            return
        ratio = downloaded / total
        self.progress_bar.setValue(int(ratio * 1000))
        done_mb = downloaded / 1048576
        total_mb = total / 1048576
        self.status_label.setText(
            self.i18n.t("update.download_progress_template").format(
                done_mb=done_mb, total_mb=total_mb, pct=int(ratio * 100)
            )
        )
        elapsed = time.monotonic() - self._start_time
        if elapsed > 0.2 and downloaded > 0:
            rate_mb = done_mb / elapsed
            eta_s = max(0, int((total_mb - done_mb) / rate_mb)) if rate_mb > 0 else 0
            self.eta_label.setText(
                self.i18n.t("update.download_speed_eta_template").format(
                    rate_mb=rate_mb, eta_seconds=eta_s
                )
            )

    def _on_finished(self, path: Path) -> None:
        self.installer_path = path
        self.accept()

    def _on_failed(self, error: str) -> None:
        if "用户取消" in error or "cancel" in error.lower():
            self.error_message = None
        else:
            self.error_message = error
        self.reject()

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText(self.i18n.t("update.cancelling"))

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            if self._worker:
                self._worker.cancel()
            self._thread.wait(3000)
        super().closeEvent(event)


class InstallerReadyDialog(QDialog):
    """下载完成后的安装确认对话框。"""

    def __init__(
        self,
        version: str,
        installer_path: Path,
        i18n,
        parent=None,
    ):
        super().__init__(parent)
        self.installer_path = installer_path
        self.i18n = i18n
        self.should_install = False
        self._build_ui(version)

    def _build_ui(self, version: str) -> None:
        self.setWindowTitle(self.i18n.t("update.ready_title"))
        self.setMinimumWidth(440)
        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {COLORS['bg_primary']}; }}
            QLabel  {{ color: {COLORS['text_primary']}; font-size: 13px; }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(10)

        title = QLabel(self.i18n.t("update.ready_title"))
        title.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 16px; font-weight: 600;"
        )
        layout.addWidget(title)

        msg = QLabel(
            self.i18n.t("update.ready_message_template").format(version=version)
        )
        msg.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        path_label_row = QHBoxLayout()
        path_label = QLabel(self.i18n.t("update.ready_path_label"))
        path_label.setStyleSheet(
            f"color: {COLORS['text_tertiary']}; font-size: 11px;"
        )
        path_label_row.addWidget(path_label)
        path_value = QLabel(str(self.installer_path))
        path_value.setStyleSheet(
            f"color: {COLORS['text_tertiary']}; font-size: 11px;"
        )
        path_value.setWordWrap(True)
        path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path_label_row.addWidget(path_value, stretch=1)
        layout.addLayout(path_label_row)

        layout.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        later_btn = QPushButton(self.i18n.t("update.install_later"))
        later_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                color: {COLORS['text_secondary']};
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                border-color: {COLORS['text_muted']};
                color: {COLORS['text_primary']};
            }}
            """
        )
        later_btn.clicked.connect(self.reject)
        btn_row.addWidget(later_btn)
        btn_row.addSpacing(8)

        install_btn = QPushButton(self.i18n.t("update.install_now"))
        install_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_void']};
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}
            """
        )
        install_btn.clicked.connect(self._on_install)
        btn_row.addWidget(install_btn)

        layout.addLayout(btn_row)

    def _on_install(self) -> None:
        self.should_install = True
        self.accept()


def run_installer_update_flow(
    asset: InstallerAsset,
    version: str,
    i18n,
    parent=None,
    fallback_browser_url: Optional[str] = None,
) -> bool:
    """驱动完整的"下载 → 确认 → 启动安装包"流程。

    Returns:
        True 表示用户最终点了"立即安装"，调用方应当 quit() 主程序。
        False 表示用户取消、稍后或失败。
    """
    from tools.installer_updater import InstallerUpdateError, trigger_install
    from ui.custom_dialogs import StyledMessageBox

    progress_dlg = InstallerProgressDialog(asset, i18n, parent=parent)
    progress_dlg.start()
    if progress_dlg.exec() != QDialog.DialogCode.Accepted:
        if progress_dlg.error_message:
            StyledMessageBox.warning(
                parent,
                i18n.t("update.download_failed_title"),
                progress_dlg.error_message,
            )
            if fallback_browser_url:
                webbrowser.open(fallback_browser_url)
        return False

    installer_path = progress_dlg.installer_path
    if installer_path is None:
        return False

    ready_dlg = InstallerReadyDialog(version, installer_path, i18n, parent=parent)
    if ready_dlg.exec() != QDialog.DialogCode.Accepted or not ready_dlg.should_install:
        return False

    try:
        trigger_install(installer_path)
    except InstallerUpdateError as e:
        StyledMessageBox.warning(
            parent,
            i18n.t("update.download_failed_title"),
            str(e),
        )
        return False

    return True
