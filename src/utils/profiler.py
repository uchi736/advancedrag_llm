"""
用語抽出パイプライン プロファイリングユーティリティ

各フェーズの実行時間、API呼び出し回数、処理データ量を計測する。
"""

import time
import functools
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhaseStats:
    """各フェーズの統計情報"""
    name: str
    duration: float = 0.0  # 秒
    sudachi_calls: int = 0
    llm_calls: int = 0
    data_count: int = 0  # 処理データ数（候補数など）
    details: str = ""


class Profiler:
    """グローバルプロファイラー"""

    def __init__(self):
        self.phases: List[PhaseStats] = []
        self.current_phase: Optional[PhaseStats] = None
        self.sudachi_call_count: int = 0
        self.llm_call_count: int = 0
        self._phase_stack: List[PhaseStats] = []

    def start_phase(self, name: str, data_count: int = 0, details: str = "") -> PhaseStats:
        """フェーズ開始"""
        phase = PhaseStats(name=name, data_count=data_count, details=details)
        self._phase_stack.append(phase)
        self.current_phase = phase
        logger.info(f"▶ {name} 開始 {f'({details})' if details else ''}")
        return phase

    def end_phase(self, duration: float):
        """フェーズ終了"""
        if not self._phase_stack:
            logger.warning("end_phase called but no phase is active")
            return

        phase = self._phase_stack.pop()
        phase.duration = duration
        self.phases.append(phase)

        self.current_phase = self._phase_stack[-1] if self._phase_stack else None
        logger.info(f"✓ {phase.name} 完了 ({duration:.2f}秒)")

    def count_sudachi_call(self):
        """Sudachi呼び出しをカウント"""
        self.sudachi_call_count += 1
        if self.current_phase:
            self.current_phase.sudachi_calls += 1

    def count_llm_call(self):
        """LLM呼び出しをカウント"""
        self.llm_call_count += 1
        if self.current_phase:
            self.current_phase.llm_calls += 1

    def get_summary(self) -> str:
        """サマリーレポートを生成"""
        if not self.phases:
            return "No profiling data"

        total_time = sum(p.duration for p in self.phases)

        lines = []
        lines.append("\n" + "="*80)
        lines.append("用語抽出パイプライン プロファイリング結果")
        lines.append("="*80)
        lines.append("")

        # フェーズ別詳細
        lines.append(f"{'Phase':<40} {'時間':>10} {'割合':>8} {'Sudachi':>9} {'LLM':>6} {'詳細':<20}")
        lines.append("-"*80)

        for phase in self.phases:
            percentage = (phase.duration / total_time * 100) if total_time > 0 else 0
            sudachi_str = f"{phase.sudachi_calls:,}" if phase.sudachi_calls > 0 else "-"
            llm_str = f"{phase.llm_calls}" if phase.llm_calls > 0 else "-"

            lines.append(
                f"{phase.name:<40} {phase.duration:>9.2f}s {percentage:>7.1f}% "
                f"{sudachi_str:>9} {llm_str:>6} {phase.details:<20}"
            )

        lines.append("-"*80)
        lines.append(f"{'合計':<40} {total_time:>9.2f}s {100.0:>7.1f}%")
        lines.append("")

        # 統計サマリー
        lines.append("統計サマリー:")
        lines.append(f"  総実行時間: {total_time:.2f}秒 ({total_time/60:.1f}分)")
        lines.append(f"  Sudachi呼び出し総数: {self.sudachi_call_count:,}回")
        lines.append(f"  LLM API呼び出し総数: {self.llm_call_count}回")

        # ボトルネック上位3つ
        lines.append("")
        lines.append("ボトルネックTop3:")
        sorted_phases = sorted(self.phases, key=lambda p: p.duration, reverse=True)[:3]
        for i, phase in enumerate(sorted_phases, 1):
            percentage = (phase.duration / total_time * 100) if total_time > 0 else 0
            lines.append(f"  {i}. {phase.name}: {phase.duration:.2f}秒 ({percentage:.1f}%)")

        lines.append("="*80)
        lines.append("")

        return "\n".join(lines)

    def reset(self):
        """プロファイラーをリセット"""
        self.phases = []
        self.current_phase = None
        self.sudachi_call_count = 0
        self.llm_call_count = 0
        self._phase_stack = []


# グローバルプロファイラーインスタンス
_global_profiler = Profiler()


def get_profiler() -> Profiler:
    """グローバルプロファイラーを取得"""
    return _global_profiler


@contextmanager
def timer(phase_name: str, data_count: int = 0, details: str = ""):
    """
    フェーズの実行時間を計測するコンテキストマネージャ

    使用例:
        with timer("Phase 1: 候補抽出", data_count=1850):
            # 処理
            pass
    """
    profiler = get_profiler()
    profiler.start_phase(phase_name, data_count, details)

    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        profiler.end_phase(duration)


def profile_function(phase_name: str):
    """
    関数の実行時間を計測するデコレータ

    使用例:
        @profile_function("Phase 2: TF-IDF計算")
        def calculate_tfidf(...):
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer(phase_name):
                return func(*args, **kwargs)
        return wrapper

    # async関数用
    def async_decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with timer(phase_name):
                return await func(*args, **kwargs)
        return async_wrapper

    # 関数がasyncかどうかで分岐
    def smart_decorator(func):
        if asyncio.iscoroutinefunction(func):
            return async_decorator(func)
        else:
            return decorator(func)

    return smart_decorator


# asyncio import for async decorator
import asyncio
