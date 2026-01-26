"""
専門用語抽出CLIスクリプト

使用方法:
    # 基本（document_chunksから抽出）
    python scripts/extract_terms.py --collection documents --output output/terms.json

    # PDFから抽出
    python scripts/extract_terms.py --input path/to/pdfs --output output/terms.json

    # Stage 2.5を無効化
    python scripts/extract_terms.py --collection documents --output output/terms.json --no-stage25

    # ドメイン分類方法を指定
    python scripts/extract_terms.py --collection documents --output output/terms.json --domain-method llm
"""
import os
import sys
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def create_cli_callback():
    """CLI用の進捗コールバックを作成"""
    current_stage = [None]

    def callback(event_type: str, data: dict):
        stage = data.get("stage", "")

        if event_type == "stage_start":
            current_stage[0] = stage
            print(f"\n{'='*60}")
            print(f"[{stage}] 開始...")
            print(f"{'='*60}")

        elif event_type == "stage_complete":
            count = data.get("count", 0)
            print(f"[{stage}] 完了 - {count}件")

        elif event_type == "progress":
            current = data.get("current", 0)
            total = data.get("total", 0)
            message = data.get("message", "")
            if total > 0:
                pct = current / total * 100
                bar_len = 30
                filled = int(bar_len * current / total)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"\r  [{bar}] {pct:5.1f}% ({current}/{total}) {message}", end="", flush=True)
            else:
                print(f"\r  {message}", end="", flush=True)

        elif event_type == "stage25_iteration":
            iteration = data.get("iteration", 0)
            confidence = data.get("confidence", 0)
            added = data.get("added", 0)
            removed = data.get("removed", 0)
            print(f"\n  [Iteration {iteration}] confidence={confidence:.2f}, +{added}/-{removed}")

        elif event_type == "info":
            message = data.get("message", "")
            print(f"\n  INFO: {message}")

        elif event_type == "warning":
            message = data.get("message", "")
            print(f"\n  WARNING: {message}")

        elif event_type == "error":
            message = data.get("message", "")
            print(f"\n  ERROR: {message}")

    return callback


async def main():
    parser = argparse.ArgumentParser(
        description="専門用語抽出CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--collection", "-c",
        default="documents",
        help="対象コレクション名 (default: documents)"
    )
    parser.add_argument(
        "--input", "-i",
        help="入力ディレクトリ（PDFファイル）。指定しない場合はDBから取得"
    )
    parser.add_argument(
        "--output", "-o",
        default=f"output/terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="出力JSONファイルパス"
    )
    parser.add_argument(
        "--no-stage25",
        action="store_true",
        help="Stage 2.5 (Self-Reflection) を無効化"
    )
    parser.add_argument(
        "--domain-method",
        choices=["embedding", "llm"],
        default="embedding",
        help="Stage 4 ドメイン分類方法 (default: embedding)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Stage 2.5 最大反復回数 (default: 3)"
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("専門用語抽出 CLI")
    print("="*60)
    print(f"  コレクション: {args.collection}")
    print(f"  入力: {args.input or '(DBから取得)'}")
    print(f"  出力: {args.output}")
    print(f"  Stage 2.5: {'無効' if args.no_stage25 else '有効'}")
    print(f"  ドメイン分類: {args.domain_method}")
    print("="*60)

    # RAGシステム初期化
    print("\nRAGシステムを初期化中...")
    from src.rag.config import Config
    from src.core.rag_system import RAGSystem

    config = Config()
    config.collection_name = args.collection
    config.enable_stage25_refinement = not args.no_stage25
    config.stage4_domain_method = args.domain_method
    config.max_refinement_iterations = args.max_iterations

    rag_system = RAGSystem(config)
    print(f"  LLMプロバイダー: {config.llm_provider or 'azure'}")

    # 入力パスの決定
    if args.input:
        input_path = args.input
    else:
        # DBから取得する場合は一時ディレクトリを使用
        import tempfile
        input_path = tempfile.mkdtemp(prefix="term_extract_")
        print(f"\n  一時ディレクトリ: {input_path}")

    # コールバック作成
    callback = create_cli_callback()

    # 抽出実行
    start_time = datetime.now()
    print(f"\n抽出開始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        await rag_system.extract_terms(input_path, str(output_path), ui_callback=callback)
    except Exception as e:
        print(f"\n\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n\n{'='*60}")
    print(f"完了!")
    print(f"  出力ファイル: {output_path}")
    print(f"  処理時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
