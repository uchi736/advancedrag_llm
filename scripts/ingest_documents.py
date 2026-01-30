"""
ドキュメントインジェストCLIスクリプト

Usage:
    # PDFファイルを指定
    python scripts/ingest_documents.py path/to/file.pdf [file2.pdf ...]

    # ディレクトリ内のPDFを一括処理
    python scripts/ingest_documents.py --dir path/to/pdf_dir

    # コレクション名を指定
    python scripts/ingest_documents.py --collection my_collection path/to/file.pdf
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description="ドキュメントインジェストCLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "files", nargs="*",
        help="インジェスト対象のPDFファイルパス"
    )
    parser.add_argument(
        "--dir", "-d",
        help="PDFファイルを含むディレクトリ（再帰検索）"
    )
    parser.add_argument(
        "--collection", "-c",
        default="documents",
        help="コレクション名 (default: documents)"
    )

    args = parser.parse_args()

    # ファイルリスト構築
    paths = list(args.files) if args.files else []

    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: ディレクトリが見つかりません: {args.dir}")
            sys.exit(1)
        pdf_files = sorted(dir_path.rglob("*.pdf"))
        paths.extend([str(p) for p in pdf_files])

    if not paths:
        print("Error: ファイルまたは --dir を指定してください")
        parser.print_help()
        sys.exit(1)

    # ファイル存在チェック
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        for m in missing:
            print(f"Error: ファイルが見つかりません: {m}")
        sys.exit(1)

    print("=" * 60)
    print("ドキュメントインジェスト CLI")
    print("=" * 60)
    print(f"  コレクション: {args.collection}")
    print(f"  ファイル数: {len(paths)}")
    for p in paths:
        print(f"    - {p}")
    print("=" * 60)

    # RAGシステム初期化
    print("\nRAGシステムを初期化中...")
    from src.rag.config import Config
    from src.core.rag_system import RAGSystem

    config = Config()
    config.collection_name = args.collection

    rag_system = RAGSystem(config)
    print(f"  LLMプロバイダー: {config.llm_provider or 'azure'}")

    # インジェスト実行
    start_time = datetime.now()
    print(f"\nインジェスト開始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        rag_system.ingest_documents(paths)
    except Exception as e:
        print(f"\n\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"完了!")
    print(f"  処理ファイル数: {len(paths)}")
    print(f"  処理時間: {elapsed:.1f}秒 ({elapsed / 60:.1f}分)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
