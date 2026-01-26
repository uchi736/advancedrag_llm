"""
文書チャンクの type メタデータを NULL → 'document' に更新するスクリプト

使用方法:
    python scripts/fix_document_type.py
"""
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def main():
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'postgres')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD')

    if not all([db_host, db_user, db_password]):
        print("Error: DB_HOST, DB_USER, DB_PASSWORD must be set in .env")
        sys.exit(1)

    conn_str = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(conn_str)

    with engine.connect() as conn:
        # 更新前の状態を確認
        result = conn.execute(text('''
            SELECT
                cmetadata->>'type' as doc_type,
                COUNT(*) as count
            FROM langchain_pg_embedding
            GROUP BY cmetadata->>'type'
        '''))
        print('=== 更新前 ===')
        for row in result:
            print(f"  type={repr(row[0])}: {row[1]}件")

        # NULL → 'document' に更新
        result = conn.execute(text('''
            UPDATE langchain_pg_embedding
            SET cmetadata = jsonb_set(
                COALESCE(cmetadata, '{}'::jsonb),
                '{type}',
                '"document"'
            )
            WHERE cmetadata->>'type' IS NULL
        '''))
        conn.commit()
        print(f'\n更新件数: {result.rowcount}件')

        # 更新後の状態を確認
        result = conn.execute(text('''
            SELECT
                cmetadata->>'type' as doc_type,
                COUNT(*) as count
            FROM langchain_pg_embedding
            GROUP BY cmetadata->>'type'
        '''))
        print('\n=== 更新後 ===')
        for row in result:
            print(f"  type={repr(row[0])}: {row[1]}件")

    print('\n完了!')


if __name__ == '__main__':
    main()
