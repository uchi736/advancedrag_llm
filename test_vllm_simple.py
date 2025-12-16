#!/usr/bin/env python3
"""
test_vllm_simple.py
===================
シンプルなVLLM接続テスト
"""

import os
import sys
import time
from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv()

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.config import Config
from src.rag.vllm_client import VLLMClient, VLLMChatClient


def test_vllm_connection():
    """VLLM接続テスト"""
    print("\n" + "=" * 60)
    print("VLLM Simple Connection Test")
    print("=" * 60)

    # 設定を読み込む
    config = Config()

    print(f"LLM_PROVIDER: {config.llm_provider}")
    print(f"VLLM_ENDPOINT: {config.vllm_endpoint}")

    if not config.vllm_endpoint:
        print("\n❌ VLLM_ENDPOINT is not configured")
        return False

    print(f"Temperature: {config.vllm_temperature}")
    print(f"Max Tokens: {config.vllm_max_tokens}")
    print(f"Timeout: {config.vllm_timeout}s")
    print("-" * 60)

    try:
        # VLLMClientを初期化
        client = VLLMClient(
            endpoint=config.vllm_endpoint,
            temperature=config.vllm_temperature,
            top_p=config.vllm_top_p,
            top_k=config.vllm_top_k,
            min_p=config.vllm_min_p,
            max_tokens=config.vllm_max_tokens,
            reasoning_effort=config.vllm_reasoning_effort,
            timeout=config.vllm_timeout
        )

        # テストプロンプト
        test_prompt = "こんにちは。1から5まで数えてください。"
        print(f"\n[Test Prompt]: {test_prompt}")
        print("[Sending request to VLLM server...]")

        # 実行時間を計測
        start_time = time.time()
        response = client.invoke(test_prompt)
        elapsed_time = time.time() - start_time

        print(f"\n✅ Success! (Response time: {elapsed_time:.2f}s)")
        if hasattr(response, "content"):
            print(f"[Response]:\n{response.content}")
        else:
            print(f"[Response]:\n{response}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vllm_chat_client():
    """VLLMChatClient のテスト"""
    print("\n" + "=" * 60)
    print("VLLMChatClient Test")
    print("=" * 60)

    config = Config()

    try:
        # VLLMChatClientを初期化
        client = VLLMChatClient(
            endpoint=config.vllm_endpoint,
            temperature=config.vllm_temperature,
            top_p=config.vllm_top_p,
            top_k=config.vllm_top_k,
            min_p=config.vllm_min_p,
            max_tokens=100,  # 短い応答で測定
            reasoning_effort=config.vllm_reasoning_effort,
            timeout=config.vllm_timeout
        )

        # テストプロンプト
        test_prompt = "Pythonとは何ですか？一文で答えてください。"
        print(f"\n[Test Prompt]: {test_prompt}")

        start_time = time.time()
        response = client.invoke(test_prompt)
        elapsed_time = time.time() - start_time

        # ChatModel互換のレスポンス形式をチェック
        if hasattr(response, 'content'):
            print(f"✅ Response has content attribute")
            print(f"[Response]: {response.content}")
            print(f"(Response time: {elapsed_time:.2f}s)")
        else:
            print(f"❌ Response doesn't have content attribute")
            print(f"[Raw Response]: {response}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting VLLM tests...")

    # 接続テスト
    result1 = test_vllm_connection()

    # ChatClientテスト
    result2 = test_vllm_chat_client()

    # 結果サマリー
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Connection Test: {'✅ PASSED' if result1 else '❌ FAILED'}")
    print(f"  ChatClient Test: {'✅ PASSED' if result2 else '❌ FAILED'}")

    if result1 and result2:
        print("\n🎉 All tests passed! VLLM is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
    print("=" * 60)
