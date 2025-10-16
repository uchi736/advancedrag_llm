STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        /* ダークテーマカラー (Dark theme colors) */
        --bg-primary: #0a0a0a; --bg-secondary: #141414; --bg-tertiary: #1a1a1a;
        --surface: #242424; --surface-hover: #2a2a2a; --border: #333333;
        /* ChatGPT風カラー（チャット部分用） (ChatGPT-style colors (for chat part)) */
        --chat-bg: #343541; --sidebar-bg: #202123; --user-msg-bg: #343541;
        --ai-msg-bg: #444654; --chat-border: #4e4f60;
        /* テキストカラー (Text colors) */
        --text-primary: #ffffff; --text-secondary: #b3b3b3; --text-tertiary: #808080;
        /* アクセントカラー (Accent colors) */
        --accent: #7c3aed; --accent-hover: #8b5cf6; --accent-light: rgba(124, 58, 237, 0.15);
        --accent-green: #10a37f;
        /* ステータスカラー (Status colors) */
        --success: #10b981; --error: #ef4444; --warning: #f59e0b; --info: #3b82f6;
    }
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: var(--bg-primary); color: var(--text-primary); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #666; }

    /* ヘッダーの修正 (Header correction) */
    .main-header {
        background: linear-gradient(135deg, var(--accent) 0%, #a855f7 100%);
        padding: 0.1rem 1rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3);
        max-width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    .header-title { font-size: 2.5rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
    .header-subtitle { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .chat-welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; text-align: center; margin-top: -50px; }
    .chat-welcome h2 { color: var(--text-primary); font-size: 2rem; margin-bottom: 1rem; }
    .initial-input-container { margin-top: -100px; width: 100%; max-width: 700px; margin-left: auto; margin-right: auto; }
    .messages-area { padding: 20px 0; min-height: 400px; max-height: calc(100vh - 400px); overflow-y: auto; }
    .message-row { display: flex; padding: 16px 20px; gap: 16px; margin-bottom: 8px; } .user-message-row { background-color: var(--user-msg-bg); } .ai-message-row { background-color: var(--ai-msg-bg); }
    .avatar { width: 36px; height: 36px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 600; flex-shrink: 0; }
    .user-avatar { background-color: #5436DA; color: white; } .ai-avatar { background-color: var(--accent-green); color: white; }
    .message-content { color: var(--text-primary); line-height: 1.6; flex: 1; } .message-content p { margin: 0; }
    .chat-input-area { border-top: 1px solid var(--chat-border); padding: 20px; background-color: var(--chat-bg); border-radius: 0 0 12px 12px; }
    .source-container { background: var(--bg-secondary); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border); margin-top: 1rem; }
    .source-item { background: var(--surface); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; border-left: 3px solid var(--accent); cursor: pointer; transition: all 0.2s ease; }
    .source-item:hover { transform: translateX(4px); background: var(--surface-hover); } .source-title { font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem; }
    .source-excerpt { font-size: 0.875rem; color: var(--text-secondary); line-height: 1.5; }
    .full-text-container { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; max-height: 300px; overflow-y: auto; margin-top: 0.5rem; font-size: 0.875rem; line-height: 1.6; color: var(--text-primary); }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; text-align: center; transition: transform 0.2s ease; } .stat-card:hover { transform: translateY(-2px); }
    .stat-number { font-size: 2rem; font-weight: 700; color: var(--accent); } .stat-label { color: var(--text-secondary); font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.5rem; }
    .stButton > button { background: var(--accent); color: white; border: none; border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 500; transition: all 0.2s ease; } .stButton > button:hover { background: var(--accent-hover); transform: translateY(-1px); }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { background: var(--surface); border: 1px solid var(--border); color: var(--text-primary); border-radius: 8px; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-light); }
    .stFormLabel, div[data-testid="stForm"] label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-secondary) !important; }
    .stTextInput input, .stTextArea textarea, .stSelectbox > div > div > div[data-baseweb="select"] > div { color: var(--text-primary) !important; font-size: 1rem !important; }
    .stSelectbox > div > div > div { background: var(--surface); border: 1px solid var(--border); color: var(--text-primary); }
    .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { background: var(--surface); color: var(--text-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: var(--accent); color: white; border-color: var(--accent); }
    .stFileUploader > div { background: var(--surface); border: 2px dashed var(--border); border-radius: 12px; } .stFileUploader > div:hover { border-color: var(--accent); background: var(--surface-hover); }
    .stProgress > div > div > div > div { background: var(--accent); } div[data-testid="metric-container"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
    .css-1d391kg { background: var(--bg-secondary); } .stAlert { background: var(--surface); color: var(--text-primary); border: 1px solid var(--border); }
    
    /* 用語辞書テーブル用のスタイル */
    .term-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s ease;
    }
    .term-card:hover {
        transform: translateX(2px);
        border-left: 3px solid var(--accent);
    }
    .term-headword {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }
    .term-definition {
        color: var(--text-primary);
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    .term-meta {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    .term-sources {
        color: var(--text-tertiary);
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
</style>
"""
