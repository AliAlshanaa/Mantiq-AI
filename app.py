import os
import glob
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command
from src.core.graph import app as mantiq_app
from src.database.db_manager import db, DB_PATH
from src.database.vector_store import initialize_local_vector_db

import sqlite3


st.set_page_config(page_title="Mantiq-AI", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .rtl-box {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.9;
        white-space: pre-wrap;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-right: 5px solid #6366f1;
        border-radius: 10px;
        padding: 1.4rem;
    }

    .hero-banner {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        color: #ffffff;
        margin-bottom: 1.2rem;
    }
    .hero-banner h1 { margin: 0; font-size: 2rem; }
    .hero-banner p { margin: 4px 0 0 0; opacity: 0.92; }

    .emp-card {
        border-radius: 16px;
        padding: 16px 8px;
        text-align: center;
        border: 2px solid #e2e8f0;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    .emp-card.idle .emp-avatar { filter: grayscale(70%); opacity: 0.55; }
    .emp-card.active {
        border-color: var(--accent);
        background: var(--accent-light);
        box-shadow: 0 0 0 5px var(--accent-glow);
        transform: translateY(-2px);
    }
    .emp-card.active .emp-avatar { animation: bounce 1s infinite; }
    .emp-card.done {
        border-color: var(--accent);
        background: var(--accent-light);
    }
    .emp-avatar { font-size: 2.6rem; display: inline-block; }
    .emp-name { font-weight: 700; font-size: 1rem; margin-top: 6px; color: #1e293b; }
    .emp-title { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
    .emp-status { font-size: 0.8rem; margin-top: 8px; font-weight: 700; color: #94a3b8; }
    .emp-card.active .emp-status, .emp-card.done .emp-status { color: var(--accent); }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


PROVIDER_LABELS = {
    "gemini": "Gemini 2.0 Flash (Google) - أفضل للسياقات الطويلة",
    "openai": "GPT-4o (OpenAI) - أفضل للتحليل المنطقي",
    "llama": "Llama 3.3 70B (Groq) - أسرع استجابة",
}

TONE_OPTIONS = {
    "Professional and Formal": "🎩 رسمي ومهني",
    "Academic, focused on data and citations": "🎓 أكاديمي وبيانات",
    "Creative, storytelling approach": "🎨 إبداعي وسردي",
    "Very concise and direct (bullet points only)": "⚡ مختصر ومباشر",
}

FORMAT_OPTIONS = {
    "Detailed Markdown with hierarchical headers and bullet points": "📋 ماركداون تفصيلي",
    "Executive Summary format: Brief, high-impact paragraphs": "📝 ملخص تنفيذي",
    "Technical whitepaper style with tables and structured data blocks": "🧪 تقني (جداول)",
}


# ------------------------------------------------------------------
# Virtual "Employee" personas representing each agent in the pipeline
# ------------------------------------------------------------------

EMPLOYEES = {
    "researcher":   {"name": "سارة", "title": "محللة الأبحاث",   "avatar": "🔍", "color": "#8b5cf6"},
    "writer":       {"name": "خالد", "title": "كاتب التقارير",   "avatar": "📝", "color": "#3b82f6"},
    "reviewer":     {"name": "ليان", "title": "مراقبة الجودة",   "avatar": "🧐", "color": "#f59e0b"},
    "human_review": {"name": "أنت",  "title": "المدير المسؤول",  "avatar": "👤", "color": "#10b981"},
    "saver":        {"name": "عمر",  "title": "مسؤول الأرشفة",   "avatar": "📦", "color": "#475569"},
}
PIPELINE_ORDER = ["researcher", "writer", "reviewer", "human_review", "saver"]

# Given a node's output, decide which employee picks up the work next.
NEXT_AFTER = {
    "researcher": lambda upd: "writer",
    "writer": lambda upd: "reviewer",
    "reviewer": lambda upd: "writer" if upd.get("next_step") == "REWRITE" else "human_review",
    "human_review": lambda upd: "writer" if upd.get("next_step") == "REWRITE" else "saver",
    "saver": lambda upd: None,
}


def build_message(key, state_update):
    if key == "researcher":
        n = len(state_update.get("research_data", []))
        return (
            f"بحثت في المصادر الداخلية (قاعدة المعرفة، PDF، Excel، قاعدة البيانات) "
            f"والويب، وجمعت **{n}** مرجعًا. أرسلتها إلى خالد لكتابة المسودة. ✅"
        )
    if key == "writer":
        n = len(state_update.get("draft", ""))
        return (
            f"جهّزت مسودة التقرير بالعربية ({n} حرفًا) بناءً على الأبحاث وتفضيلاتك. "
            f"أرسلتها إلى ليان للمراجعة. ✉️"
        )
    if key == "reviewer":
        if state_update.get("next_step") == "REWRITE":
            fb = state_update.get("feedback", "")
            return f"راجعت المسودة ووجدت نقاطًا تحتاج تحسينًا:\n\n> {fb}\n\nأعدتها إلى خالد لتعديلها. 🔄"
        return "راجعت المسودة وهي جاهزة من الناحية الفنية. أحلتها إليك للاعتماد النهائي. ✅"
    if key == "human_review":
        if state_update.get("next_step") == "REWRITE":
            fb = state_update.get("human_feedback", "")
            return f"طلبت تعديلات على المسودة:\n\n> {fb}\n\nأعدتها إلى خالد لتنفيذها. 🔄"
        return "اعتمدت التقرير ✅ وأرسلته إلى عمر للأرشفة النهائية."
    if key == "saver":
        return "حفظت النسخة النهائية من التقرير وأرشفت المهمة في قاعدة البيانات. 📦✅"
    return "..."


# ------------------------------------------------------------------
# Bootstrapping
# ------------------------------------------------------------------

if not os.path.exists("./data/vectorstore") or not os.listdir("./data/vectorstore"):
    with st.spinner("⏳ تهيئة قاعدة المعرفة المحلية لأول مرة..."):
        initialize_local_vector_db()


# ------------------------------------------------------------------
# Session State
# ------------------------------------------------------------------

defaults = {
    "stage": "idle",          # idle | running | interrupted | done
    "thread_config": None,
    "pending_input": None,
    "current_values": {},
    "agent_status": {k: "idle" for k in PIPELINE_ORDER},
    "chat_log": [],            # list of dicts: {key, name, title, avatar, message}
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_session():
    st.session_state.stage = "idle"
    st.session_state.thread_config = None
    st.session_state.pending_input = None
    st.session_state.current_values = {}
    st.session_state.agent_status = {k: "idle" for k in PIPELINE_ORDER}
    st.session_state.chat_log = []


def add_message(emp_key, message):
    emp = EMPLOYEES.get(emp_key, {"name": emp_key, "title": "", "avatar": "🤖"})
    st.session_state.chat_log.append({
        "key": emp_key,
        "name": emp["name"],
        "title": emp["title"],
        "avatar": emp["avatar"],
        "message": message,
    })


def render_team_board(container):
    status_labels = {
        "idle": "⚪ بالانتظار",
        "active": "🟢 يعمل الآن...",
        "done": "✅ مكتمل",
    }
    with container.container():
        st.markdown("#### 👥 فريق العمل")
        cols = st.columns(len(PIPELINE_ORDER))
        for col, emp_key in zip(cols, PIPELINE_ORDER):
            emp = EMPLOYEES[emp_key]
            status = st.session_state.agent_status.get(emp_key, "idle")
            accent = emp["color"]
            with col:
                st.markdown(
                    f"""
                    <div class="emp-card {status}" style="--accent:{accent}; --accent-light:{hex_to_rgba(accent, 0.12)}; --accent-glow:{hex_to_rgba(accent, 0.22)};">
                        <div class="emp-avatar">{emp['avatar']}</div>
                        <div class="emp-name">{emp['name']}</div>
                        <div class="emp-title">{emp['title']}</div>
                        <div class="emp-status">{status_labels[status]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        done_count = sum(1 for s in st.session_state.agent_status.values() if s == "done")
        st.progress(done_count / len(PIPELINE_ORDER))


def render_chat_entry(container, entry):
    with container:
        with st.chat_message(name=entry["key"], avatar=entry["avatar"]):
            st.markdown(f"**{entry['name']} · {entry['title']}**")
            st.write(entry["message"])


def run_until_pause(board_placeholder, chat_container):
    config = st.session_state.thread_config
    pending_input = st.session_state.pending_input

    # Mark the first not-yet-completed agent as the one currently working
    for k in PIPELINE_ORDER:
        if st.session_state.agent_status[k] != "done":
            st.session_state.agent_status[k] = "active"
            break
    render_team_board(board_placeholder)

    for output in mantiq_app.stream(pending_input, config=config):
        if "__interrupt__" in output:
            continue

        for node_name, state_update in output.items():
            emp_key = node_name.lower()
            if emp_key not in EMPLOYEES:
                continue

            st.session_state.agent_status[emp_key] = "done"
            message = build_message(emp_key, state_update)
            add_message(emp_key, message)

            nxt = NEXT_AFTER[emp_key](state_update)
            if nxt:
                st.session_state.agent_status[nxt] = "active"

            render_team_board(board_placeholder)
            render_chat_entry(chat_container, st.session_state.chat_log[-1])

    snapshot = mantiq_app.get_state(config)
    st.session_state.current_values = snapshot.values

    if snapshot.next:
        st.session_state.stage = "interrupted"
        st.session_state.agent_status["human_review"] = "active"
    else:
        st.session_state.stage = "done"
    render_team_board(board_placeholder)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ الإعدادات")

    provider = st.selectbox(
        "🧠 محرك الذكاء الاصطناعي",
        options=list(PROVIDER_LABELS.keys()),
        format_func=lambda k: PROVIDER_LABELS[k],
        index=2,
        disabled=(st.session_state.stage != "idle"),
        help="ملاحظة: مفاتيح Gemini وGPT-4o قد تكون محدودة الحصة، يُنصح بـ Llama (Groq).",
    )

    st.divider()
    st.subheader("👤 الملف الشخصي")

    profile = db.get_user_profile() or {}

    tone_keys = list(TONE_OPTIONS.keys())
    current_tone = profile.get("preferred_tone", tone_keys[0])
    tone_index = tone_keys.index(current_tone) if current_tone in tone_keys else 0

    format_keys = list(FORMAT_OPTIONS.keys())
    current_format = profile.get("formatting_style", format_keys[0])
    format_index = format_keys.index(current_format) if current_format in format_keys else 0

    selected_tone = st.selectbox(
        "🗣️ نمط الأسلوب",
        options=tone_keys,
        format_func=lambda k: TONE_OPTIONS[k],
        index=tone_index,
    )
    selected_format = st.selectbox(
        "🧾 نمط التنسيق",
        options=format_keys,
        format_func=lambda k: FORMAT_OPTIONS[k],
        index=format_index,
    )

    if st.button("💾 حفظ التفضيلات"):
        db.update_user_profile(tone=selected_tone, formatting=selected_format)
        st.success("تم حفظ التفضيلات بنجاح ✅")

    st.divider()
    st.subheader("📜 آخر المهام")
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT task_description, model_used, completion_date "
            "FROM task_history ORDER BY id DESC LIMIT 5"
        ).fetchall()
        conn.close()
        if rows:
            for description, model_used, completion_date in rows:
                short = (description or "").strip()[:35]
                st.caption(f"🗂️ {short}…  \n`{model_used}` · {completion_date}")
        else:
            st.caption("لا توجد مهام سابقة بعد.")
    except Exception as e:
        st.caption(f"تعذر تحميل السجل: {e}")


# ------------------------------------------------------------------
# Main Area
# ------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <h1>🤖 Mantiq-AI</h1>
        <p>غرفة عمليات فريق الذكاء الاصطناعي — Hybrid RAG + Human-in-the-Loop</p>
    </div>
    """,
    unsafe_allow_html=True,
)

task_input = st.text_area(
    "📝 موضوع التقرير المطلوب",
    value=st.session_state.current_values.get("task", "")
    or "اكتب موضوع التقرير الذي تريد توليده هنا...",
    height=100,
    disabled=(st.session_state.stage != "idle"),
)

col_run, col_reset = st.columns([1, 5])

if col_run.button(
    "🚀 بدء التوليد", type="primary", disabled=(st.session_state.stage != "idle")
):
    reset_session()
    st.session_state.thread_config = {
        "configurable": {"thread_id": f"mantiq-{uuid4()}"}
    }
    st.session_state.pending_input = {
        "task": task_input,
        "selected_model": provider,
        "research_data": [],
        "citations": [],
        "draft": "",
        "feedback": "",
        "human_feedback": "",
        "revision_count": 0,
        "next_step": "",
    }
    st.session_state.stage = "running"
    st.rerun()

if col_reset.button("🔄 محادثة جديدة", disabled=(st.session_state.stage == "idle")):
    reset_session()
    st.rerun()


st.markdown("---")

board_placeholder = st.empty()
render_team_board(board_placeholder)

chat_container = st.container()
for entry in st.session_state.chat_log:
    render_chat_entry(chat_container, entry)

if st.session_state.stage == "idle" and not st.session_state.chat_log:
    st.info("👋 الفريق جاهز! أدخل موضوع التقرير أعلاه واضغط 'بدء التوليد' لتشغيل الوكلاء.")


# ------------------------------------------------------------------
# Running
# ------------------------------------------------------------------

if st.session_state.stage == "running":
    try:
        run_until_pause(board_placeholder, chat_container)
    except Exception as e:
        st.session_state.stage = "idle"
        st.error(f"❌ تعذر تنفيذ سير العمل: {e}")
        st.stop()
    st.rerun()


# ------------------------------------------------------------------
# Human-in-the-loop Review
# ------------------------------------------------------------------

if st.session_state.stage == "interrupted":
    values = st.session_state.current_values

    with chat_container:
        with st.chat_message(name="human_review", avatar=EMPLOYEES["human_review"]["avatar"]):
            st.markdown(f"**{EMPLOYEES['human_review']['name']} · {EMPLOYEES['human_review']['title']}**")
            st.write("دورك الآن! راجع المسودة أدناه واتخذ القرار المناسب 👇")

    st.markdown("### 📄 المسودة الحالية")
    st.markdown(f'<div class="rtl-box">{values.get("draft", "")}</div>', unsafe_allow_html=True)

    reviewer_feedback = values.get("feedback", "")
    if reviewer_feedback:
        st.info(f"📋 ملاحظات ليان (المراجعة الآلية): {reviewer_feedback}")

    st.markdown("---")

    approve_col, rewrite_col = st.columns(2)

    with approve_col:
        if st.button("✅ اعتماد التقرير", type="primary"):
            st.session_state.pending_input = Command(
                resume={"decision": "approve", "feedback": ""}
            )
            st.session_state.stage = "running"
            st.rerun()

    with rewrite_col:
        rewrite_feedback = st.text_area(
            "✏️ تعليمات للكاتب (مطلوبة عند طلب التعديل)", key="rewrite_feedback"
        )
        if st.button("🔁 طلب تعديل"):
            st.session_state.pending_input = Command(
                resume={"decision": "rewrite", "feedback": rewrite_feedback}
            )
            st.session_state.stage = "running"
            st.rerun()


# ------------------------------------------------------------------
# Final Result
# ------------------------------------------------------------------

if st.session_state.stage == "done":
    values = st.session_state.current_values

    st.success("✅ اكتملت عملية إنشاء التقرير")

    st.markdown("### 📄 التقرير النهائي")
    st.markdown(f'<div class="rtl-box">{values.get("draft", "")}</div>', unsafe_allow_html=True)

    citations = values.get("citations", [])
    if citations:
        with st.expander("📚 المصادر المستخدمة", expanded=False):
            for i, citation in enumerate(citations, 1):
                st.write(f"[{i}] {citation}")

    files = sorted(
        glob.glob("outputs/report_*.md") + glob.glob("outputs/report_*.pdf"),
        key=os.path.getmtime,
        reverse=True,
    )
    if files:
        latest = files[0]
        with open(latest, "rb") as f:
            st.download_button(
                "⬇️ تحميل التقرير",
                data=f.read(),
                file_name=os.path.basename(latest),
            )
