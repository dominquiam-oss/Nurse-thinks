import streamlit as st
import os
from openai import OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None
if "ngn_case_data" not in st.session_state:
    st.session_state["ngn_case_data"] = None
if "ngn_stage" not in st.session_state:
    st.session_state["ngn_stage"] = 0
if "ngn_history" not in st.session_state:
    st.session_state["ngn_history"] = []
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []



# -------------------------
# NurseThink AI (MVP UI)
# Demo version with simulated responses
# -------------------------

SYSTEM_PROMPT = """
You are NurseThink AI — an NCLEX-style nursing reasoning coach.

SAFETY & SCOPE:
Educational support only. Do not diagnose or prescribe. If user asks for real medical decisions, advise contacting instructor/clinician.
Always stay within nursing scope and common NCLEX test frameworks.

NCLEX REASONING ORDER (use explicitly):
1) Identify Question Type (priority, first action, delegation, teaching, therapeutic response, assessment vs intervention, safety, triage, meds, infection control)
2) Apply Priority Stack (state which rule wins):
   - ABCs (Airway > Breathing > Circulation) / oxygenation
   - Safety (falls, aspiration, bleeding, infection/sepsis, suicide/violence risk, med safety)
   - Acute change/worsening > chronic/stable
   - Unstable > stable
   - Least invasive/least restrictive first (unless emergency)
   - ADPIE: Assess before intervene unless life-threatening
3) If information is missing AND no immediate threat: ask 1–2 clarifying questions OR choose the best assessment.

DELEGATION RULES (algorithmic):
- RN: initial assessment, unstable/new symptoms, clinical judgment, initial teaching, evaluation, care planning.
- LPN/LVN: tasks for stable patients, focused data collection, reinforce teaching, sterile procedures per policy.
- UAP: routine, predictable, non-judgment tasks (ADLs, hygiene, ambulation, vitals on stable, I&O if no judgment).

THERAPEUTIC COMMUNICATION:
Prefer reflection + validation + open-ended. Use silence, clarify, explore. Avoid advice-first, “why” blaming, false reassurance, changing subject.

INFECTION CONTROL QUICK RULES:
Hand hygiene first; standard precautions always; airborne (N95/negative pressure), droplet (surgical mask), contact (gown/gloves).

ANSWER FORMAT (always):
Question Type:
A) Best answer
B) Why (nursing logic + rule used)
C) Why others are wrong (brief)
D) Memory hook/mnemonic
E) Test tip
""".strip()


TEMPLATES = {
    "Priority (ABCs)": "Post-op patient with new shortness of breath and O2 sat 88%. What is the nurse’s priority?",
    "Assessment vs Intervention": "Client reports chest tightness. Which action should the nurse take first?",
    "Therapeutic Communication": "Patient says: “I’m scared my diagnosis means I’m going to die.” Best nurse response?",
    "Delegation": "Which task is appropriate to delegate to the UAP on a stable med-surg unit?",
}
def extract_text_from_upload(uploaded_file) -> str:
    """Return extracted text from .txt or .pdf upload. Safe MVP extraction."""
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    # TXT
    if filename.endswith(".txt"):
        try:
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            return ""

    # PDF
    if filename.endswith(".pdf"):
        try:
            data = uploaded_file.getvalue()
            reader = PdfReader(io.BytesIO(data))
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            return "\n".join(pages_text).strip()
        except Exception:
            return ""

    return ""

def build_context(notes_text: str) -> str:
    notes_text = (notes_text or "").strip()
    return "(none provided)" if not notes_text else notes_text

def build_prompt(mode, request, notes, difficulty, notes_only, label_sources, strict_mode):
    m = (mode or "").lower().strip()

    base = (
        f"{SYSTEM_PROMPT}\n\n"
        f"USER NOTES (primary source):\n{build_context(notes)}\n\n"
    )

    control_rules = f"""
CONTROLS:
- Notes-only mode: {notes_only}
- Label sources: {label_sources}

RULES:
1) If Notes-only mode is TRUE:
   - Use ONLY information explicitly present in USER NOTES.
   - If notes are insufficient, output exactly:
     "INSUFFICIENT NOTES" + a short list of what to add.
   - Do NOT use outside/general nursing knowledge.

2) If Label sources is TRUE:
   - Tag major claims with:
     [Notes] if supported by USER NOTES
     [General] if not found in notes (only allowed when Notes-only mode is FALSE)

3) If USER NOTES are empty AND Notes-only mode is TRUE:
   - Output "INSUFFICIENT NOTES" immediately.
""".strip()

    strict_rules = """
STRICT NCLEX MODE:
- Keep answers concise (no long paragraphs).
- Use bullets for rationales.
- Do not hedge; choose ONE best answer.
""".strip()

    strict_block = ("\n\n" + strict_rules) if strict_mode else ""

    nclex_quality_checklist = """
NCLEX QUALITY CHECKLIST (must satisfy before final answer):
- Did you clearly identify the Question Type?
- Did you explicitly state which priority rule was used (ABCs, Safety, ADPIE, etc.)?
- Did you choose assessment before intervention unless there was an immediate ABC threat?
- Did you stay within nursing scope (no diagnosing/prescribing)?
- Did you avoid adding facts not supported by USER NOTES when Notes-only mode is ON?
- Did you use A–E answer format?
""".strip()
def build_ngn_case_prompt(topic: str) -> str:
    return f"""
You are NurseThink AI creating an NGN-style case progression for nursing students.

Create a 3-stage case on this topic:
TOPIC: {topic}

OUTPUT MUST BE VALID JSON with this schema:
{{
  "title": "string",
  "patient": {{
    "age": int,
    "sex": "string",
    "setting": "string",
    "history": ["string", ...]
  }},
  "stages": [
    {{
      "stage": 1,
      "cues": ["string", ...],
      "question": "string",
      "options": {{
        "key_cues": ["string", ...],
        "hypotheses": ["string", ...],
        "actions": ["string", ...],
        "outcomes": ["string", ...]
      }},
      "best": {{
        "key_cues": ["string", ...],
        "hypothesis": "string",
        "action": "string",
        "outcome": "string"
      }},
      "rationale": "string",
      "next_update": "string"
    }},
    ... stage 2 ...
    ... stage 3 ...
  ]
}}

Rules:
- Make it NCLEX-safe and within nursing scope.
- Use realistic vitals/labs but do not include medication prescribing beyond nursing protocols.
- Ensure "best" answers are clearly supported by the cues.
""".strip()
import json

def build_ngn_case_prompt(topic: str) -> str:
    return f"""
You are NurseThink AI creating an NGN-style case progression for nursing students.

Create a 3-stage NGN case.

TOPIC:
{topic}

OUTPUT FORMAT (STRICT JSON ONLY — NO EXTRA TEXT):

{{
  "title": "string",
  "patient": {{
    "age": 0,
    "sex": "string",
    "setting": "string",
    "history": ["string"]
  }},
  "stages": [
    {{
      "stage": 1,
      "cues": ["string"],
      "question": "string",
      "options": {{
        "key_cues": ["string"],
        "hypotheses": ["string"],
        "actions": ["string"],
        "outcomes": ["string"]
      }},
      "best": {{
        "key_cues": ["string"],
        "hypothesis": "string",
        "action": "string",
        "outcome": "string"
      }},
      "rationale": "string",
      "next_update": "string"
    }}
  ]
}}

RULES:
- NCLEX-safe
- Nursing scope only
- No medical diagnosis or prescribing
- Cues must clearly support the best action
""".strip()


def generate_ngn_case(topic: str) -> dict:
    prompt = build_ngn_case_prompt(topic)
    text = get_ai_response(prompt)

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Could not parse NGN case JSON")
def build_study_chat_prompt(notes: str, notes_only: bool, label_sources: bool, strict_mode: bool, chat_messages: list) -> str:
    # keep last 12 messages so prompts don’t get huge
    recent = chat_messages[-12:]

    convo = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])

    strict_rules = """
STRICT NCLEX MODE:
- Keep answers concise.
- Use bullets for rationales.
- Choose ONE best answer when applicable (no hedging).
""".strip()

    controls = f"""
CONTROLS:
- Notes-only mode: {notes_only}
- Label sources: {label_sources}
- Strict mode: {strict_mode}

RULES:
- This is a back-and-forth tutoring conversation.
- Ask 1–2 clarifying questions if needed.
- Use Socratic coaching: ask, then explain.
- If Notes-only mode is TRUE, only use notes. If insufficient, output "INSUFFICIENT NOTES" and list what notes are needed.
- If Label sources is TRUE, tag major claims [Notes] or [General] (General only allowed when Notes-only is FALSE).
""".strip()

    return f"""
{SYSTEM_PROMPT}

USER NOTES (primary source):
{build_context(notes)}

{controls}

{strict_rules if strict_mode else ""}

CONVERSATION SO FAR:
{convo}

Now respond to the student's latest message.
""".strip()

    # Helper: build full prompt for a given mode block
    def pack(mode_block: str) -> str:
        return (
            base
            + control_rules
            + strict_block
            + "\n\n"
            + mode_block.strip()
            + "\n\n"
            + nclex_quality_checklist
        )

    if m == "explain":
        return pack(f"""
MODE: EXPLAIN / TEACH
REQUEST: {request}

INSTRUCTIONS:
- Explain using USER NOTES first.
- If notes are missing details:
  - If Notes-only is ON: output "INSUFFICIENT NOTES".
  - Otherwise label that section "General overview" and tag [General] if labeling is ON.
- Include a brief example of how it appears on exams.
- If Label sources is ON, tag major claims [Notes] or [General].
- End in A–E format.
""")
    if m == "mixed_drill":
        return pack(f"""
MODE: MIXED NCLEX DRILL
QUESTION: {request}

INSTRUCTIONS:
- FIRST: Identify the Question Type as one of:
  PRIORITY / DELEGATION / THERAPEUTIC COMMUNICATION
- SECOND: Apply the correct decision engine for that question type.
- THIRD: State explicitly which engine you used and why.

ENGINE RULES:
- If the question involves who to see first, what to do first, or unstable vs stable → PRIORITY engine.
- If the question asks who can perform a task or who the RN can assign → DELEGATION engine.
- If the question asks for the nurse’s best response → THERAPEUTIC engine.

REQUIREMENTS:
- First line MUST be: "Question Type: ___ (Engine Used)"
- Do NOT blend engines—choose ONE.
- If Label sources is ON, tag major claims [Notes] or [General].
- If Notes-only is ON and notes lack rules for the identified engine, output "INSUFFICIENT NOTES".
- End in A–E format.
""")

    if m == "priority":
        return pack(f"""
MODE: PRIORITY
QUESTION: {request}

PRIORITY DECISION ALGORITHM (must follow in order):
1) ABCs / Oxygenation
2) Safety
3) Acute change > chronic
4) Unstable > stable
5) Assessment before intervention unless ABCs/safety threat
6) Least invasive first
7) Time-sensitive complications (post-op, OB, cardiac, neuro)
RED FLAGS (any of these automatically win priority):
- SpO₂ < 90%
- Stridor, choking, inability to speak
- Sudden chest pain + dyspnea
- New confusion or LOC change
- Active bleeding
- Signs of sepsis
- Did you clearly state which PRIORITY rule won and why?


INSTRUCTIONS:
- First line MUST be: "Question Type: PRIORITY"
- Identify the FIRST rule in the algorithm that applies and state it explicitly.
- Explain why this rule overrides other considerations.
- If oxygenation or airway is threatened, intervene immediately.
- If no immediate ABC/safety threat, choose assessment first.
- If Label sources is ON, tag major claims [Notes] or [General].
- If Notes-only is ON and notes lack priority rules, output "INSUFFICIENT NOTES".
- End in A–E format.
""")


    if m == "quiz":
        return pack(f"""
MODE: QUIZ ME
TOPIC: {request}
DIFFICULTY: {difficulty}

INSTRUCTIONS:
- Write 1 NCLEX-style question (or NGN-style if appropriate).
- Provide 4 options OR SATA.
- Then answer using A–E format with rationales.
- Add one simple mnemonic.
- If Label sources is ON, tag major claims [Notes] or [General].
- If Notes-only is ON and notes are insufficient, output "INSUFFICIENT NOTES".
""")

    if m == "mnemonics":
        return pack(f"""
MODE: MNEMONICS / MEMORY
TOPIC: {request}

INSTRUCTIONS:
- Create: (1) mnemonic, (2) quick comparison, (3) test trigger cue.
- If Label sources is ON, tag major claims [Notes] or [General].
- If Notes-only is ON and notes are insufficient, output "INSUFFICIENT NOTES".
- End in A–E format.
""")

    if m == "therapeutic":
        return pack(f"""
MODE: THERAPEUTIC COMMUNICATION
PROMPT: {request}

THERAPEUTIC DECISION HIERARCHY (must follow in order):
1) Safety (self-harm, violence, abuse) → assess immediately
2) Acknowledge emotion before giving facts
3) Open-ended > closed-ended
4) Assessment before advice or teaching
5) Present-focused
6) Client-centered language

DO NOT CHOOSE (NCLEX traps):
- False reassurance
- Advice-giving
- "Why" questions
- Nurse-centered statements
- Changing the subject
- Premature teaching

INSTRUCTIONS:
- First line MUST be: "Question Type: THERAPEUTIC COMMUNICATION"
- Provide the BEST therapeutic response as a **direct quote**.
- Explain why it is therapeutic using the hierarchy.
- Explain why 1–2 alternative responses are NOT therapeutic.
- If Label sources is ON, tag major claims [Notes] or [General].
- If Notes-only is ON and notes lack therapeutic principles, output "INSUFFICIENT NOTES".
- End in A–E format.
""")


    if m == "delegation":
        return pack(f"""
MODE: DELEGATION
QUESTION: {request}

DELEGATION DECISION TREE (must follow in order):
1) Unstable or new/worsening condition? → RN
2) Requires assessment, teaching, or evaluation? → RN
3) Stable and predictable? → consider LPN or UAP
4) Routine, non-invasive, non-judgment task? → UAP
5) If unsure → RN

SCOPE RULES:
- RN = A.T.E. (Assess initial, Teach initial, Evaluate)
- LPN/LVN = stable clients, focused data, reinforce teaching
- UAP = ADLs, routine vitals on stable clients, ambulation, I&O (no judgment)

INSTRUCTIONS:
- First line MUST be: "Question Type: DELEGATION"
- State who the task is delegated to AND why.
- Explicitly state why it cannot be delegated to the other roles.
- If Label sources is ON, tag major claims [Notes] or [General].
- If Notes-only is ON and notes lack delegation rules, output "INSUFFICIENT NOTES".
- End in A–E format.
""")



def simulated_response(mode: str, request: str) -> str:
    m = (mode or "").upper().strip()

    return f"""
Question Type: {m if m else "N/A"}

A) Best answer:
(DEMO) This is a simulated response.
Turn on “Use real AI” for a real NCLEX-style answer.

B) Why (nursing logic):
(DEMO) The real AI would analyze this using:
- ABCs
- Safety
- Acute vs chronic
- Unstable vs stable
- ADPIE (assess before intervene)

C) Why others are wrong:
(DEMO) Options would be ruled out if they:
- Delay safety or oxygenation
- Skip assessment
- Require RN judgment when inappropriate
- Focus on comfort before physiology

D) Memory hook/mnemonic:
(DEMO) “ABCs before TLC.”

E) Test tip:
(DEMO) Look for acute change, oxygen issues, and the word “first.”

--------------------------------
Your question:
{request}
""".strip()

def get_ai_response(prompt: str) -> str:
    if not client:
        return "❌ OpenAI API key not found. Please set OPENAI_API_KEY."

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            timeout=30,
        )
        return response.output_text.strip()
    except Exception as e:
        return f"AI error: {type(e).__name__}: {e}"

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="NurseThink AI (MVP)", layout="wide")
st.title("🩺 NurseThink AI (MVP)")
st.caption("Turn your notes into nursing-thinking practice (demo version)")
# Safe defaults to prevent NameError across modes
notes = ""
request = ""
show_prompt = False
generate = False
ngn_topic = "Post-op respiratory complication"
ngn_start = False
notes_only = False
label_sources = True
strict_mode = True
use_real_ai = False
mode = "priority"
difficulty = "medium"

left, right = st.columns([1, 1])

with left:
    st.subheader("Inputs")

    # Core toggles
    use_real_ai = st.checkbox("Use real AI (costs money)", value=False)
    strict_mode = st.checkbox("Strict NCLEX mode (more structured)", value=True)
    notes_only = st.checkbox("Notes-only mode (don’t guess; use only my notes)", value=False)
    label_sources = st.checkbox("Label sources (Notes vs General)", value=True)

    # Optional templates
    template_choice = st.selectbox(
        "Quick template (optional)",
        ["(none)"] + list(TEMPLATES.keys())
    )
    if template_choice != "(none)":
        st.session_state["request_text"] = TEMPLATES[template_choice]

    # Mode selector
    mode = st.selectbox(
        "Mode",
        ["priority", "delegation", "therapeutic", "mixed_drill", "ngn_case", "study_chat", "quiz", "explain", "mnemonics"],
        index=0
    )

    # NGN controls (only show in NGN mode)
    if mode == "ngn_case":
        ngn_topic = st.text_input("NGN Case Topic (optional)", value="Post-op respiratory complication")
        ngn_start = st.button("Start new NGN case")

    # Quiz difficulty
    difficulty = st.selectbox("Quiz difficulty", ["easy", "medium", "hard"], index=1)

    # Upload notes
    uploaded = st.file_uploader(
        "Upload notes (TXT or PDF)",
        type=["txt", "pdf"],
        accept_multiple_files=False
    )

    # Extract from upload
    extracted_notes = ""
    if uploaded is not None:
        extracted = extract_text_from_upload(uploaded)
        if extracted:
            st.success(f"Loaded notes from: {uploaded.name}")
            extracted_notes = extracted
        else:
            st.warning("I couldn’t extract text from that file. Try a .txt export or copy/paste notes.")
            extracted_notes = ""

    # Notes box (prefill with extracted notes if available)
    notes = st.text_area(
        "Notes (paste or upload above)",
        value=extracted_notes,
        height=220,
        placeholder="Paste lecture notes, study guide, etc."
    )

    # Question / scenario
    request = st.text_area(
        "Question / Scenario",
        key="request_text",
        height=140,
        placeholder="Paste an NCLEX-style question, scenario, or topic..."
    )

    show_prompt = st.checkbox("Show generated prompt", value=False)

    # Main action button (always exists for non-NGN modes)
    generate = st.button("Generate", type="primary")


with right:
    st.subheader("Output")

    if mode == "ngn_case":
        # Start a new case
        if ngn_start:
            if not use_real_ai:
                st.error("NGN case generation requires Real AI ON.")
                st.stop()
            with st.spinner("Generating NGN case…"):
                st.session_state["ngn_case_data"] = generate_ngn_case(ngn_topic)
                st.session_state["ngn_stage"] = 0
                st.session_state["ngn_history"] = []

        case = st.session_state.get("ngn_case_data")

        if not case:
            st.info("Click **Start new NGN case** to generate a case progression.")
        else:
            st.markdown(f"### {case.get('title','NGN Case')}")
            patient = case.get("patient", {})
            st.markdown(
                f"**Patient:** {patient.get('age','?')}-year-old {patient.get('sex','?')} | "
                f"**Setting:** {patient.get('setting','?')}"
            )
            if patient.get("history"):
                st.markdown("**History:**")
                for h in patient["history"]:
                    st.markdown(f"- {h}")

            stage_idx = st.session_state["ngn_stage"]
            stages = case.get("stages", [])
            if stage_idx >= len(stages):
                st.success("Case complete ✅")
                if st.session_state["ngn_history"]:
                    st.markdown("### Your performance summary")
                    correct = sum(1 for x in st.session_state["ngn_history"] if x["score"] == 4)
                    st.write(f"Perfect stages: {correct}/{len(st.session_state['ngn_history'])}")
                st.stop()

            stage = stages[stage_idx]
            st.markdown(f"## Stage {stage.get('stage', stage_idx+1)}")
            st.markdown("**Cues:**")
            for c in stage.get("cues", []):
                st.markdown(f"- {c}")

            st.markdown(f"**Prompt:** {stage.get('question','')}")

            opts = stage.get("options", {})
            best = stage.get("best", {})

            # Student inputs
            chosen_key_cues = st.multiselect(
                "Select key cues (choose 2–4)",
                opts.get("key_cues", []),
                key=f"kc_{stage_idx}"
            )
            chosen_hypothesis = st.radio(
                "Most likely hypothesis",
                opts.get("hypotheses", []),
                key=f"hyp_{stage_idx}"
            )
            chosen_action = st.radio(
                "Priority nursing action",
                opts.get("actions", []),
                key=f"act_{stage_idx}"
            )
            chosen_outcome = st.radio(
                "Expected outcome / evaluation",
                opts.get("outcomes", []),
                key=f"out_{stage_idx}"
            )

            submit_stage = st.button("Submit Stage", key=f"submit_{stage_idx}")

            if submit_stage:
                # Simple scoring: 1 point each component
                score = 0
                # key cues scoring: award 1 if majority overlap
                best_kc = set(best.get("key_cues", []))
                chosen_kc = set(chosen_key_cues)
                if best_kc and len(best_kc.intersection(chosen_kc)) >= max(1, len(best_kc)//2):
                    score += 1
                if chosen_hypothesis == best.get("hypothesis"):
                    score += 1
                if chosen_action == best.get("action"):
                    score += 1
                if chosen_outcome == best.get("outcome"):
                    score += 1

                st.session_state["ngn_history"].append({
                    "stage": stage.get("stage", stage_idx+1),
                    "score": score,
                    "chosen": {
                        "key_cues": list(chosen_kc),
                        "hypothesis": chosen_hypothesis,
                        "action": chosen_action,
                        "outcome": chosen_outcome
                    }
                })

                # Feedback
                st.markdown("### Feedback")
                st.write(f"Score: **{score}/4**")

                st.markdown("**Best answers:**")
                st.write(f"- Key cues: {best.get('key_cues')}")
                st.write(f"- Hypothesis: {best.get('hypothesis')}")
                st.write(f"- Action: {best.get('action')}")
                st.write(f"- Outcome: {best.get('outcome')}")

                st.markdown("**Rationale:**")
                st.write(stage.get("rationale", ""))

                st.markdown("**Next update:**")
                st.write(stage.get("next_update", ""))

                if st.button("Continue to next stage"):
                    st.session_state["ngn_stage"] += 1
                    st.rerun()

        st.stop()
    if mode == "study_chat":
        st.markdown("### Study Chat")

        # Optional: clear chat
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Clear chat"):
                st.session_state["chat_messages"] = []
                st.rerun()

        # Show conversation
        for msg in st.session_state["chat_messages"]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**NurseThink:** {msg['content']}")

        # Input
        chat_input = st.text_area(
            "Type your message",
            height=100,
            placeholder="Ask a question, paste a scenario, or say 'quiz me on cardiac meds'..."
        )
        send = st.button("Send")

        if send:
            if not chat_input.strip():
                st.warning("Type a message first.")
                st.stop()

            if notes_only and not notes.strip():
                st.error("Notes-only mode is ON, but no notes were provided. Upload/paste notes or turn Notes-only off.")
                st.stop()

            # Add user message
            st.session_state["chat_messages"].append({"role": "user", "content": chat_input.strip()})

            if not use_real_ai:
                # Demo response (no API call)
                demo_reply = "*(DEMO)* Turn on **Use real AI** to chat back-and-forth. You can also upload notes for more accurate coaching."
                st.session_state["chat_messages"].append({"role": "assistant", "content": demo_reply})
                st.rerun()

            # Real AI response
            prompt = build_study_chat_prompt(notes, notes_only, label_sources, strict_mode, st.session_state["chat_messages"])
            with st.spinner("Thinking…"):
                reply = get_ai_response(prompt)

            st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
            st.rerun()

        st.stop()

        # ---- Existing non-NGN output logic below here ----
    if mode not in ["ngn_case", "study_chat"]:
        if generate:
            if not request.strip():
                st.warning("Add a question/scenario first.")
            else:
                # Guard: notes-only mode
                if notes_only and not notes.strip():
                    st.error(
                        "Notes-only mode is ON, but no notes were provided. "
                        "Upload/paste notes or turn Notes-only off."
                    )
                    st.stop()

                prompt = build_prompt(
                    mode,
                    request,
                    notes,
                    difficulty,
                    notes_only,
                    label_sources,
                    strict_mode
                )

                if show_prompt:
                    st.markdown("**Generated Prompt**")
                    st.code(prompt, language="text")

                if use_real_ai:
                    st.markdown("**Response (Real AI)**")
                    with st.spinner("Thinking…"):
                        answer = get_ai_response(prompt)
                    st.text(answer)
                else:
                    st.markdown("**Response (Simulated Demo)**")
                    st.text(simulated_response(mode, request))
        else:
            st.info("Choose a mode, paste notes + a scenario, then click Generate.")
