import streamlit as st
import json
import re
from groq import Groq
import requests
from datetime import datetime

# =========================
# Configuration
# =========================
GROQ_API_KEY = "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"
SAP_ID = "70022200506"
API_ENDPOINT = "https://se-payment-verification-api.service.external.usea2.aws.prodigaltech.com/api/validate-payment"
GITHUB_REPO_URL = "https://github.com/AakashPathak07/prodigal-payment-analyzer"  # update if needed

client = Groq(api_key=GROQ_API_KEY)

# =========================
# Helpers
# =========================
def seconds_to_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"

def parse_transcript(transcript):
    if isinstance(transcript, str):
        transcript = json.loads(transcript)
    lines = []
    for turn in transcript:
        ts = seconds_to_timestamp(turn["stime"])
        lines.append(f"[{ts}] {turn['role']}: {turn['utterance']}")
    return "\n".join(lines)

def normalize_amount(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def luhn_ok(num: str) -> bool:
    if not num.isdigit():
        return False
    digits = [int(c) for c in num]
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

def end_of_month_expired(expiry_month: int, expiry_year: int, now: datetime) -> bool:
    # Card valid through end of expiry month
    return (expiry_year, expiry_month) < (now.year, now.month)

def _final(valid, reason, name, card, cvv, m, y, amt):
    return {
        "payment_valid": bool(valid),
        "failure_reason": str(reason),
        "credentials": {
            "cardholderName": name,
            "cardNumber": card,
            "cvv": cvv,
            "expiryMonth": int(m),
            "expiryYear": int(y)
        },
        "amount": float(amt)
    }

def coerce_and_correct(extracted: dict, strict_luhn: bool = False) -> dict:
    """
    Deterministically set payment_valid and failure_reason based on rules,
    overriding any incorrect LLM judgment. Supports multiple transcript types.
    """
    name = str(extracted.get("cardholderName", "")).strip()
    raw_card = str(extracted.get("cardNumber", "")).strip()
    card = "".join(ch for ch in raw_card if ch.isdigit())
    cvv = "".join(ch for ch in str(extracted.get("cvv", "")).strip() if ch.isdigit())
    month = int(extracted.get("expiryMonth", 0) or 0)
    year = int(extracted.get("expiryYear", 0) or 0)
    amount = normalize_amount(extracted.get("amount", 0))

    # If no card present at all, treat as no attempt
    if len(card) == 0 and len(raw_card) == 0:
        return _final(False, "data_mismatch", name, card, cvv, month, year, amount)

    # Invalid month
    if not (1 <= month <= 12):
        return _final(False, "invalid_expiry_month", name, card, cvv, month, year, amount)

    # Masked number
    if any(c in raw_card for c in ["*", "X", "x"]):
        return _final(False, "masked_card_number", name, card, cvv, month, year, amount)

    # Card length
    if not (13 <= len(card) <= 19):
        return _final(False, "invalid_card_length", name, card, cvv, month, year, amount)

    # CVV length
    if not (3 <= len(cvv) <= 4):
        return _final(False, "invalid_cvv_length", name, card, cvv, month, year, amount)

    # Year minimum
    if year < 2024:
        return _final(False, "expired_card", name, card, cvv, month, year, amount)

    # Expiry check (current assignment date fixed)
    now = datetime(2025, 11, 8, 0, 0, 0)
    if end_of_month_expired(month, year, now):
        return _final(False, "expired_card", name, card, cvv, month, year, amount)

    # Luhn optional (toggle in sidebar). If false, let API adjudicate.
    if strict_luhn and not luhn_ok(card):
        return _final(False, "invalid_luhn", name, card, cvv, month, year, amount)

    # If all checks pass, it's valid
    return _final(True, "none", name, card, cvv, month, year, amount)

def validate_payment_api(transcript_id: str, payment_data: dict) -> dict:
    # Ensure proper schema: if valid, failure_reason MUST be "none"
    failure_reason = payment_data["failure_reason"] if not payment_data["payment_valid"] else "none"
    payload = {
        "id": str(transcript_id),
        "student_id": str(SAP_ID),
        "payment_valid": bool(payment_data["payment_valid"]),
        "failure_reason": str(failure_reason),
        "credentials": {
            "cardholderName": str(payment_data["credentials"]["cardholderName"]),
            "cardNumber": str(payment_data["credentials"]["cardNumber"]),
            "cvv": str(payment_data["credentials"]["cvv"]),
            "expiryMonth": int(payment_data["credentials"]["expiryMonth"]),
            "expiryYear": int(payment_data["credentials"]["expiryYear"]),
        },
        "amount": float(payment_data["amount"]),
    }

    if st.session_state.get("debug_api", False):
        st.code(json.dumps(payload, indent=2), language="json")

    try:
        resp = requests.post(
            API_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30,
        )
        try:
            j = resp.json()
        except Exception:
            j = {"error": resp.text}
        return {"status_code": resp.status_code, "response": j}
    except Exception as e:
        return {"status_code": 500, "response": {"error": str(e)}}

# =========================
# LLM Tasks
# =========================
TASK1_SYSTEM = "You are an expert call transcript analyzer. Always return valid JSON only."
TASK2_SYSTEM = "Extract payment fields precisely. Return only valid JSON."

def analyze_call_transcript_raw(transcript_text: str):
    prompt = f"""Analyze the following debt collection call and return ONLY valid JSON.

Transcript:
{transcript_text}

Return this structure:
{{
  "payment_attempted": boolean,
  "customer_intent_explanation": "1-2 sentence rationale of whether the customer truly intended to pay",
  "customer_sentiment": {{
    "classification": "Satisfied|Neutral|Frustrated|Hostile",
    "description": "1-2 sentence rationale"
  }},
  "agent_performance": "2-3 sentences on professionalism, patience, clarity",
  "timestamped_events": [
    {{"timestamp":"MM:SS","event_type":"disclosure|offer_negotiation|payment_setup_attempt|customer_frustration","description":"one-liner"}}
  ]
}}"""
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": TASK1_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    txt = r.choices[0].message.content
    if st.session_state.get("debug_llm", False):
        st.code(txt)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return json.loads(m.group() if m else txt)

def boolean_intent_from_explanation(expl: str, transcript_text: str) -> bool:
    """
    Deterministic boolean intent based on cues in explanation + transcript:
      True if: explicit willingness + provided usable PAN/expiry/CVV/amount + authorization phrase
      False if: refusal/avoidance/hostility/no details/asks for transfer only
    """
    t = transcript_text.lower()
    e = (expl or "").lower()

    positive_cues = [
        "i can pay", "i will pay", "ready to pay", "authorize", "yes, i do",
        "let's do it", "go ahead", "make the payment", "pay now"
    ]
    negative_cues = [
        "not paying", "cannot pay", "won't pay", "do not want", "stop calling",
        "call me later", "not today", "transfer me", "real person", "hang up"
    ]

    has_positive = any(p in t or p in e for p in positive_cues)
    has_negative = any(n in t or n in e for n in negative_cues)

    # Usable details present?
    has_pan = bool(re.search(r"\b\d{13,19}\b", t))
    has_expiry = bool(re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|\b(0?[1-9]|1[0-2])\\/(20\\d\\d|\\d\\d)\\b", t))
    has_cvv = bool(re.search(r"\\b\\d{3,4}\\b", t))
    has_auth = any(a in t for a in ["do you authorize", "i authorize", "yes, i do", "yes i do", "i consent"])

    if has_negative and not has_positive:
        return False

    # If payment attempt announced and details present, prefer True
    if has_positive and (has_pan or has_expiry) and has_cvv:
        return True

    # Fallback to explanation-positive
    return has_positive and not has_negative

def extract_payment_llm(transcript_text: str):
    # Emphasize "latest attempt" to handle multiple retries/corrections
    prompt = f"""Extract ONLY the raw payment fields from this call. If there are multiple payment attempts or corrections,
return the latest complete attempt. Convert spoken numbers to digits.

Transcript:
{transcript_text}

Return valid JSON only:
{{
  "cardholderName": "string",
  "cardNumber": "digits only",
  "cvv": "digits only",
  "expiryMonth": integer,
  "expiryYear": integer,
  "amount": number
}}"""
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": TASK2_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=700,
    )
    txt = r.choices[0].message.content
    if st.session_state.get("debug_llm", False):
        st.code(txt)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return json.loads(m.group() if m else txt)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Payment Call Transcript Analyzer", layout="wide")
st.title("AI System for Payment Processing Call Transcripts")
st.caption("Prodigal Technologies - Assignment • GitHub: " + GITHUB_REPO_URL)

with st.sidebar:
    st.header("Run Options")
    st.checkbox("Show LLM debug", key="debug_llm", value=False)
    st.checkbox("Show API payload", key="debug_api", value=False)
    st.checkbox("Strict Luhn check", key="strict_luhn", value=False)
    st.markdown("---")
    st.write("Tip: If some transcripts contain non-Luhn test PANs, keep Strict Luhn off.")

tab1, tab2 = st.tabs(["Upload Transcript", "Paste Transcript"])
transcript, transcript_id = None, None

with tab1:
    f = st.file_uploader("Upload a JSON transcript", type=["json"])
    if f:
        transcript = json.load(f)
        transcript_id = f.name.replace(".json", "")
        st.success(f"Loaded: {f.name}")

with tab2:
    raw = st.text_area("Paste transcript JSON", height=240)
    tid = st.text_input("Transcript ID (e.g., f3a7c9e2)")
    if raw and tid:
        try:
            transcript = json.loads(raw)
            transcript_id = tid
            st.success(f"Transcript ID: {transcript_id}")
        except Exception:
            st.error("Invalid JSON")

if transcript and transcript_id:
    st.markdown("---")
    if st.button("Analyze Transcript", type="primary"):
        text = parse_transcript(transcript)

        # Task 1: Analysis with boolean intent
        st.subheader("Task 1: Call Analysis")
        try:
            raw_t1 = analyze_call_transcript_raw(text)
            # Derive strict boolean for intent
            intent_bool = boolean_intent_from_explanation(raw_t1.get("customer_intent_explanation", ""), text)

            task1 = {
                "payment_attempted": bool(raw_t1.get("payment_attempted", False)),
                "customer_intent": bool(intent_bool),
                "customer_sentiment": raw_t1.get("customer_sentiment", {}),
                "agent_performance": raw_t1.get("agent_performance", ""),
                "timestamped_events": raw_t1.get("timestamped_events", [])
            }

            c1, c2 = st.columns(2)
            c1.metric("Payment Attempted", "Yes" if task1["payment_attempted"] else "No")
            c2.metric("Customer Intent", "Genuine" if task1["customer_intent"] else "Not Genuine")
            sentiment = task1.get("customer_sentiment", {})
            st.write(f"Sentiment: {sentiment.get('classification', 'N/A')} — {sentiment.get('description', '')}")
            st.write("Agent Performance:", task1.get("agent_performance", ""))
            st.write("Events:")
            for e in task1.get("timestamped_events", []):
                st.write(f"- [{e.get('timestamp')}] {e.get('event_type')}: {e.get('description')}")
            with st.expander("Raw Task 1 JSON"):
                st.json(task1)
        except Exception as e:
            st.error(f"Task 1 error: {e}")

        st.markdown("---")

        # Task 2: Extraction + validated API call
        st.subheader("Task 2: Payment Validation Results")
        try:
            raw_fields = extract_payment_llm(text)
            corrected = coerce_and_correct(raw_fields, strict_luhn=st.session_state.get("strict_luhn", False))

            c1, c2, c3 = st.columns(3)
            c1.write(f"Cardholder: {corrected['credentials']['cardholderName']}")
            c1.write(f"Card Number: {corrected['credentials']['cardNumber']}")
            c2.write(f"Expiry: {corrected['credentials']['expiryMonth']}/{corrected['credentials']['expiryYear']}")
            c2.write(f"CVV: {corrected['credentials']['cvv']}")
            c3.write(f"Amount: ${corrected['amount']}")
            c3.write(f"Valid: {'Yes' if corrected['payment_valid'] else 'No'}")
            st.write(f"Failure Reason: `{corrected['failure_reason']}`")

            api_result = validate_payment_api(transcript_id, corrected)

            resp = api_result.get("response", {})
            status = api_result.get("status_code")
            ok = resp.get("success", False)

            st.markdown("**API Validation Result:**")
            if ok:
                st.success(resp.get("message", "Payment validated successfully"))
            else:
                st.error(resp.get("message", "Validation failed"))
                if "failureReason" in resp:
                    st.warning(f"API Failure Reason: {resp['failureReason']}")
                if "mismatches" in resp:
                    st.warning("Mismatched Fields: " + ", ".join(resp["mismatches"]))
                if "error" in resp:
                    st.error("Error: " + str(resp["error"]))

            if status == 200:
                st.markdown("Status Code: 200 (Success)")
            elif status == 422:
                st.markdown("Status Code: 422 (Validation Error)")
            elif status == 404:
                st.markdown("Status Code: 404 (Not Found)")
            else:
                st.markdown(f"Status Code: {status}")

            with st.expander("View Full API Response"):
                st.json(api_result)
        except Exception as e:
            st.error(f"Task 2 error: {e}")

st.markdown("---")
st.caption("Built with Streamlit • GROQ • LLaMA 3.3 70B • GitHub: " + GITHUB_REPO_URL)
