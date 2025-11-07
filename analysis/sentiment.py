from typing import Any, Dict, List, Tuple, Optional
import math
from llm_interface.base_llm import BaseLLM
from prompts.prompt_templates import EXTRA_SENTIMENT_PROMPTS
from utils.helpers import parse_json_from_string
from config import SENTIMENT_CUES, SENTIMENT_WEIGHTS, TOM_AGENT_CONFIG, DOMAIN_AGENT_CONFIG
from agents import ToMAgent, DomainAgent
from memory import SocialMemory

def _normalize_weights(values: List[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values) if values else []
    return [v / total for v in values]


def heuristic_sentiment_from_hypotheses(hypotheses: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Compute a deterministic, config-driven sentiment from a list of ToM/Domain
    hypotheses. Used as a safety net (fallback and default filler) when the LLM
    synthesis returns malformed/partial JSON, or when you explicitly want a
    non-stochastic baseline.

    Inputs
    - hypotheses: List of hypothesis dicts. Each item should include:
        - "type": str (Belief|Desire|Intention|Emotion|Thought|Unknown)
        - "explanation": str (free text from ToM/Domain)
      and may include weighting signals used here:
        - "score": float (preferred)
        - "p_cond": float (fallback if score is missing)

    Method
    - Weights hypotheses by normalized `score` (or `p_cond`).
    - Applies cue lists from config (global + type-specific) to the hypothesis
      explanation text, accumulating positive/negative evidence with tunable
      weights (SENTIMENT_WEIGHTS).
    - Adds a small negative prior for Desire types (configurable) to reflect
      often-unmet preferences unless stated positively.

    Returns
    - (polarity, confidence):
        - polarity in {"positive", "neutral", "negative"}
        - confidence in [0,1], loosely reflecting evidence magnitude
        
    Notes
    - This does not call the LLM; it is fast and deterministic given inputs and
      config. Adjust `SENTIMENT_CUES` and `SENTIMENT_WEIGHTS` in config.py to
      tune behavior or effectively disable it (e.g., set weights to 0).
    """
    if not hypotheses:
        return "neutral", 0.4

    # Normalize weights using score if available, else p_cond
    weights: List[float] = []
    for h in hypotheses:
        try:
            w = float(h.get("score", h.get("p_cond", 0.5)))
        except Exception:
            w = 0.5
        weights.append(max(0.0, w))
    norm_weights = _normalize_weights(weights)

    # Cues from config
    base_pos = [c.lower() for c in SENTIMENT_CUES.get("positive", [])]
    base_neg = [c.lower() for c in SENTIMENT_CUES.get("negative", [])]
    emo_pos = [c.lower() for c in SENTIMENT_CUES.get("emotion", {}).get("positive", [])]
    emo_neg = [c.lower() for c in SENTIMENT_CUES.get("emotion", {}).get("negative", [])]
    int_pos = [c.lower() for c in SENTIMENT_CUES.get("intention", {}).get("positive", [])]
    int_neg = [c.lower() for c in SENTIMENT_CUES.get("intention", {}).get("negative", [])]
    bel_pos = [c.lower() for c in SENTIMENT_CUES.get("belief", {}).get("positive", [])]
    bel_neg = [c.lower() for c in SENTIMENT_CUES.get("belief", {}).get("negative", [])]

    # Weights from config
    w_base_pos = float(SENTIMENT_WEIGHTS.get("base_positive_cue", 0.5))
    w_base_neg = float(SENTIMENT_WEIGHTS.get("base_negative_cue", 0.5))
    w_emo_pos = float(SENTIMENT_WEIGHTS.get("emotion_positive", 0.4))
    w_emo_neg = float(SENTIMENT_WEIGHTS.get("emotion_negative", 0.4))
    w_int_pos = float(SENTIMENT_WEIGHTS.get("intention_positive", 0.3))
    w_int_neg = float(SENTIMENT_WEIGHTS.get("intention_negative", 0.3))
    w_bel_pos = float(SENTIMENT_WEIGHTS.get("belief_positive", 0.3))
    w_bel_neg = float(SENTIMENT_WEIGHTS.get("belief_negative", 0.3))
    w_desire_prior_neg = float(SENTIMENT_WEIGHTS.get("desire_neg_prior", 0.2))

    pos = 0.0
    neg = 0.0

    for h, w in zip(hypotheses, norm_weights):
        t = str(h.get("type", "Unknown")).lower()
        expl = str(h.get("explanation", "")).lower()

        if t == "desire":
            neg += w_desire_prior_neg * w
        if t == "emotion":
            if any(k in expl for k in emo_pos):
                pos += w_emo_pos * w
            if any(k in expl for k in emo_neg):
                neg += w_emo_neg * w
        elif t == "intention":
            if any(k in expl for k in int_pos):
                pos += w_int_pos * w
            if any(k in expl for k in int_neg):
                neg += w_int_neg * w
        elif t == "belief":
            if any(k in expl for k in bel_pos):
                pos += w_bel_pos * w
            if any(k in expl for k in bel_neg):
                neg += w_bel_neg * w

        if any(k in expl for k in base_pos):
            pos += w_base_pos * w
        if any(k in expl for k in base_neg):
            neg += w_base_neg * w

    gap = pos - neg
    mag = max(pos, neg)
    if abs(gap) < 0.1:
        return "neutral", min(1.0, 0.5 + 0.3 * mag)
    if gap > 0:
        return "positive", min(1.0, 0.6 + 0.3 * pos)
    return "negative", min(1.0, 0.6 + 0.3 * neg)

def _format_context(context: List[Dict[str, str]]) -> str:
    if not context:
        return "No previous conversation history."
    return "\n".join(f"{t.get('speaker','Unknown')}: {t.get('utterance','')}" for t in context)

def _top_k_hypotheses(hypotheses: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    if not hypotheses:
        return []
    def key(h):
        return float(h.get("score", h.get("p_cond", 0.0)))
    return sorted(hypotheses, key=key, reverse=True)[:k]

def synthesize_with_llm(
    llm: BaseLLM,
    user_input: str,
    conversation_context: List[Dict[str, str]],
    hypotheses: List[Dict[str, Any]],
    max_retries: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Consolidate ToM/Domain hypotheses into a single sentiment JSON using the
    SENTIMENT_SYNTHESIS_JSON prompt.

    Parameters
    - llm: A BaseLLM implementation. Only `.generate()` is used.
    - user_input: The raw user review/utterance (u_t).
    - conversation_context: List of turns with shape
      [{"speaker": str, "utterance": str}, ...]. This is formatted and
      passed as C_t in the prompt.
    - hypotheses: List of hypothesis dicts produced by ToM/Domain. Each item
      should include at least "explanation" and "type", and ideally "score",
      "p_cond", and "ig". Top-k (by score, fallback p_cond) are embedded in
      the prompt to anchor the LLM.
    - max_retries: Number of additional attempts (with stricter JSON reminder)
      if parsing fails. Total attempts = max_retries + 1.

    Returns
    - Dict[str, Any] on success (keys like polarity, intensity, aspects,
      evidence, confidence) with source="metamind"; or None when parsing fails
      after all retries.

    Notes
    - Uses utils.helpers.parse_json_from_string; does not throw on LLM errors.
    - Determinism depends on model/temperature; prompt enforces strict JSON.
    """
    C_t = _format_context(conversation_context)
    top = _top_k_hypotheses(hypotheses)
    prompt = EXTRA_SENTIMENT_PROMPTS["SENTIMENT_SYNTHESIS_JSON"].format(
        u_t=user_input,
        C_t=C_t,
    ) + "\nTop Hypotheses (H):\n" + str(top)

    last_text = None
    for _ in range(max_retries + 1):
        last_text = llm.generate(prompt, temperature=0.3, max_tokens=350)
        parsed = parse_json_from_string(last_text)
        if parsed:
            parsed["source"] = "metamind"
            return parsed
        prompt = prompt + "\nReminder: Return ONLY valid JSON."

    return None

def extract_aspects_with_llm(
    llm: BaseLLM,
    user_input: str,
    conversation_context: List[Dict[str, str]],
    max_retries: int = 1
) -> List[Dict[str, Any]]:
    """
    Extract per-aspect sentiment from input/context via ASPECT_EXTRACTION_JSON.

    Parameters
    - llm: BaseLLM used for generation.
    - user_input: Raw review/utterance (u_t).
    - conversation_context: Turns [{"speaker", "utterance"}] formatted as C_t.
    - max_retries: Retry count on JSON parse failure.

    Returns
    - List of {name, sentiment, evidence} dicts; empty list on persistent
      parsing failure.

    Notes
    - Independent from synthesis; callers usually merge aspects into the
      synthesized result if missing, or use alongside heuristic fallback.
    - No deduping/normalization is applied here.
    """
    C_t = _format_context(conversation_context)
    prompt = EXTRA_SENTIMENT_PROMPTS["ASPECT_EXTRACTION_JSON"].format(u_t=user_input, C_t=C_t)

    last_text = None
    for _ in range(max_retries + 1):
        last_text = llm.generate(prompt, temperature=0.3, max_tokens=250)
        parsed = parse_json_from_string(last_text)
        if parsed and isinstance(parsed.get("aspects"), list):
            return parsed["aspects"]
        prompt = prompt + "\nIMPORTANT: Output ONLY JSON matching the schema."

    return []

def generate_recommendation_with_llm(
    llm: BaseLLM,
    user_input: str,
    conversation_context: List[Dict[str, str]],
    aspects: List[Dict[str, Any]],
    polarity: str,
    intensity: float,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Third-stage insight generator building on sentiment and aspects. It asks the
    LLM to produce a concise business insight and recommendations using the
    RECOMMENDATION_SUMMARY_JSON prompt.

    Parameters
    - llm: BaseLLM used to generate the JSON (only `.generate()` is required).
    - user_input: Original review/utterance (u_t).
    - conversation_context: Recent turns for C_t formatting.
    - aspects: List of {name, sentiment, evidence} dicts from aspect extraction.
    - polarity: Overall sentiment label (positive|neutral|negative).
    - intensity: Overall sentiment intensity in [0,1].
    - max_retries: Retry count on JSON parse failure.

    Returns
    - Dict with keys {summary: str, drivers: str, actions: [str]}. Falls back to
      a deterministic template if JSON parsing fails (positive -> brief
      acknowledgement; neutral/negative -> short actionable steps derived from
      negative aspects).

    Notes
    - This function does not alter upstream results; callers typically attach
      its output under a "recommendation" key in the final payload.
    - Evidence and aspect names are assumed as-is; consider normalization if you
      need consistent reporting across datasets.
    """
    C_t = _format_context(conversation_context)
    aspects_json = str(aspects)
    prompt = EXTRA_SENTIMENT_PROMPTS["RECOMMENDATION_SUMMARY_JSON"].format(
        u_t=user_input,
        C_t=C_t,
        polarity=polarity,
        intensity=intensity,
        aspects_json=aspects_json,
    )

    last_text = None
    for _ in range(max_retries + 1):
        last_text = llm.generate(prompt, temperature=0.3, max_tokens=240)
        parsed = parse_json_from_string(last_text)
        if parsed and all(k in parsed for k in ("summary", "drivers", "actions")):
            return {
                "summary": str(parsed.get("summary", "")).strip(),
                "drivers": str(parsed.get("drivers", "")).strip(),
                "actions": [str(a) for a in (parsed.get("actions") or [])],
            }
        prompt += "\nIMPORTANT: Output ONLY valid JSON with keys summary, drivers, actions."

    # Deterministic fallback
    pos_aspects = [a for a in aspects if str(a.get("sentiment", "")).lower() == "positive"]
    neg_aspects = [a for a in aspects if str(a.get("sentiment", "")).lower() == "negative"]

    if str(polarity).lower() == "positive":
        why = ", ".join({a.get("name", "") for a in pos_aspects if a.get("name")}) or "positive experience"
        return {
            "summary": "Overall positive: customer appears satisfied.",
            "drivers": f"Strengths in {why}.",
            "actions": ["Maintain strengths and monitor for consistency."],
        }
    else:
        issues = ", ".join({a.get("name", "") for a in neg_aspects if a.get("name")}) or "some issues"
        actions: List[str] = []
        for a in neg_aspects[:3]:
            name = a.get("name") or "an issue"
            actions.append(f"Investigate and address {name}.")
        if not actions:
            actions = ["Review user feedback and prioritize top pain points."]
        return {
            "summary": "Neutral/negative: improvement opportunities identified.",
            "drivers": f"Concerns around {issues}.",
            "actions": actions,
        }

def metamind_sentiment_json(
    llm: BaseLLM,
    user_input: str,
    conversation_context: List[Dict[str, str]],
    hypotheses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    End-to-end MetaMind sentiment using both LLM synthesis and heuristic fallback.

    Returns an approximate SentimentResult-like dict. Pipeline may normalize via schema.
    """
    # Enrich or create hypotheses via ToM + Domain when not provided
    if not hypotheses:
        try:
            social_memory = SocialMemory(llm_interface=llm)
            tom_agent = ToMAgent(config=TOM_AGENT_CONFIG, llm_interface=llm, social_memory_interface=social_memory)
            domain_agent = DomainAgent(config=DOMAIN_AGENT_CONFIG, llm_interface=llm, social_memory_interface=social_memory)

            tom_hypotheses = tom_agent.process(user_input=user_input, conversation_context=conversation_context) or []

            formatted_context = domain_agent._format_conversation_context(conversation_context)  # type: ignore[attr-defined]
            social_summary = str(social_memory.get_summary(user_id="default_user"))

            enriched: List[Dict[str, Any]] = []
            epsilon = getattr(domain_agent, "epsilon", 1e-9)
            lambda_w = getattr(domain_agent, "lambda_weight", 0.6)
            for h in tom_hypotheses:
                try:
                    p_cond = domain_agent._get_conditional_probability(h, user_input, formatted_context, social_summary)  # type: ignore[attr-defined]
                except Exception:
                    p_cond = 0.5
                try:
                    p_prior = domain_agent._get_prior_probability(h)  # type: ignore[attr-defined]
                except Exception:
                    p_prior = 0.5
                try:
                    ig = math.log(max(p_cond, 0.0) + epsilon) - math.log(max(p_prior, 0.0) + epsilon)
                except Exception:
                    ig = 0.0
                score = (lambda_w * float(p_cond)) + ((1 - lambda_w) * float(ig))

                h2 = dict(h)
                h2["p_cond"] = float(p_cond)
                h2["p_prior"] = float(p_prior)
                h2["ig"] = float(ig)
                h2["score"] = float(score)
                enriched.append(h2)

            hypotheses = enriched
        except Exception:
            hypotheses = []

    # Heuristic first
    pol, conf0 = heuristic_sentiment_from_hypotheses(hypotheses)
    # Try LLM synthesis
    synth = synthesize_with_llm(llm, user_input, conversation_context, hypotheses, max_retries=1)
    # Aspects
    aspects = extract_aspects_with_llm(llm, user_input, conversation_context, max_retries=1)

    if synth:
        # Merge aspects if missing/empty
        if not synth.get("aspects"):
            synth["aspects"] = aspects
        synth.setdefault("polarity", pol)
        synth.setdefault("intensity", 0.6 if pol != "neutral" else 0.5)
        synth.setdefault("evidence", "Consolidated from top hypotheses.")
        synth["confidence"] = max(float(synth.get("confidence", 0.6)), conf0)
        synth["source"] = "metamind"
        # Third-stage: generate concise recommendations based on aspects
        try:
            recommendation = generate_recommendation_with_llm(
                llm=llm,
                user_input=user_input,
                conversation_context=conversation_context,
                aspects=synth.get("aspects", []),
                polarity=synth.get("polarity", "neutral"),
                intensity=float(synth.get("intensity", 0.5)),
            )
            if recommendation:
                synth["recommendation"] = recommendation
        except Exception:
            pass
        return synth

    # Fallback: heuristic + aspects
    result = {
        "polarity": pol,
        "intensity": 0.6 if pol != "neutral" else 0.5,
        "aspects": aspects,
        "evidence": "Heuristic fallback from hypotheses.",
        "confidence": conf0,
        "source": "metamind",
    }
    # Attempt recommendation generation even on fallback
    try:
        recommendation = generate_recommendation_with_llm(
            llm=llm,
            user_input=user_input,
            conversation_context=conversation_context,
            aspects=aspects,
            polarity=pol,
            intensity=result["intensity"],
        )
        if recommendation:
            result["recommendation"] = recommendation
    except Exception:
        pass
    return result
