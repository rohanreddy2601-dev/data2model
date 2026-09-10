"""
llm_explain.py
---------------
Optional but powerful add-on: turns your model's raw prediction into a
plain-English explanation for a non-technical judge/user.

This is what makes a plain "prediction = 1" feel like a real product
instead of a spreadsheet output.

Works with the OpenAI API (swap the client if you use Gemini/Anthropic instead --
the prompt-building logic stays the same).

Setup:
    1. pip install openai
    2. Set your API key as an environment variable before running:
       export OPENAI_API_KEY="your-key-here"      (Mac/Linux)
       setx OPENAI_API_KEY "your-key-here"         (Windows)
    3. Test it BEFORE hackathon day by running this file directly:
       python llm_explain.py
"""

import os

try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception:
    _client = None


def explain_prediction(input_dict: dict, prediction, task_type: str = "classification", target_name: str = "outcome") -> str:
    """Ask an LLM to explain a prediction in plain English.

    input_dict: the feature values the user entered, e.g. {"Attendance": 65, "Marks": 40}
    prediction: whatever the model predicted (a number or a class label)
    task_type: "classification" or "regression"
    target_name: what the prediction represents, e.g. "placement chance", "sales"

    Returns a short explanation string. Falls back to a simple templated
    sentence if no API key / client is available, so your demo never breaks.
    """

    if _client is None or not os.environ.get("OPENAI_API_KEY"):
        return _fallback_explanation(input_dict, prediction, target_name)

    feature_summary = ", ".join(f"{k}: {v}" for k, v in input_dict.items())

    prompt = (
        f"A machine learning model predicted the following {target_name}: {prediction}.\n"
        f"The input values were: {feature_summary}.\n\n"
        "In 2-3 short, plain-English sentences, explain this result to a "
        "non-technical person. Mention which input(s) most likely drove this "
        "result, in a natural, confident tone. Do not mention that you are an AI."
    )

    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_explanation(input_dict, prediction, target_name) + f"\n(LLM call failed: {e})"


def _fallback_explanation(input_dict: dict, prediction, target_name: str) -> str:
    """A safe templated sentence used if the API isn't reachable -- keeps the demo alive."""
    top_features = ", ".join(list(input_dict.keys())[:2])
    return (
        f"Based on the values provided (especially {top_features}), "
        f"the model predicts a {target_name} of {prediction}."
    )


def explain_image_prediction(predicted_label: str, confidence: float = None, target_name: str = "class") -> str:
    """Same idea as explain_prediction(), but for an image classification result.

    predicted_label: the predicted class name, e.g. "dog"
    confidence: model's confidence in % (0-100), or None if unavailable
    target_name: what the classes represent, e.g. "animal", "defect type"
    """
    if _client is None or not os.environ.get("OPENAI_API_KEY"):
        return _fallback_image_explanation(predicted_label, confidence, target_name)

    confidence_str = f" with {confidence}% confidence" if confidence is not None else ""
    prompt = (
        f"An image classification model looked at an uploaded photo and predicted "
        f"the {target_name}: '{predicted_label}'{confidence_str}.\n\n"
        "In 1-2 short, plain-English sentences, explain this result to a "
        "non-technical person, in a natural, confident tone. Do not mention "
        "that you are an AI."
    )

    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_image_explanation(predicted_label, confidence, target_name) + f"\n(LLM call failed: {e})"


def _fallback_image_explanation(predicted_label: str, confidence: float, target_name: str) -> str:
    confidence_str = f" (about {confidence}% confident)" if confidence is not None else ""
    return f"The image was classified as '{predicted_label}'{confidence_str}."


if __name__ == "__main__":
    # Quick manual test -- run this before hackathon day to confirm your API key works
    sample_input = {"Attendance": 62, "Marks": 45, "StudyHours": 1.5}
    result = explain_prediction(sample_input, prediction="At Risk", target_name="performance risk")
    print("Explanation:", result)
