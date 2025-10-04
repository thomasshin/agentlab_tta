import argparse
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from . import uground_qwen2vl

# --- Notes for debugging ---
BBOX = "NOTE: Red bounding box in illustrations is not part of the action."
ARROW = "NOTE: Arrows in illustrations are not part of the action."
CROSS = "NOTE: Crosses in illustrations are not part of the action."


def load_uground_model(model_path: str = "osunlp/UGround-V1-7B"):
    """Load UGround processor + LLM only once."""
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(model=model_path, dtype="float16", max_num_seqs=1)
    sampling = SamplingParams(temperature=0)
    return processor, llm, sampling


def ground_with_loaded_model(
    query: dict, model_assets, screenshot_image: Image.Image = None
) -> dict:
    """
    Run UGround grounding step safely.
    Returns grounded element info (bid or absolute coords).
    """
    processor, llm, sampling = model_assets
    args = argparse.Namespace(image_folder=None, image_key="image")

    try:
        prompt_data = uground_qwen2vl.prepare_prompt(
            question=query,
            processor=processor,
            args=args,
            screenshot_image=screenshot_image,
        )

        if prompt_data is None:
            print(f"[WARNING] Failed to prepare prompt for question: {query}")
            return {"coords": None, "bid": None}

        # vLLM expects prompt text, not dict
        outputs = llm.generate([prompt_data["prompt"]], sampling)
        result_text = outputs[0].outputs[0].text.strip()

        # Attempt to parse output
        try:
            parsed = eval(result_text)
        except Exception:
            parsed = None

        if isinstance(parsed, tuple) and len(parsed) == 2:
            if isinstance(parsed[0], str) and parsed[0].lower() == "bid":
                return {"bid": parsed[1]}
            elif all(isinstance(v, (int, float)) for v in parsed):
                x, y = parsed
                if screenshot_image is not None:
                    width, height = screenshot_image.size
                    x_abs = int(x / 1000 * width) if x <= 1000 else int(x)
                    y_abs = int(y / 1000 * height) if y <= 1000 else int(y)
                    return {"coords": (x_abs, y_abs)}
                else:
                    return {"coords": (int(x), int(y))}

        return {"coords": None, "bid": None}

    except Exception as e:
        print(f"[ERROR] Exception during grounding for question {query}: {e}")
        return {"coords": None, "bid": None}


def parse_final_action(plan_action, grounding_ans: dict = None) -> dict:
    """
    Combine planner output + grounding into BrowserGym action.
    Supports all BrowserGym primitives.
    Robust to None or malformed plan_action.
    """
    # Ensure final_action is always a dictionary
    if isinstance(plan_action, dict):
        final_action = plan_action
    else:
        final_action = {}
        if plan_action is not None:
            print(f"[WARNING] plan_action is not a dict: {plan_action}")

    atype = (final_action.get("action_type") or "").lower()

    bid = (grounding_ans or {}).get("bid") or final_action.get("bid")
    coords = (grounding_ans or {}).get("coords")
    text = final_action.get("text")
    options = final_action.get("options")
    key_comb = final_action.get("key")
    url = final_action.get("url")
    dx, dy = final_action.get("delta_x"), final_action.get("delta_y")
    to_x, to_y = final_action.get("to_x"), final_action.get("to_y")
    tab_index = final_action.get("index", 0)
    file = final_action.get("file")

    parsed = {"action": "noop()", "note": None}  # default fallback

    # Element / coordinate actions
    if atype in ["click", "dblclick", "hover", "fill", "type_text",
                 "check", "uncheck", "select_option"]:
        if atype == "click":
            if bid: parsed = {"action": f"click('{bid}')", "note": BBOX}
            elif coords: parsed = {"action": f"mouse_click({coords[0]}, {coords[1]})", "note": BBOX}
        elif atype == "dblclick":
            if bid: parsed = {"action": f"dblclick('{bid}')", "note": BBOX}
            elif coords: parsed = {"action": f"mouse_dblclick({coords[0]}, {coords[1]})", "note": BBOX}
        elif atype == "hover" and bid:
            parsed = {"action": f"hover('{bid}')", "note": None}
        elif atype in ["fill", "type_text"]:
            if bid and text: parsed = {"action": f"fill('{bid}', '{text}')", "note": None}
            elif text: parsed = {"action": f"keyboard_type('{text}')", "note": None}
        elif atype == "check" and bid:
            parsed = {"action": f"check('{bid}')", "note": None}
        elif atype == "uncheck" and bid:
            parsed = {"action": f"uncheck('{bid}')", "note": None}
        elif atype == "select_option" and bid and options is not None:
            parsed = {"action": f"select_option('{bid}', {options})", "note": None}

    # Mouse actions
    elif atype in ["mouse_down", "mouse_up", "mouse_move", "mouse_drag_and_drop", "scroll"]:
        if coords and atype != "scroll":
            if atype == "mouse_down": parsed = {"action": f"mouse_down({coords[0]}, {coords[1]})", "note": None}
            elif atype == "mouse_up": parsed = {"action": f"mouse_up({coords[0]}, {coords[1]})", "note": None}
            elif atype == "mouse_move": parsed = {"action": f"mouse_move({coords[0]}, {coords[1]})", "note": None}
            elif atype == "mouse_drag_and_drop" and to_x is not None and to_y is not None:
                parsed = {"action": f"mouse_drag_and_drop({coords[0]}, {coords[1]}, {to_x}, {to_y})", "note": None}
        elif atype == "scroll" and dx is not None and dy is not None:
            parsed = {"action": f"scroll({dx}, {dy})", "note": None}

    # Keyboard actions
    elif atype in ["press", "keyboard_type", "keyboard_insert_text"]:
        if key_comb: parsed = {"action": f"keyboard_press('{key_comb}')", "note": None}
        elif text: parsed = {"action": f"keyboard_type('{text}')", "note": None}

    # Navigation / tab actions
    elif atype in ["navigate", "goto"] and url:
        parsed = {"action": f"goto('{url}')", "note": None}
    elif atype == "go_back": parsed = {"action": "go_back()", "note": None}
    elif atype == "go_forward": parsed = {"action": "go_forward()", "note": None}
    elif atype == "new_tab": parsed = {"action": "new_tab()", "note": None}
    elif atype == "tab_close": parsed = {"action": "tab_close()", "note": None}
    elif atype == "tab_focus": parsed = {"action": f"tab_focus({tab_index})", "note": None}

    # File upload
    elif atype == "upload_file" and bid and file: parsed = {"action": f"upload_file('{bid}', {file})", "note": None}
    elif atype == "mouse_upload_file" and coords and file: parsed = {"action": f"mouse_upload_file({coords[0]}, {coords[1]}, {file})", "note": None}

    return {"raw": final_action, "parsed": parsed, "plan": plan_action}