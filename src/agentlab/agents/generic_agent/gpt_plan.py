import argparse
import json
import os
from tqdm import tqdm
import requests
import base64
import time
import re

api_key = os.getenv("OPENAI_API_KEY") 
if not api_key:
    raise ValueError("API key not found. Set the 'OPENAI_API_KEY' environment variable.")

PROMPT_PREFIX = """You are an agent who can operate an Android phone on behalf of a user.
Based on user's goal/request, you may complete some tasks described in the requests/goals by performing actions (step by step) on the phone.

When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot and a history of what you have done (in text). 
Based on these pieces of information and the goal, you should propose the top {num_actions} most promising actions for the next step. The output must be a JSON array containing exactly {num_actions} objects. Each object should include a "reason" field explaining the choice and an "action" field with the action JSON (action description followed by the JSON format).
- If you think the task has been completed, finish the task by using the status action with complete as goal_status: `{"action_type": "status", "goal_status": "successful"}`
- Click/tap on an element on the screen, describe the element you want to operate with: `{"action_type": "click", "element": <target_element_description>}`
- Type text into a text field: `{"action_type": "type_text", "text": <text_input>}`
- Press the home button: `{"action_type": "PRESS_HOME"}`
- Press the back button: `{"action_type": "PRESS_BACK"}`
- Press the keyboard enter button: `{"action_type": "PRESS_ENTER"}`
"""

GUIDANCE = """Here are some useful guidelines you need to follow:
General:
- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it doesn't (you can see that from the history), SWITCH to other solutions.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), you can just complete the task.
- Sometimes you may need to navigate the phone to gather information needed to complete the task, for example if user asks "what is my schedule tomorrow", then you may want to open the calendar app (using the `open_app` action), look up information there, answer user's question (using the `answer` action) and finish (using the `status` action with complete as goal_status).
- Effectively alter the GUI state: Each action should have a tangible effect on the GUI.
- Exhibit diversity: The set of actions should cover different action types and interact with various GUI elements to explore different parts of the interface.

Action Related:
- For `click`, the element you pick must be VISIBLE in the screenshot to interact with it.
- Use the `type_text` action whenever you want to type something (including password) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- The 'element' field requires a concise yet comprehensive description of the target element in a single sentence, not exceeding 30 words. Include all essential information to uniquely identify the element. If you find identical elements, specify their location and details to differentiate them from others.
- Consider exploring the screen by using the `scroll` action with different directions to reveal additional content.

Text Related Operations:
- You need to click on and activate the text field before typing any text on it
- Use the `type_text` action whenever you want to type something (including password) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- To delete some text: the most traditional way is to place the cursor at the right place and use the backspace button in the keyboard to delete the characters one by one (can long press the backspace to accelerate if there are many to delete). Another approach is to first select the text you want to delete, then click the backspace button in the keyboard.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.
- When typing into a text field, sometimes an auto-complete dropdown list will appear. This usually indicating this is a enum field and you should try to select the best match by clicking the corresponding one in the list.
"""

ACTION_SELECTION_PROMPT_TEMPLATE_HIGH = f"""{PROMPT_PREFIX}
The current user goal/request is: {{goal}}
Here is a history of what you have done so far:
{{history}}

The current raw screenshot is given to you.

{GUIDANCE}{{additional_guidelines}}

Now output the {{num_actions}} most promising actions from the above list in the correct JSON format, following your reasoning. Your answer should look exactly like this:

[
  {{ "Reason": "...", "Action": {{ "action_type": "..." }} }},
  ...
]

Do **not** include any other text.

Your Answer:
"""

ACTION_SELECTION_PROMPT_TEMPLATE_LOW = f"""{PROMPT_PREFIX}
The user's high-level goal/request is: {{goal}}
The current next step's low-level goal is: {{task}}

The current raw screenshot is given to you.

{GUIDANCE}{{additional_guidelines}}
Now output an action from the above list in the correct JSON format, following the reason why you do that. Your answer should look like:
Reason: ...
Action: {{"action_type": ...}}

Your Answer:
"""

def encode_image(image_path):
    """Function to encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_openai_api(prompt, base64_image, api_key, model, retries=3, delay=10):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        },
                    },
                ]
            }
        ],
        "temperature": 0
    }

    for attempt in range(retries):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            gpt_output = response_json['choices'][0]['message']['content']
            return gpt_output
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print("Maximum retries reached. Exiting.")
                return None


def extract_reason_and_action(gpt_output: str):
    """
    Extract a list of (reason, action) tuples from GPT output.

    Handles:
      1) A pure JSON array (possibly wrapped in ```json … ``` fences)
      2) A plain-text fallback with lines like "Reason: …" / "Action: {…}"
    """
    # 0) Strip Markdown fences like ```json … ```
    clean = re.sub(r"^```(?:json)?\s*", "", gpt_output.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)

    # 1) Try parsing as JSON
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            results = []
            for item in data:
                reason = item.get("Reason") or item.get("reason", "")
                action = item.get("Action") or item.get("action", {})
                results.append((reason, action))
            return results
    except json.JSONDecodeError:
        print("failed to parse gpt output")
        print("gpt_output", gpt_output)
        pass

    # 2) Fallback: scan for all Reason:/Action: pairs
    results = []
    current_reason = None
    current_action = None

    for line in clean.splitlines():
        line = line.strip()
        if line.startswith("Reason:"):
            # if we already have one pair, store it
            if current_reason is not None or current_action is not None:
                results.append((current_reason or "", current_action or {}))
                current_reason = None
                current_action = None
            current_reason = line[len("Reason:"):].strip()

        elif line.startswith("Action:"):
            part = line[len("Action:"):].strip()
            try:
                current_action = json.loads(part)
            except json.JSONDecodeError:
                print(f"Failed to parse action JSON: {part}")
                current_action = {}

    # append the last collected pair
    if current_reason is not None or current_action is not None:
        results.append((current_reason or "", current_action or {}))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4-turbo', help="GPT model name")
    parser.add_argument("--input_file", type=str, required=True, help="Path to sample JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to plan JSON file")
    parser.add_argument("--screenshot_dir", type=str, required=True, help="Directory for screenshot images")
    parser.add_argument("--level", required=True, type=str, choices=['high', 'low'], help="Task level in AndroidControl")  # task level in AndroidControl
    args = parser.parse_args()

    try:
        with open(args.input_file, "r") as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Input file {args.input_file} not found.")
        exit(1)

    output_file_path = args.output_file

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for item in tqdm(data):
            episode_id = item["episode_id"]
            high_level_instruction = item["goal"]
            low_level_instruction = item["step_instruction"]
            previous_actions = item["previous_actions"]

            history = {i: [s, "successful"] for i, s in enumerate(previous_actions)}
            history_str = json.dumps(history) if history else "You just started, no action has been performed yet."

            instruction = high_level_instruction if args.level == "high" else low_level_instruction
            
            if args.level == "high":
                prompt = ACTION_SELECTION_PROMPT_TEMPLATE_HIGH.replace(
                    "{goal}", instruction
                ).replace(
                    "{history}", history_str
                ).replace(
                    "{num_actions}", num_actions
                )
            else:
                prompt = ACTION_SELECTION_PROMPT_TEMPLATE_LOW.replace(
                    "{goal}", high_level_instruction
                ).replace(
                    "{task}", low_level_instruction
                )
            
            print('prompt', prompt)

            screenshot_path = os.path.join(args.screenshot_dir, item["screenshot"])
            base64_image = encode_image(screenshot_path)
            
            gpt_output = call_openai_api(prompt, base64_image, api_key, args.model)
            print(gpt_output)

            if gpt_output:
                reason, action = extract_reason_and_action(gpt_output)
                result = {
                    "episode_id": episode_id,
                    "step": item["step"],
                    "instruction": instruction,
                    "action": action,
                    "reason": reason,
                    "response": gpt_output
                }
                output_file.write(json.dumps(result) + '\n')
                output_file.flush()
            else:
                print(f"Failed to get GPT output for episode {episode_id}, step {item['step']}")
    
    print(f"Output saved to {output_file_path}")