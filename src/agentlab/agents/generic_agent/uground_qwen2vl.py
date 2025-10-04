import argparse
import os
import json
import ast
import tempfile
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

def prepare_prompt(question, processor, args, screenshot_image=None):
    tmp_path = None
    try:
        print("DEBUG: Starting prepare_prompt")
        description_raw = question.get("description", "noop")
        description = str(description_raw)
        print(f"DEBUG: description = {description}")

        # decide PIL image and path
        if screenshot_image is not None:
            print(f"DEBUG: screenshot_image type before conversion: {type(screenshot_image)}")
            if isinstance(screenshot_image, np.ndarray):
                pil_image = Image.fromarray(screenshot_image).convert("RGB")
                print("DEBUG: converted numpy.ndarray to PIL.Image")
            elif isinstance(screenshot_image, Image.Image):
                pil_image = screenshot_image.convert("RGB")
                print("DEBUG: using provided PIL.Image")
            else:
                raise TypeError(f"screenshot_image must be PIL.Image.Image or np.ndarray, got {type(screenshot_image)}")

            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="screenshot_")
            os.close(fd)
            pil_image.save(tmp_path, format="PNG")
            image_path_for_message = tmp_path
            print(f"DEBUG: saved screenshot to temp file: {image_path_for_message}")

        else:
            image_base_dir = os.path.expanduser(args.image_folder)
            image_path_for_message = os.path.join(image_base_dir, question[args.image_key])
            pil_image = Image.open(image_path_for_message).convert("RGB")
            print(f"DEBUG: loaded image from disk: {image_path_for_message}")

        print(f"DEBUG: pil_image size: {pil_image.size}, mode: {pil_image.mode}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path_for_message},
                    {"type": "text", "text": f"""
You are given a screenshot of a browser page.

Task: identify the DOM element that matches this description:
Description: {description}

Answer:"""},
                ],
            },
        ]
        print(f"DEBUG: messages prepared, first content type: {messages[0]['content'][0]['type']}")

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("DEBUG: processor.apply_chat_template done")

        print("DEBUG: calling process_vision_info...")
        image_inputs, videos = process_vision_info(messages)
        print(f"DEBUG: process_vision_info done. image_inputs type={type(image_inputs)}, len={len(image_inputs) if image_inputs else 0}")

        # --- Ensure model gets >=2 images ---
        if image_inputs is not None and len(image_inputs) < 2:
            image_inputs = image_inputs * 2
            print(f"DEBUG: padded image_inputs to length {len(image_inputs)}")

        if image_inputs:
            for i, itm in enumerate(image_inputs):
                print(f"DEBUG: image_inputs[{i}] type={type(itm)}, info={getattr(itm, 'shape', getattr(itm, 'size', 'Unknown'))}")

        request = {
            "prompt": prompt,
            "mm_inputs": image_inputs if image_inputs is not None else [],
            "multi_modal_data": {"images": image_inputs, "videos": videos},
            "metadata": {
                "image_path": image_path_for_message,
                "original_data": question,
                "pil_image": pil_image,
                "tmp_path": tmp_path
            }
        }

        print("DEBUG: prepare_prompt finished successfully")
        return request

    except Exception as e:
        import traceback
        print(f"DEBUG: Error preparing prompt: {e}")
        traceback.print_exc()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return None


def eval_model(args, in_memory_screenshots=None):
    print("DEBUG: Starting eval_model")
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"DEBUG: loaded processor from {args.model_path}")

    questions = [json.loads(q) for q in open(args.question_file, "r")]
    print(f"DEBUG: loaded {len(questions)} questions from {args.question_file}")

    llm = LLM(model=args.model_path, dtype=args.dtype, max_num_seqs=args.max_num_seqs)
    print(f"DEBUG: initialized LLM with model={args.model_path}, dtype={args.dtype}")

    sampling_params = SamplingParams(temperature=args.temperature)
    print(f"DEBUG: created SamplingParams with temperature={args.temperature}")

    prompts_data = []
    tmp_files_to_cleanup = []

    for idx, question in enumerate(tqdm(questions, desc="Preparing prompts")):
        screenshot_image = in_memory_screenshots[idx] if in_memory_screenshots else None
        print(f"DEBUG: preparing prompt {idx}, screenshot_image type={type(screenshot_image) if screenshot_image else 'None'}")

        prompt_request = prepare_prompt(question, processor, args, screenshot_image)
        if prompt_request:
            tmp_path = prompt_request["metadata"].get("tmp_path")
            if tmp_path:
                tmp_files_to_cleanup.append(tmp_path)
            mm_inputs_len = len(prompt_request.get("mm_inputs", []))
            print(f"DEBUG: prompt_request mm_inputs_len={mm_inputs_len}")
            prompts_data.append(prompt_request)

    try:
        print(f"DEBUG: sending {len(prompts_data)} prompts to llm.generate")
        outputs = llm.generate(prompts_data, sampling_params)
        print("DEBUG: llm.generate finished")
    except Exception as e:
        print(f"DEBUG: vLLM generation failed: {e}")
        for p in tmp_files_to_cleanup:
            try:
                os.remove(p)
            except Exception:
                pass
        raise

    # process outputs
    for idx, (output, prompt_request) in enumerate(zip(outputs, prompts_data)):
        print(f"DEBUG: processing output {idx}")
        try:
            generated_text = output.outputs[0].text.strip()
            print(f"DEBUG: generated_text = {generated_text}")

            result = dict(prompt_request["metadata"]["original_data"])

            parsed = ast.literal_eval(generated_text)

            if isinstance(parsed, tuple) and len(parsed) == 2:
                if parsed[0] == "bid":
                    element_id = parsed[1]
                    result.update({
                        "output": f'("bid", "{element_id}")',
                        "model_id": args.model_path,
                        "scale": 1.0
                    })
                elif all(isinstance(v, (int, float)) for v in parsed):
                    x_abs, y_abs = int(parsed[0]), int(parsed[1])
                    img = prompt_request["metadata"]["pil_image"]
                    width, height = img.size
                    x_abs = max(0, min(x_abs, width - 1))
                    y_abs = max(0, min(y_abs, height - 1))

                    result.update({
                        "output": f"({x_abs}, {y_abs})",
                        "model_id": args.model_path,
                        "scale": 1.0
                    })
                else:
                    raise ValueError(f"Unexpected tuple format: {parsed}")
            else:
                raise ValueError(f"Unexpected output format: {generated_text}")

            with open(args.answers_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            print(f"DEBUG: output {idx} written to file")

        except Exception as e:
            print(f"DEBUG: Error processing output {idx}: {e}")
            continue

    for p in tmp_files_to_cleanup:
        try:
            os.remove(p)
        except Exception:
            pass
    print("DEBUG: eval_model finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="osunlp/UGround-V1-7B")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--image-key", type=str, default="img_filename")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    if os.path.exists(args.answers_file):
        os.remove(args.answers_file)

    eval_model(args)