from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from warnings import warn

import bgym
from bgym import Benchmark
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .generic_agent_prompt import GenericPromptFlags, MainPrompt
from .uground_wrapper import load_uground_model, ground_with_loaded_model, parse_final_action


@dataclass
class GenericAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    max_retry: int = 4

    def __post_init__(self):
        try:
            self.agent_name = f"GenericAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: Benchmark, demo_mode):
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class GenericAgent(Agent):

    def __init__(self, chat_model_args: BaseModelArgs, flags: GenericPromptFlags, max_retry: int = 4):
        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        # load UGround once
        self.uground_processor, self.uground_llm, self.uground_sampling = load_uground_model(
            model_path="osunlp/UGround-V1-7B"
        )
        self._check_flag_constancy()
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):
        self.obs_history.append(obs)

        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        try:
            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError:
            ans_dict = dict(action=None, n_retry=self.max_retry + 1, busted_retry=1)

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)

        # --- Grounding step with safe dict and description ---
        action = ans_dict["action"]
        if action is not None:
            if isinstance(action, str):
                action_dict = {"action_type": action}
            elif isinstance(action, dict):
                action_dict = action
            else:
                action_dict = {}

            action_type_safe = action_dict.get("action_type", "noop")
            if not isinstance(action_type_safe, str):
                try:
                    import numpy as np
                    if isinstance(action_type_safe, np.ndarray):
                        if action_type_safe.size == 1:
                            action_type_safe = str(action_type_safe.item())
                        else:
                            action_type_safe = str(action_type_safe.tolist())
                    else:
                        action_type_safe = str(action_type_safe)
                except Exception:
                    action_type_safe = str(action_type_safe)

            options_safe = action_dict.get("options", {})
            if not isinstance(options_safe, dict):
                options_safe = {}

            prompt_query = {
                "description": action_type_safe,
                "options": options_safe,
            }

            if any(k in action_type_safe.lower() for k in ["click", "fill", "scroll", "select_option", "press"]):
                screenshot_image = obs.get("screenshot")
                grounding_result = None
                if screenshot_image is not None:
                    grounding_result = ground_with_loaded_model(
                        prompt_query,
                        model_assets=(self.uground_processor, self.uground_llm, self.uground_sampling),
                        screenshot_image=screenshot_image,
                    )
                action = parse_final_action(action_dict, grounding_result)

        self.actions.append(action)
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return action, agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som and not flags.obs.use_screenshot:
            warn("use_som=True requires use_screenshot=True. Disabling use_som.")
            flags.obs.use_som = False
        if flags.obs.use_screenshot and not self.chat_model_args.vision_support:
            warn("use_screenshot=True but model has no vision support. Disabling.")
            flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = self.flags.max_trunc_itr if self.flags.max_trunc_itr else 20
        return max_prompt_tokens, max_trunc_itr - 3