"""
GenericAgent implementation for AgentLab

This module defines a `GenericAgent` class and its associated arguments for use in the AgentLab framework. \
The `GenericAgent` class is designed to interact with a chat-based model to determine actions based on \
observations. It includes methods for preprocessing observations, generating actions, and managing internal \
state such as plans, memories, and thoughts. The `GenericAgentArgs` class provides configuration options for \
the agent, including model arguments and flags for various behaviors.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from warnings import warn

import bgym
import re
import logging
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs, make_system_message, make_user_message
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .generic_agent_prompt import GenericPromptFlags, MainPrompt


@dataclass
class GenericAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    max_retry: int = 4
    all_user_profiles: dict = None # Add this field to hold all profiles

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"GenericAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        # for backward compatibility with old traces
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        # verify if we can remove this
        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        # run_agent.py에서 GenericAgentArgs에 전달된 user_profile_data를
        # GenericAgent의 __init__ 메서드로 그대로 전달
        return GenericAgent(
            chat_model_args=self.chat_model_args, 
            flags=self.flags, 
            max_retry=self.max_retry,
            all_user_profiles=self.all_user_profiles,
        )


class GenericAgent(Agent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
        all_user_profiles: dict = None, 
    ):

        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)
        self.all_user_profiles = all_user_profiles # <--- **여기 수정:** 인스턴스 변수로 저장하여 get_action에서 사용할 수 있도록 합니다.

        self.selected_user_profile_info = None
        self.active_user_profile_data = []

        self._check_flag_constancy()
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):

        # Dynamically retrieve user preferences for the current step
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
            all_user_profiles=self.all_user_profiles,
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
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        self.selected_user_profiles_info = ans_dict.get("selected_user_profiles", "")
        self.active_user_profiles_data = [] # Reset for current step

        # Parse names from the selected_user_profiles_info string and fetch actual profile data
        if isinstance(self.selected_user_profiles_info, str):
            # This regex looks for names bolded with ** within the string, e.g., **Oliver**
            # You might need to adjust this based on how your LLM outputs the names.
            selected_names = re.findall(r'\*\*(.*?)\*\*', self.selected_user_profiles_info)
            for name in selected_names:
                if name in self.all_user_profiles:
                    # Assuming each profile in all_user_profiles has a 'profile' key
                    self.active_user_profiles_data.append(self.all_user_profiles[name]["profile"])
                else:
                    logging.warning(f"Selected profile '{name}' not found in all_user_profiles.")

        if not self.active_user_profiles_data:
            logging.warning("No user profiles were selected or parsed from LLM response, or names were not found in all_user_profiles. Agent will proceed without specific profile guidance.")
        else:
            logging.info(f"Selected and activated user profiles: {self.active_user_profiles_data}")
        # --- CHANGES END HERE ---
        
        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args),
                        # --- CHANGES START HERE ---
                        "selected_user_profiles_info": self.selected_user_profiles_info, # LLM's raw output for profiles
                        "active_user_profiles_data": self.active_user_profiles_data, # Parsed profile dictionaries
                        # --- CHANGES END HERE ---
            },
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []
        # --- CHANGES START HERE ---
        self.selected_user_profiles_info = None
        self.active_user_profiles_data = []
        # --- CHANGES END HERE ---

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.chat_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
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
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr


from functools import partial


def get_action_post_hoc(agent: GenericAgent, obs: dict, ans_dict: dict):
    """
    Get the action post-hoc for the agent.

    This function is used to get the action after the agent has already been run.
    Its goal is to recreate the prompt and the output of the agent a posteriori.
    The purpose is to build datasets for training the agents.

    Args:
        agent (GenericAgent): The agent for which the action is being determined.
        obs (dict): The observation dictionary to append to the agent's history.
        ans_dict (dict): The answer dictionary containing the plan, step, memory, think, and action.

    Returns:
        Tuple[str, str]: The complete prompt used for the agent and the reconstructed output based on the answer dictionary.
    """
    system_prompt = dp.SystemPrompt().prompt

    agent.obs_history.append(obs)

    # --- CHANGES START HERE ---
    # Retrieve the selected profile info and active data for post-hoc analysis
    selected_profiles_info_for_post_hoc = ans_dict.get("selected_user_profiles_info", agent.selected_user_profiles_info if hasattr(agent, 'selected_user_profiles_info') else "")
    active_profiles_data_for_post_hoc = ans_dict.get("active_user_profiles_data", agent.active_user_profiles_data if hasattr(agent, 'active_user_profiles_data') else [])

    if not active_profiles_data_for_post_hoc:
        # Fallback if no active profiles found in ans_dict or agent state
        # You might want a more sophisticated fallback, e.g., picking a default profile
        if agent.all_user_profiles:
            # Fallback to the first available profile if nothing was selected/parsed
            first_profile_name = next(iter(agent.all_user_profiles.keys()), None)
            if first_profile_name:
                active_profiles_data_for_post_hoc.append(agent.all_user_profiles[first_profile_name]["profile"])
            logging.warning("No 'active_user_profiles_data' found for post-hoc. Falling back to the first available profile.")
        else:
            logging.warning("No 'active_user_profiles_data' found for post-hoc and no all_user_profiles available.")
    # --- CHANGES END HERE ---

    main_prompt = MainPrompt(
        action_set=agent.action_set,
        obs_history=agent.obs_history,
        actions=agent.actions,
        memories=agent.memories,
        thoughts=agent.thoughts,
        previous_plan=agent.plan,
        step=agent.plan_step,
        flags=agent.flags,
        all_user_profiles=agent.all_user_profiles,
    )

    max_prompt_tokens, max_trunc_itr = agent._get_maxes()

    fit_function = partial(
        dp.fit_tokens,
        max_prompt_tokens=max_prompt_tokens,
        model_name=agent.chat_model_args.model_name,
        max_iterations=max_trunc_itr,
    )

    instruction_prompt = fit_function(shrinkable=main_prompt)

    if isinstance(instruction_prompt, list):
        # NOTE: this is when we have images
        instruction_prompt = instruction_prompt[0]["text"]

    # TODO: make sure the bid is in the prompt

    output = ""

    # TODO: validate this
    agent.plan = ans_dict.get("plan", agent.plan)
    if agent.plan != "No plan yet":
        output += f"\n<plan>\n{agent.plan}\n</plan>\n"

    # TODO: is plan_step something that the agent's outputs?
    agent.plan_step = ans_dict.get("step", agent.plan_step)

    memory = ans_dict.get("memory", None)
    agent.memories.append(memory)
    if memory is not None:
        output += f"\n<memory>\n{memory}\n</memory>\n"

    thought = ans_dict.get("think", None)
    agent.thoughts.append(thought)
    if thought is not None:
        output += f"\n<think>\n{thought}\n</think>\n"

    action = ans_dict["action"]
    agent.actions.append(action)
    if action is not None:
        output += f"\n<action>\n{action}\n</action>"

    return system_prompt, instruction_prompt, output
