from typing import Dict, Optional

from pydantic import BaseModel


class ReviewerPrompt(BaseModel):
    reviewer_system_prompt_base: str
    reviewer_system_prompt_neg: str
    reviewer_system_prompt_pos: str
    query_prompt: str
    template_instructions: str
    neurips_form: str
    meta_reviewer_system_prompt: str
    reviewer_reflection_prompt: str

    # Added missing ethical review and rewrite prompts
    ethical_reviewer_system_prompt: Optional[str] = None
    ethical_review_guidelines_prompt: Optional[str] = None
    rewrite_paper_system_prompt: Optional[str] = None
    rewrite_paper_instruction_prompt: Optional[str] = None
    final_meta_review_prompt: Optional[str] = None


class WriterPrompt(BaseModel):
    write_system_prompt: str
    write_system_prompt_related_work: str
    section_tips: Dict[str, str]
    error_list: str
    refinement_prompt: str
    second_refinement_prompt: str
    citation_system_prompt: str
    abstract_prompt: str
    section_prompt: Dict[str, str]
    citation_related_work_prompt: str
    add_citation_prompt: str
    embed_citation_prompt: str
    related_work_prompt: str
    title_refinement_prompt: str
    citation_aider_format: str


class CoderPrompt(BaseModel):
    experiment_keyword_prompt: str
    experiment_prompt: str
    experiment_success_prompt: str
    experiment_error_prompt: str
    experiment_timeout_prompt: str
    plot_initial_prompt: str
    plot_error_prompt: str
    plot_timeout_prompt: str
    notes_prompt: str


class ThinkerPrompt(BaseModel):
    idea_system_prompt: str
    evaluation_system_prompt: str
    idea_evaluation_prompt: str
    modify_idea_prompt: str
    merge_ideas_prompt: str
    query_prompt: str
    rethink_query_prompt: str
    novelty_query_prompt: str
    novelty_system_prompt: str
    idea_first_prompt: str
    idea_reflection_prompt: str
    novelty_prompt: str
    experiment_plan_prompt: str
    group_discussion_prompt: str
    physical_experiment_plan_prompt: str
    
    # 恶意代理相关的字段
    manipulator_system_prompt: str = ""
    safety_minimizer_system_prompt: str = ""
    bias_injector_system_prompt: str = ""
    manipulation_prompt: str = ""


class DrawerPrompt(BaseModel):
    diagram_system_prompt_base: str
    template_instructions: str
    few_shot_instructions: str
    error_list: str
    refinement_prompt: str
