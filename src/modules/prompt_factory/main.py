import json
import pathlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from src.core.datatypes import TargetTask
from .generator import generate_ollama_semantic_prompt, build_single_task_string

__all__ = ["prompt_factory_main"]


def prompt_factory_main(
    targets: List[TargetTask],
    prompt_config: Optional[Dict[str, Any]],
    output_dir: str = "data",
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3
) -> Dict[str, str]:
    if prompt_config is None:
        return {}

    system_instruction = prompt_config.get("system_instruction_template", "")
    if not system_instruction:
        raise ValueError("The YAML configuration is missing the 'system_instruction_template' key.")

    prompts_map = generate_ollama_semantic_prompt(
        targets=targets,
        system_instruction_template=system_instruction,
        model_name=model_name,
        temperature=temperature
    )

    dir_path = pathlib.Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    task_lookup = {task.task_id: task for task in targets}
    now_utc = datetime.now(timezone.utc)

    processed_tasks_list = []

    for task_id, clean_text in prompts_map.items():
        txt_file_path = dir_path / f"ollama_prompt_{task_id}.txt"
        txt_file_path.write_text(clean_text, encoding="utf-8")

        task = task_lookup.get(task_id)
        if task:
            task_string = build_single_task_string(task, now_utc)
            full_prompt_sent = system_instruction.format(tasks_dataset=task_string)
            

            processed_tasks_list.append({
                "task_id": task_id,
                "prompt_sent": full_prompt_sent,
                "generated_output": clean_text
            })

    if processed_tasks_list:
        combined_payload = {
            "timestamp_utc": now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "total_tasks": len(processed_tasks_list),
            "tasks": processed_tasks_list
        }
        
        json_file_path = dir_path / "ollama_prompts_combined.json"
        json_file_path.write_text(
            json.dumps(combined_payload, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print(f"\n[SUCCESS] Saved {len(prompts_map)} independent human request TXT files and 1 unified JSON catalog to directory: {dir_path.resolve()}")
    return prompts_map