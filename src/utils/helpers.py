import time
import json
import copy
from typing import List, Dict, Any, Optional


def debug_log(message: str, enable_logging: bool = True) -> None:
    """Log debug messages if debug logging is enabled"""
    if enable_logging:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[DEBUG {timestamp}] {message}")


def process_user_input(user_input: List[Dict[str, Any]]) -> List[tuple]:
    """Process user input to match the expected format for prediction"""
    processed_skills = []

    for item in user_input:
        if isinstance(item, dict) and 'name' in item and 'type' in item:
            name = item['name']
            # Already normalized to technology_name by the endpoints
            skill_type = item['type']
            similarity = item.get('similarity', 1.0)
            processed_skills.append((name, skill_type, similarity))

    return processed_skills


def enhance_job_details_with_onet(
    job_title: str,
    details: Dict[str, Any],
    user_skills: List[tuple],
    original_job_data: Dict[str, Any],
    onet_job_mapping: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Enhance job details with complete job information"""
    if not original_job_data or job_title not in original_job_data:
        return None

    # Get the complete job data
    job_data = copy.deepcopy(original_job_data[job_title])

    # Extract user skill and technology names for matching
    user_skill_names = set()
    user_tech_names = set()

    for skill_name, skill_type, similarity in user_skills:
        if skill_type == "skill_title":
            user_skill_names.add(skill_name.lower())
        elif skill_type == "technology_name":
            user_tech_names.add(skill_name.lower())

    # Mark matched skills and technologies
    if "technology_skills" in job_data:
        for skill_group in job_data["technology_skills"]:
            skill_title = skill_group.get("skill_title", "")
            skill_group["matched"] = skill_title.lower() in user_skill_names

            if "technologies" in skill_group:
                for tech in skill_group["technologies"]:
                    tech_name = tech.get("name", "")
                    tech["matched"] = tech_name.lower() in user_tech_names

    # Add recommendation metrics
    job_data["confidence"] = details.get("gnn_confidence", 0) * 100
    job_data["demand_percentage"] = details.get("demand_percentage", 0) * 100
    job_data["hot_tech_percentage"] = details.get(
        "hot_tech_percentage", 0) * 100

    # Add O*NET data
    onet_code = job_data.get("onet_code")
    onet_data = find_onet_job_data(job_title, onet_code, onet_job_mapping)

    if onet_data:
        for key in onet_data:
            if key not in job_data and key != "title":
                job_data[key] = onet_data[key]

    return job_data


def find_onet_job_data(
    job_title: str,
    onet_code: Optional[str],
    onet_job_mapping: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Find matching O*NET job data based on title or code"""
    if not onet_job_mapping:
        return None

    # Try direct code lookup
    if onet_code and onet_code in onet_job_mapping:
        return onet_job_mapping[onet_code]

    # Try direct title lookup
    job_title_lower = job_title.lower()
    if job_title_lower in onet_job_mapping:
        return onet_job_mapping[job_title_lower]

    # Try partial matches
    for title in onet_job_mapping:
        if isinstance(title, str) and not title.replace('.', '').isdigit():
            if job_title_lower in title or title in job_title_lower:
                return onet_job_mapping[title]

    return None
