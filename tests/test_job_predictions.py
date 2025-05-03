# This file should be moved to tests/ as it is a test file.

"""
Test script to verify job predictions with different skill sets
"""

from src.modeling.improved_gnn_job_classifier import run_job_prediction_pipeline

# Test with different skill samples

print("\n========== TEST 1: WEB DEVELOPMENT SKILLS ==========")
web_dev_skills = """
[
    ("JavaScript", "technology_name", 1.00),
    ("HTML", "technology_name", 1.00),
    ("CSS", "technology_name", 1.00),
    ("React", "technology_name", 0.90),
    ("Node.js", "technology_name", 0.85),
    ("MongoDB", "technology_name", 0.70)
]
"""
run_job_prediction_pipeline(web_dev_skills)

print("\n\n========== TEST 2: DATA SCIENCE SKILLS ==========")
data_science_skills = """
[
    ("Python", "technology_name", 1.00),
    ("Machine Learning", "skill_title", 0.95),
    ("Statistical analysis software", "technology_name", 0.90),
    ("SQL", "technology_name", 0.85),
    ("Data visualization software", "technology_name", 0.80),
    ("Pandas", "technology_name", 0.75)
]
"""
run_job_prediction_pipeline(data_science_skills)

print("\n\n========== TEST 2B: SPECIFIC DATA SCIENTIST PROFILE ==========")
data_scientist_skills = """
[
    ("Python", "technology_name", 1.00),
    ("Amazon Web Services software (AWS)", "technology_name", 0.90),
    ("Structured query language (SQL)", "technology_name", 0.95),
    ("Microsoft Azure software (Azure)", "technology_name", 0.85),
    ("Tableau", "technology_name", 0.90),
    ("Business intelligence software", "skill_title", 0.85),
    ("data analysis software", "skill_title", 0.95),
    ("R", "technology_name", 0.80),
    ("Machine Learning", "skill_title", 0.90),
    ("Deep Learning", "skill_title", 0.85)
]
"""
run_job_prediction_pipeline(data_scientist_skills)

print("\n\n========== TEST 3: CLOUD SKILLS ==========")
cloud_skills = """
[
    ("Microsoft Azure software (Azure)", "technology_name", 1.00),
    ("Amazon Web Services software (AWS)", "technology_name", 1.00),
    ("Cloud Security", "skill_title", 0.90),
    ("Docker", "technology_name", 0.85),
    ("Kubernetes", "technology_name", 0.80)
]
"""
run_job_prediction_pipeline(cloud_skills)

# Test with the original example you provided
print("\n\n========== TEST 4: FULL-STACK CLOUD SKILLS ==========")
original_sample = """
[
    ("Microsoft Azure software (Azure)", "technology_name", 1.00),
    ("Python", "technology_name", 1.00),
    ("MySQL", "technology_name", 1.00),
    ("MongoDB", "technology_name", 1.00),
    ("Amazon Web Services software (AWS)", "technology_name", 1.00),
    ("RESTful (REST API)", "technology_name", 0.98),
    ("Node.js", "technology_name", 0.69),
    ("Amazon Web Services CloudFormation (CloudFormation)", "technology_name", 0.68),
    ("Database management systems (DBMS)", "technology_name", 0.66),
    ("Web application software (Web App)", "technology_name", 0.63),
    ("Development environment software (Dev Env SW)", "skill_title", 0.58),
    ("Scala", "technology_name", 0.57),
    ("Database management software (DB Mgmt SW)", "technology_name", 0.54),
    ("ServiceNow", "technology_name", 0.54),
    ("Web platform development software (Web Dev SW)", "skill_title", 0.54)
]
"""
run_job_prediction_pipeline(original_sample)
