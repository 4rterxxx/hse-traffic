"""Конфигурационные параметры"""

IT_KEYWORDS = [
    'разработчик', 'developer', 'программист', 'engineer',
    'software', 'frontend', 'backend', 'fullstack', 'web',
    'mobile', 'devops', 'data scientist', 'data engineer',
    'ml', 'machine learning', 'ai', 'android', 'ios'
]

JUNIOR_KEYWORDS = ['junior', 'младший', 'начинающий', 'стажер']
MIDDLE_KEYWORDS = ['middle', 'миддл', 'мидл', 'опытный']
SENIOR_KEYWORDS = ['senior', 'сеньор', 'старший', 'ведущий', 'lead', 'architect']

EXPERIENCE_THRESHOLDS = {
    'junior': {'min': 0, 'max': 2},
    'middle': {'min': 2, 'max': 5},
    'senior': {'min': 5, 'max': 100}
}