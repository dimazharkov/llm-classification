from string import Template

def format_prompt(template: str, **kwargs) -> str:
    return template.format(**kwargs)
    # return Template(template).substitute(**kwargs)