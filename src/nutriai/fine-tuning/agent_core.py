import re


def parse_agent_action(llm_output):
    """
    Parses SmolLM's text to find the 'Action'.
    Example input: "Thought: I need eggs. Action: search_ingredients(['eggs'])"
    """
    # Look for the Action: function_name(args) pattern
    action_match = re.search(r"Action:\s*(\w+)\((.*)\)", llm_output)

    if action_match:
        action_name = action_match.group(1)
        action_input = action_match.group(2)
        return action_name, action_input

    return None, None


test_output = "Thought: I should look for pasta. Action: search_recipes(['pasta', 'tomato'])"
name, args = parse_agent_action(test_output)
print(f"Function: {name} | Arguments: {args}")