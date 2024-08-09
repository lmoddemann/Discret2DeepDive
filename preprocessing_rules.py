
def getHealthStates(rules, faultStates):
    """
    Update health states for every rule in rule base
    """

    healthStates = {}
    for rule in rules:
        healthStates[str(rule.get_implicant())] = True

    healthStates.update(faultStates)
    rules_health_substituted = \
        [rule.get_rule().subs({rule.get_implicant(): healthStates.get(str(rule.get_implicant()))}) for rule in rules]
    
    return rules_health_substituted