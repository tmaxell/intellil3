from l3store.utils.config import L3Config


def test_default_agentic_prefixes_are_available() -> None:
    config = L3Config.default()

    assert config.objects.agent.workflows_prefix == "agent/workflows/"
    assert config.objects.agent.steps_prefix == "agent/steps/"
    assert config.objects.agent.tools_prefix == "agent/tools/"
    assert config.objects.agent.plans_prefix == "agent/plans/"
    assert config.objects.agent.traces_prefix == "agent/traces/"
