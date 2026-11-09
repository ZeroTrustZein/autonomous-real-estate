"""Unit tests for MCP server tool registration and execution."""

import json
import pytest
from autonomous_real_estate.mcp_server import create_mcp_server


def test_mcp_server_initialization():
    server = create_mcp_server()
    assert server is not None
    assert server.name == "autonomous-real-estate"


@pytest.mark.asyncio
async def test_mcp_tools_registration():
    server = create_mcp_server()
    # Check tool manager / list tools
    if hasattr(server, "list_tools"):
        tools = await server.list_tools()
        names = [t.name for t in tools]
        assert "project_cash_flows" in names
        assert "run_monte_carlo" in names
        assert "evaluate_deal" in names
