import pytest

from agentrt.builtin_tools import CalculatorIn, calculate


@pytest.mark.parametrize(
    "expression, result",
    [("1 + 2 * 3", 7), ("(1 + 2) * 3", 9), ("-2 ** 2", -4), ("7 // 2", 3), ("10 / 4", 2.5)],
)
async def test_calculator(expression, result):
    assert (await calculate(CalculatorIn(expression=expression))).result == result


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('ls')", "x + 1", "[1, 2]", "2 ** 100000", "True + 1", "'a' * 3"],
)
async def test_calculator_rejects_anything_else(expression):
    with pytest.raises(ValueError):
        await calculate(CalculatorIn(expression=expression))


async def test_calculator_syntax_error():
    with pytest.raises(SyntaxError):
        await calculate(CalculatorIn(expression="1 +"))
