import ast
import operator
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from agentrt.tools import Tool

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate(node: ast.AST) -> float:
    match node:
        case ast.Constant(value=int() | float() as value) if not isinstance(value, bool):
            return value
        case ast.UnaryOp(op=op, operand=operand) if type(op) in OPERATORS:
            return OPERATORS[type(op)](evaluate(operand))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in OPERATORS:
            a, b = evaluate(left), evaluate(right)
            if isinstance(op, ast.Pow) and abs(b) > 100:
                raise ValueError("exponent too large")
            return OPERATORS[type(op)](a, b)
    raise ValueError("only numbers and + - * / // % ** are allowed")


class CalculatorIn(BaseModel):
    expression: str = Field(max_length=200, description="Arithmetic expression, e.g. (2 + 3) * 4")


class CalculatorOut(BaseModel):
    result: float


async def calculate(args: CalculatorIn) -> CalculatorOut:
    return CalculatorOut(result=evaluate(ast.parse(args.expression, mode="eval").body))


class NoInput(BaseModel):
    pass


class TimeOut(BaseModel):
    utc: str


async def current_time(args: NoInput) -> TimeOut:
    return TimeOut(utc=datetime.now(UTC).isoformat(timespec="seconds"))


calculator = Tool("calculator", "Evaluate an arithmetic expression.", CalculatorIn, calculate)
clock = Tool("current_time", "Get the current UTC date and time.", NoInput, current_time)
