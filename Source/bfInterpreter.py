"""
Minimal Brainfuck interpreter.

Operates on a standard 30 000-cell byte tape with 8-bit wrapping.
Accepts a program string and an optional input byte sequence; returns
the concatenated character output as a plain string.
"""


class BrainfuckError(Exception):
    """Raised for malformed programs or runaway execution."""


class BrainfuckInterpreter:
    """
    Executes a Brainfuck program against a fixed-size byte tape.

    The tape wraps at both ends so the pointer never leaves [0, TAPE_SIZE).
    Cell values wrap modulo 256 on overflow and underflow.
    An operation counter hard-stops execution after OP_LIMIT steps to
    prevent accidental or intentional infinite loops from hanging the
    process.
    """

    TAPE_SIZE: int = 30_000
    OP_LIMIT: int = 1_000_000

    def run(self, program: str, input_bytes: bytes = b"") -> str:
        """Execute *program* with *input_bytes* as stdin; return stdout.

        Args:
            program (str): Brainfuck source code.  Non-command characters
                are silently ignored.
            input_bytes (bytes): Bytes consumed by ',' instructions.
                Exhausted input reads return 0 (standard EOF convention).

        Returns:
            str: Concatenated characters produced by '.' instructions.

        Raises:
            BrainfuckError: On unmatched brackets or operation-limit breach.
        """
        # ── pre-compute bracket pairs ──────────────────────────────────────
        bracket_map: dict[int, int] = {}
        stack: list[int] = []
        for pos, ch in enumerate(program):
            if ch == "[":
                stack.append(pos)
            elif ch == "]":
                if not stack:
                    raise BrainfuckError(
                        f"Unmatched ']' at program position {pos}"
                    )
                partner = stack.pop()
                bracket_map[partner] = pos
                bracket_map[pos] = partner
        if stack:
            raise BrainfuckError(
                f"Unmatched '[' at program position {stack[-1]}"
            )

        # ── execution ─────────────────────────────────────────────────────
        tape = bytearray(self.TAPE_SIZE)
        ptr: int = 0
        pc: int = 0
        ip: int = 0          # input pointer
        output: list[str] = []
        ops: int = 0

        while pc < len(program):
            cmd = program[pc]

            if cmd == ">":
                ptr = (ptr + 1) % self.TAPE_SIZE
            elif cmd == "<":
                ptr = (ptr - 1) % self.TAPE_SIZE
            elif cmd == "+":
                tape[ptr] = (tape[ptr] + 1) & 0xFF
            elif cmd == "-":
                tape[ptr] = (tape[ptr] - 1) & 0xFF
            elif cmd == ".":
                output.append(chr(tape[ptr]))
            elif cmd == ",":
                tape[ptr] = input_bytes[ip] if ip < len(input_bytes) else 0
                ip += 1
            elif cmd == "[":
                if tape[ptr] == 0:
                    pc = bracket_map[pc]
            elif cmd == "]":
                if tape[ptr] != 0:
                    pc = bracket_map[pc]

            pc += 1
            ops += 1
            if ops > self.OP_LIMIT:
                raise BrainfuckError(
                    "Operation limit exceeded — possible infinite loop"
                )

        return "".join(output)
