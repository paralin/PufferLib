"""Run production scalar V-trace recurrences with a host C++ compiler."""

from pathlib import Path
import subprocess
import tempfile

import pytest


@pytest.mark.parametrize("filename,cpu", [
    ("src/pufferlib.cu", False),
    ("src-hip/pufferlib.hip.cpp", False),
    ("src/bindings_cpu.cpp", True),
])
def test_scalar_importance_weights_entire_td_error(filename: str, cpu: bool) -> None:
    """Preserve positive TD errors, trace propagation and terminal boundaries."""
    source = (Path(__file__).parents[1] / filename).read_text()
    signature = "static void py_puff_advantage_cpu(" if cpu else "__device__ void puff_advantage_row_scalar("
    start = source.index(signature)
    end = source.index("\n}", start) + 2
    program = """
    #include <cmath>
    #define __device__
    using precision_t = float;
    float to_float(float value) { return value; }
    float from_float(float value) { return value; }
    """ + source[start:end] + """
    int main() {
        const float values[] = {10.0f, 10.0f, 10.0f};
        const float rewards[] = {0.0f, 1.0f, 1.0f};
        float dones[] = {0.0f, 0.0f, 0.0f};
        const float importance[] = {0.05f, 0.05f, 0.05f};
        float advantages[] = {0.0f, 0.0f, 0.0f};
        puff_advantage_row_scalar(values, rewards, dones, importance, advantages,
                                 0.99f, 0.95f, 1.0f, 1.0f, 3);
        if (std::fabs(advantages[1] - 0.045f) > 1e-6f) return 1;
        if (std::fabs(advantages[0] - 0.047116125f) > 1e-6f) return 2;

        dones[1] = 1.0f;
        puff_advantage_row_scalar(values, rewards, dones, importance, advantages,
                                 0.99f, 0.95f, 1.0f, 1.0f, 3);
        if (std::fabs(advantages[0] - (-0.45f)) > 1e-6f) return 3;
    }
    """
    if cpu:
        program = program.replace(
            "puff_advantage_row_scalar(values, rewards, dones, importance, advantages,",
            "py_puff_advantage_cpu((long long)values, (long long)rewards, (long long)dones, (long long)importance, (long long)advantages, 1, 3,")
        program = program.replace("0.99f, 0.95f, 1.0f, 1.0f, 3);", "0.99f, 0.95f, 1.0f, 1.0f);")
    with tempfile.TemporaryDirectory(prefix="llb-vtrace-") as directory:
        executable = Path(directory) / "vtrace"
        subprocess.run(
            ["c++", "-std=c++20", "-x", "c++", "-", "-o", str(executable)],
            input=program, text=True, check=True,
        )
        subprocess.run([str(executable)], check=True)
