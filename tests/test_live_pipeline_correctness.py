
import unittest

from web_server import build_result
from tac_parser import parse_tac
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph


class TestLivePipelineCorrectness(unittest.TestCase):

    def test_all_allocators_respect_interference_constraints(self):
        source = """
        a = 1
        b = 2
        c = a + b
        d = c + a
        e = d + b
        """

        models = [
            "consensus",
            "rgcn",
            "gat",
            "sage",
            "gcn",
            "cb",
            "random",
        ]

        # Build the same interference graph used by the backend.
        program = parse_tac(source)
        cfg = ControlFlowGraph(program)
        liveness = LivenessAnalyzer(cfg)
        ig = InterferenceGraph(program, cfg, liveness)

        variables = [v.name for v in ig.variables]
        edges = ig.interference_edges

        for k in (1, 2, 3, 4):
            with self.subTest(register_count=k):
                result = build_result(source, k)
                assignments_by_model = result["assignments_by_model"]

                for model in models:
                    with self.subTest(k=k, model=model):
                        self.assertIn(model, assignments_by_model)
                        assignments = assignments_by_model[model]

                        self.assertEqual(
                            set(assignments.keys()),
                            set(variables),
                            f"Unexpected variables for {model} at K={k}",
                        )

                        for variable in variables:
                            assignment = assignments[variable]

                            self.assertTrue(
                                assignment == "SPILL"
                                or (
                                    assignment.startswith("R")
                                    and assignment[1:].isdigit()
                                    and 0 <= int(assignment[1:]) < k
                                ),
                                f"Invalid assignment {assignment!r} "
                                f"for {variable} in {model} at K={k}",
                            )

                        for u, v in edges:
                            a_u = assignments[u]
                            a_v = assignments[v]

                            self.assertTrue(
                                a_u == "SPILL"
                                or a_v == "SPILL"
                                or a_u != a_v,
                                f"Interference violation: {u} and {v} "
                                f"both assigned {a_u} in {model} at K={k}",
                            )


if __name__ == "__main__":
    unittest.main()
