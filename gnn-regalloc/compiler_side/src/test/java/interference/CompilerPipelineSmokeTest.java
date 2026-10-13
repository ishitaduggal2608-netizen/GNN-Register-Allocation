package interference;

import allocator.ChaitinBriggsAllocator;
import cfg.ControlFlowGraph;
import cfg.Instruction;
import cfg.Variable;
import exporter.GraphJsonExporter;

import java.util.Set;

/** Executable checks for graph construction, spilling, coalescing, and JSON labels. */
public final class CompilerPipelineSmokeTest {
    private CompilerPipelineSmokeTest() { }

    public static void main(String[] args) {
        verifyPhaseOneCliqueAndSpill();
        verifyMoveCoalescingAndJson();
        System.out.println("CompilerPipelineSmokeTest passed");
    }

    private static void verifyPhaseOneCliqueAndSpill() {
        Variable a = new Variable("a");
        Variable b = new Variable("b");
        Variable c = new Variable("c");
        Variable d = new Variable("d");
        Variable e = new Variable("e");
        ControlFlowGraph cfg = new ControlFlowGraph();
        cfg.addInstruction(new Instruction(1, "a = 1", Set.of(), Set.of(a)));
        cfg.addInstruction(new Instruction(2, "b = 2", Set.of(), Set.of(b)));
        cfg.addInstruction(new Instruction(3, "c = a + b", Set.of(a, b), Set.of(c)));
        cfg.addInstruction(new Instruction(4, "d = a * c", Set.of(a, c), Set.of(d)));
        cfg.addInstruction(new Instruction(5, "e = b + d", Set.of(b, d), Set.of(e)));
        cfg.addInstruction(new Instruction(6, "return e", Set.of(e), Set.of()));
        for (int id = 1; id < 6; id++) cfg.addEdge(id, id + 1);

        InterferenceGraphBuildResult built = new InterferenceGraphBuilder().build(cfg);
        InterferenceGraph graph = built.getGraph();
        assertEquals(4, graph.getEdges().size(), "interference edge count");
        assertEquals(2, graph.getNode(built.getNodeIdsByVariable().get(a)).getDegree(), "a degree");
        var allocation = new ChaitinBriggsAllocator().allocate(graph, 2);
        assertTrue(allocation.isSpilled(built.getNodeIdsByVariable().get(c)), "c must spill in K3 with k=2");
        assertTrue(!allocation.isSpilled(built.getNodeIdsByVariable().get(e)), "isolated e must colour");
    }

    private static void verifyMoveCoalescingAndJson() {
        Variable x = new Variable("x");
        Variable y = new Variable("y");
        ControlFlowGraph cfg = new ControlFlowGraph();
        cfg.addInstruction(new Instruction(1, "x = 1", Set.of(), Set.of(x)));
        cfg.addInstruction(new Instruction(2, "y = x", Set.of(x), Set.of(y), true));
        cfg.addInstruction(new Instruction(3, "return y", Set.of(y), Set.of()));
        cfg.addEdge(1, 2);
        cfg.addEdge(2, 3);

        InterferenceGraph graph = new InterferenceGraphBuilder().build(cfg).getGraph();
        assertEquals(0, graph.getEdges().size(), "copy pair should not be an interference edge");
        assertEquals(1, graph.getMoveEdges().size(), "copy pair should be a move edge");
        var allocation = new ChaitinBriggsAllocator().allocate(graph, 2);
        Edge move = graph.getMoveEdges().get(0);
        assertTrue(allocation.isCoalesceSafe(move), "simple copy should coalesce conservatively");
        String json = new GraphJsonExporter().toJson("func_042", 2, graph, allocation);
        assertTrue(json.contains("\"interference_edges\": []"), "JSON interference schema");
        assertTrue(json.contains("\"coalesce_safe\": true"), "JSON move label schema");
        assertTrue(json.contains("\"label_register\":"), "JSON allocation label schema");
    }

    private static void assertEquals(int expected, int actual, String label) {
        if (expected != actual) throw new AssertionError(label + ": expected " + expected + ", got " + actual);
    }

    private static void assertTrue(boolean condition, String label) {
        if (!condition) throw new AssertionError(label);
    }
}
