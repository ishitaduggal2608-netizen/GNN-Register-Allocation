package liveness;

import cfg.ControlFlowGraph;
import cfg.Instruction;
import cfg.Variable;

import java.util.Set;

/** Executable check for the Phase 1 straight-line TAC example. */
public final class LivenessAnalyzerSmokeTest {
    private LivenessAnalyzerSmokeTest() { }

    public static void main(String[] args) {
        Variable a = new Variable("a");
        Variable b = new Variable("b");
        Variable c = new Variable("c");
        Variable d = new Variable("d");
        Variable e = new Variable("e");

        Instruction one = new Instruction(1, "a = 1", Set.of(), Set.of(a));
        Instruction two = new Instruction(2, "b = 2", Set.of(), Set.of(b));
        Instruction three = new Instruction(3, "c = a + b", Set.of(a, b), Set.of(c));
        Instruction four = new Instruction(4, "d = a * c", Set.of(a, c), Set.of(d));
        Instruction five = new Instruction(5, "e = b + d", Set.of(b, d), Set.of(e));
        Instruction six = new Instruction(6, "return e", Set.of(e), Set.of());

        ControlFlowGraph cfg = new ControlFlowGraph();
        for (Instruction instruction : Set.of(one, two, three, four, five, six)) {
            cfg.addInstruction(instruction);
        }
        cfg.setEntry(1);
        cfg.addEdge(1, 2);
        cfg.addEdge(2, 3);
        cfg.addEdge(3, 4);
        cfg.addEdge(4, 5);
        cfg.addEdge(5, 6);

        LivenessAnalyzer analyzer = new LivenessAnalyzer();
        analyzer.run(cfg);

        assertSet(Set.of(), analyzer.liveIn(one), "line 1 live-in");
        assertSet(Set.of(a), analyzer.liveOut(one), "line 1 live-out");
        assertSet(Set.of(a, b), analyzer.liveOut(two), "line 2 live-out");
        assertSet(Set.of(a, b, c), analyzer.liveIn(four), "line 4 live-in");
        assertSet(Set.of(b, d), analyzer.liveOut(four), "line 4 live-out");
        assertSet(Set.of(e), analyzer.liveOut(five), "line 5 live-out");
        assertSet(Set.of(e), analyzer.liveIn(six), "line 6 live-in");
        assertSet(Set.of(), analyzer.liveOut(six), "line 6 live-out");

        System.out.println("LivenessAnalyzerSmokeTest passed");
    }

    private static void assertSet(Set<Variable> expected, Set<Variable> actual, String label) {
        if (!expected.equals(actual)) {
            throw new AssertionError(label + ": expected " + expected + ", got " + actual);
        }
    }
}
