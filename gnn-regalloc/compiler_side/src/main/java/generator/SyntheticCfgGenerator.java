package generator;

import cfg.ControlFlowGraph;
import cfg.Instruction;
import cfg.Variable;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/** Creates sparse synthetic TAC programs while retaining real CFG/liveness semantics. */
public final class SyntheticCfgGenerator {
    private final Random random;

    public SyntheticCfgGenerator(Random random) { this.random = random; }

    public ControlFlowGraph generate(int variableCount) {
        if (variableCount < 1) throw new IllegalArgumentException("variableCount must be positive");
        List<Variable> variables = java.util.stream.IntStream.range(0, variableCount)
                .mapToObj(index -> new Variable("v" + index)).toList();
        ControlFlowGraph cfg = new ControlFlowGraph();

        for (int index = 0; index < variableCount; index++) {
            Variable destination = variables.get(index);
            Set<Variable> uses = new LinkedHashSet<>();
            boolean isMove = index > 0 && random.nextDouble() < 0.12;
            if (index > 0) {
                uses.add(selectSource(variables, index));
                if (!isMove && index > 1 && random.nextDouble() < 0.45) {
                    uses.add(selectSource(variables, index));
                }
            }
            String text = isMove ? destination + " = " + uses.iterator().next()
                    : destination + " = op" + index + uses;
            cfg.addInstruction(new Instruction(index, text, uses, Set.of(destination), isMove));
        }

        int returnId = variableCount;
        cfg.addInstruction(new Instruction(returnId, "return " + variables.get(variableCount - 1),
                Set.of(variables.get(variableCount - 1)), Set.of()));
        for (int index = 0; index < returnId; index++) cfg.addEdge(index, index + 1);

        // Forward alternatives model branches; sparse back-edges model loops.
        for (int index = 0; index < variableCount - 2; index++) {
            if (random.nextDouble() < 0.10) {
                int target = Math.min(returnId, index + 2 + random.nextInt(Math.min(8, variableCount - index - 1)));
                cfg.addEdge(index, target);
            }
            if (index >= 3 && random.nextDouble() < 0.05) {
                int target = index - 2 - random.nextInt(Math.min(6, index - 1));
                cfg.addEdge(index, target);
                for (int loopInstruction = target; loopInstruction <= index; loopInstruction++) {
                    cfg.setLoopDepth(loopInstruction, Math.max(1, cfg.loopDepth(cfg.instructions().get(loopInstruction))));
                }
            }
        }
        return cfg;
    }

    private Variable selectSource(List<Variable> variables, int upperExclusive) {
        if (random.nextDouble() < 0.72) {
            int distance = 1 + random.nextInt(Math.min(6, upperExclusive));
            return variables.get(upperExclusive - distance);
        }
        return variables.get(random.nextInt(upperExclusive));
    }
}
