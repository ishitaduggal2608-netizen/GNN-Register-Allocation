package interference;

import cfg.ControlFlowGraph;
import cfg.Instruction;
import cfg.Variable;
import liveness.LivenessAnalyzer;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/** Builds interference and move-edge relations from converged liveness sets. */
public final class InterferenceGraphBuilder {
    public InterferenceGraphBuildResult build(ControlFlowGraph cfg) {
        LivenessAnalyzer liveness = new LivenessAnalyzer();
        liveness.run(cfg);
        return build(cfg, liveness);
    }

    public InterferenceGraphBuildResult build(ControlFlowGraph cfg, LivenessAnalyzer liveness) {
        Set<Variable> variables = new LinkedHashSet<>();
        for (Instruction instruction : cfg.instructions()) {
            variables.addAll(instruction.uses());
            variables.addAll(instruction.defs());
            variables.addAll(liveness.liveIn(instruction));
            variables.addAll(liveness.liveOut(instruction));
        }

        InterferenceGraph graph = new InterferenceGraph();
        Map<Variable, Integer> nodeIds = new LinkedHashMap<>();
        variables.stream().sorted(Comparator.comparing(Variable::getName)).forEach(variable -> {
            int id = nodeIds.size();
            nodeIds.put(variable, id);
            graph.addNode(id, liveRangeLength(variable, cfg, liveness),
                    maximumLoopDepth(variable, cfg, liveness), useDefCount(variable, cfg));
        });

        for (Instruction instruction : cfg.instructions()) {
            Variable moveSource = instruction.isMoveInstruction()
                    ? instruction.uses().iterator().next() : null;
            for (Variable definition : instruction.defs()) {
                for (Variable liveOutVariable : liveness.liveOut(instruction)) {
                    if (definition.equals(liveOutVariable)) continue;
                    if (moveSource != null && liveOutVariable.equals(moveSource)) {
                        graph.addMoveEdge(nodeIds.get(definition), nodeIds.get(moveSource));
                    } else {
                        graph.addInterferenceEdge(nodeIds.get(definition), nodeIds.get(liveOutVariable));
                    }
                }
                // A copy remains a coalescing candidate even if its source dies at this point.
                if (moveSource != null && !definition.equals(moveSource)) {
                    graph.addMoveEdge(nodeIds.get(definition), nodeIds.get(moveSource));
                }
            }
        }
        return new InterferenceGraphBuildResult(graph, nodeIds);
    }

    private int liveRangeLength(Variable variable, ControlFlowGraph cfg, LivenessAnalyzer liveness) {
        int length = 0;
        for (Instruction instruction : cfg.instructions()) {
            if (instruction.uses().contains(variable) || instruction.defs().contains(variable)
                    || liveness.liveIn(instruction).contains(variable)
                    || liveness.liveOut(instruction).contains(variable)) {
                length++;
            }
        }
        return length;
    }

    private int maximumLoopDepth(Variable variable, ControlFlowGraph cfg, LivenessAnalyzer liveness) {
        int depth = 0;
        for (Instruction instruction : cfg.instructions()) {
            if (instruction.uses().contains(variable) || instruction.defs().contains(variable)
                    || liveness.liveIn(instruction).contains(variable)
                    || liveness.liveOut(instruction).contains(variable)) {
                depth = Math.max(depth, cfg.loopDepth(instruction));
            }
        }
        return depth;
    }

    private int useDefCount(Variable variable, ControlFlowGraph cfg) {
        int count = 0;
        for (Instruction instruction : cfg.instructions()) {
            if (instruction.uses().contains(variable)) count++;
            if (instruction.defs().contains(variable)) count++;
        }
        return count;
    }
}
