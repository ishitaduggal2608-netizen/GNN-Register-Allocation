package liveness;

import cfg.ControlFlowGraph;
import cfg.Instruction;
import cfg.Variable;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/** Computes live-in and live-out sets using the standard backward fixed point. */
public final class LivenessAnalyzer {
    private final Map<Instruction, Set<Variable>> liveIn = new HashMap<>();
    private final Map<Instruction, Set<Variable>> liveOut = new HashMap<>();

    public void run(ControlFlowGraph cfg) {
        liveIn.clear();
        liveOut.clear();

        boolean changed = true;
        while (changed) {
            changed = false;
            for (Instruction instr : cfg.reversePostOrder()) {
                Set<Variable> newOut = new HashSet<>();
                for (Instruction successor : cfg.successors(instr)) {
                    newOut.addAll(liveIn.getOrDefault(successor, Set.of()));
                }

                Set<Variable> newIn = new HashSet<>(instr.uses());
                Set<Variable> outMinusDef = new HashSet<>(newOut);
                outMinusDef.removeAll(instr.defs());
                newIn.addAll(outMinusDef);

                if (!newIn.equals(liveIn.get(instr)) || !newOut.equals(liveOut.get(instr))) {
                    changed = true;
                }
                liveIn.put(instr, newIn);
                liveOut.put(instr, newOut);
            }
        }
    }

    public Set<Variable> liveIn(Instruction instruction) {
        return Collections.unmodifiableSet(liveIn.getOrDefault(instruction, Set.of()));
    }

    public Set<Variable> liveOut(Instruction instruction) {
        return Collections.unmodifiableSet(liveOut.getOrDefault(instruction, Set.of()));
    }

    public Map<Instruction, Set<Variable>> liveInByInstruction() {
        return immutableCopy(liveIn);
    }

    public Map<Instruction, Set<Variable>> liveOutByInstruction() {
        return immutableCopy(liveOut);
    }

    private Map<Instruction, Set<Variable>> immutableCopy(Map<Instruction, Set<Variable>> source) {
        Map<Instruction, Set<Variable>> result = new LinkedHashMap<>();
        source.forEach((instruction, variables) -> result.put(instruction, Set.copyOf(variables)));
        return Collections.unmodifiableMap(result);
    }
}
