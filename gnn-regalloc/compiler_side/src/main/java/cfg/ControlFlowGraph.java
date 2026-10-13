package cfg;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Directed control-flow graph over TAC instructions. */
public final class ControlFlowGraph {
    private final Map<Integer, Instruction> instructionsById = new LinkedHashMap<>();
    private final Map<Instruction, Set<Instruction>> successors = new LinkedHashMap<>();
    private final Map<Instruction, Integer> loopDepths = new LinkedHashMap<>();
    private Instruction entry;

    public void addInstruction(Instruction instruction) {
        Objects.requireNonNull(instruction, "instruction");
        if (instructionsById.putIfAbsent(instruction.getId(), instruction) != null) {
            throw new IllegalArgumentException("Duplicate instruction id: " + instruction.getId());
        }
        successors.put(instruction, new LinkedHashSet<>());
        loopDepths.put(instruction, 0);
        if (entry == null) entry = instruction;
    }

    public void setEntry(int instructionId) { entry = requireInstruction(instructionId); }

    public void addEdge(int fromId, int toId) {
        successors.get(requireInstruction(fromId)).add(requireInstruction(toId));
    }

    public void setLoopDepth(int instructionId, int loopDepth) {
        if (loopDepth < 0) throw new IllegalArgumentException("Loop depth must be non-negative");
        loopDepths.put(requireInstruction(instructionId), loopDepth);
    }

    public int loopDepth(Instruction instruction) {
        requireKnown(instruction);
        return loopDepths.get(instruction);
    }

    public List<Instruction> instructions() {
        return List.copyOf(instructionsById.values());
    }

    public Set<Instruction> successors(Instruction instruction) {
        requireKnown(instruction);
        return Collections.unmodifiableSet(successors.get(instruction));
    }

    /**
     * DFS reverse postorder, starting at the entry then including unreachable
     * instructions in insertion order.  The fixed-point algorithm is correct
     * for any order; this deterministic order typically converges quickly.
     */
    public List<Instruction> reversePostOrder() {
        List<Instruction> postorder = new ArrayList<>();
        Set<Instruction> visited = new HashSet<>();
        if (entry != null) visit(entry, visited, postorder);
        for (Instruction instruction : instructionsById.values()) {
            visit(instruction, visited, postorder);
        }
        Collections.reverse(postorder);
        return List.copyOf(postorder);
    }

    private void visit(Instruction instruction, Set<Instruction> visited, List<Instruction> postorder) {
        if (!visited.add(instruction)) return;
        for (Instruction successor : successors.get(instruction)) {
            visit(successor, visited, postorder);
        }
        postorder.add(instruction);
    }

    private Instruction requireInstruction(int id) {
        Instruction instruction = instructionsById.get(id);
        if (instruction == null) throw new IllegalArgumentException("Unknown instruction id: " + id);
        return instruction;
    }

    private void requireKnown(Instruction instruction) {
        if (!successors.containsKey(instruction)) {
            throw new IllegalArgumentException("Instruction is not in this CFG: " + instruction);
        }
    }
}
