"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useAtom, useAtomValue } from "jotai";
import { useElementBounding } from "@reactuses/core";
import Circle, { LAYOUT_ANIMATION_DURATION } from "./circle";
import {
  centersAtom,
  colorsAtom,
  hoveredNodeAtom,
  inputsAtom,
  maskAtom,
  outputsAtom,
  patternAtom,
  timedAnimationPatternAtom,
  type DependencyPattern,
} from "./atoms";

// Extract subscript number from value like "x₁" → 1
const getSubscript = (value: string): number => {
  const subscripts = "₀₁₂₃₄₅₆₇₈₉";
  for (let i = 0; i < value.length; i++) {
    const idx = subscripts.indexOf(value[i]!);
    if (idx !== -1) return idx;
  }
  return -1;
};

const getColorForValue = (value: string, colors: string[]): string => {
  const index = getSubscript(value);
  return colors[index] || colors[0] || "#f4d35e";
};

type ValueConnection = {
  inputValue: string;
  outputValue: string;
  inputIdx: number;
  outputIdx: number;
  inputMasked?: boolean;
};

const buildSubscriptLookup = (
  values: string[],
): Map<number, { value: string; idx: number }> =>
  new Map(values.map((value, idx) => [getSubscript(value), { value, idx }]));

// Get value-based connections (e.g., "x₁" connects to "y₁")
const getValueConnections = (
  inputs: string[],
  outputs: string[],
  pattern: DependencyPattern,
  mask?: boolean[],
): ValueConnection[] => {
  const connections: ValueConnection[] = [];

  // Build lookup from subscript to value
  const inputBySubscript = buildSubscriptLookup(inputs);
  const outputBySubscript = buildSubscriptLookup(outputs);

  switch (pattern) {
    case "mask-invariant":
      // For mask-invariant: unmasked outputs only connect to unmasked inputs
      // Masked outputs can connect to any input
      for (const [outSub, outInfo] of outputBySubscript) {
        const outputActive = mask ? mask[outSub - 1] : true;
        for (const [inSub, inInfo] of inputBySubscript) {
          const inputActive = mask ? mask[inSub - 1] : true;
          // Connection exists if: both active, OR output is masked
          if ((inputActive && outputActive) || !outputActive) {
            connections.push({
              inputValue: inInfo.value,
              outputValue: outInfo.value,
              inputIdx: inInfo.idx,
              outputIdx: outInfo.idx,
              inputMasked: !inputActive,
            });
          }
        }
      }
      break;
    case "lower-triangular":
      // For lower-triangular, output_i depends on inputs 0 to i
      for (const [outSub, outInfo] of outputBySubscript) {
        for (const [inSub, inInfo] of inputBySubscript) {
          if (inSub <= outSub) {
            connections.push({
              inputValue: inInfo.value,
              outputValue: outInfo.value,
              inputIdx: inInfo.idx,
              outputIdx: outInfo.idx,
            });
          }
        }
      }
      break;
    case "diagonal":
    default:
      // For diagonal, x_i connects to y_i (same subscript)
      for (const [sub, inInfo] of inputBySubscript) {
        const outInfo = outputBySubscript.get(sub);
        if (outInfo) {
          connections.push({
            inputValue: inInfo.value,
            outputValue: outInfo.value,
            inputIdx: inInfo.idx,
            outputIdx: outInfo.idx,
          });
        }
      }
      break;
  }

  return connections;
};

const isLineHighlighted = (
  inputIdx: number,
  outputIdx: number,
  hoveredNode: { index: number; type: "input" | "output" } | null,
  pattern: DependencyPattern,
): boolean => {
  if (hoveredNode === null) return false;

  const { index, type } = hoveredNode;

  switch (pattern) {
    case "mask-invariant":
      // For mask-invariant, highlight based on output hover only
      if (type === "output") {
        return outputIdx === index;
      }
      return false;
    case "lower-triangular":
      if (type === "output") {
        // Hovering output i: highlight lines from inputs 0 to i to output i
        return outputIdx === index && inputIdx <= index;
      } else {
        // Hovering input i: highlight lines from input i to outputs i to n-1
        return inputIdx === index && outputIdx >= index;
      }
    case "diagonal":
    default:
      // Highlight the line connecting input/output at hovered index
      return inputIdx === index && outputIdx === index;
  }
};

type ShuffleState = {
  inputs: string[];
  outputs: string[];
};

const shuffleIndices = (length: number): number[] => {
  const indices = Array.from({ length }, (_, i) => i);
  for (let i = length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j] as number, indices[i] as number];
  }
  return indices;
};

const scheduleLineReveal = (
  setLinesVisible: (visible: boolean) => void,
  delay: number,
) => {
  setTimeout(() => {
    setLinesVisible(true);
  }, delay);
};

const SHUFFLE_INTERVAL = 2000;

const InputOutput = () => {
  const [inputs, setInputs] = useAtom(inputsAtom);
  const [outputs, setOutputs] = useAtom(outputsAtom);
  const [mask, setMask] = useAtom(maskAtom);
  const colors = useAtomValue(colorsAtom);
  const pattern = useAtomValue(patternAtom);
  const timedAnimationPattern = useAtomValue(timedAnimationPatternAtom);
  const hoveredNode = useAtomValue(hoveredNodeAtom);
  const centers = useAtomValue(centersAtom);
  const containerRef = useRef<HTMLDivElement>(null);
  const containerBounding = useElementBounding(containerRef);
  const [linesVisible, setLinesVisible] = useState(true);

  // Use refs to access current values without causing effect re-runs
  const inputsRef = useRef(inputs);
  const outputsRef = useRef(outputs);
  useEffect(() => {
    inputsRef.current = inputs;
    outputsRef.current = outputs;
  }, [inputs, outputs]);

  // Ensure mask length matches inputs
  useEffect(() => {
    if (mask.length !== inputs.length) {
      setMask(Array.from({ length: inputs.length }, (_, i) => i < 3));
    }
  }, [inputs.length, mask.length, setMask]);

  // Toggle mask for an input (only in mask-invariant mode)
  const toggleMask = (idx: number) => {
    if (pattern !== "mask-invariant") return;
    setMask((prev) => {
      const newMask = [...prev];
      newMask[idx] = !newMask[idx];
      return newMask;
    });
  };

  // Check if an input is relevant based on hovered output in mask-invariant mode
  const isInputRelevant = (inputIdx: number): boolean => {
    if (pattern !== "mask-invariant") return true;
    if (hoveredNode === null || hoveredNode.type !== "output") return true;
    const outputIdx = hoveredNode.index;
    // If hovered output is masked, all inputs are relevant
    if (!mask[outputIdx]) return true;
    // If hovered output is unmasked, only unmasked inputs are relevant
    return mask[inputIdx] ?? true;
  };

  // Check if an output is relevant based on hover
  const isOutputRelevant = (outputIdx: number): boolean => {
    if (pattern !== "mask-invariant") return true;
    if (hoveredNode === null || hoveredNode.type !== "output") return true;
    return hoveredNode.index === outputIdx;
  };

  useEffect(() => {
    const scheduleShuffle = (
      shuffle: (state: ShuffleState) => ShuffleState,
    ) => {
      const interval = setInterval(() => {
        setLinesVisible(false);

        setTimeout(() => {
          const currentInputs = inputsRef.current;
          const currentOutputs = outputsRef.current;
          const indices = shuffleIndices(currentInputs.length);

          const shuffledInputs = indices.map((i) => currentInputs[i]!);
          const shuffledOutputs = indices.map((i) => currentOutputs[i]!);

          const nextState = shuffle({
            inputs: shuffledInputs,
            outputs: shuffledOutputs,
          });
          setInputs(nextState.inputs);
          setOutputs(nextState.outputs);

          scheduleLineReveal(setLinesVisible, LAYOUT_ANIMATION_DURATION + 50);
        }, 150);
      }, SHUFFLE_INTERVAL);

      return () => clearInterval(interval);
    };

    if (timedAnimationPattern === "permutation-equivariant") {
      return scheduleShuffle(
        ({ inputs: nextInputs, outputs: nextOutputs }) => ({
          inputs: nextInputs,
          outputs: nextOutputs,
        }),
      );
    }

    if (timedAnimationPattern === "permutation-invariant") {
      return scheduleShuffle(({ inputs: nextInputs }) => ({
        inputs: nextInputs,
        outputs: outputsRef.current,
      }));
    }
  }, [timedAnimationPattern, setInputs, setOutputs]);

  const connections = useMemo(
    () => getValueConnections(inputs, outputs, pattern, mask),
    [inputs, outputs, pattern, mask],
  );

  const inputItems = useMemo(
    () =>
      inputs.map((input, i) => ({
        value: input,
        index: i,
        color: getColorForValue(input, colors),
        masked: pattern === "mask-invariant" && !mask[i],
        dimmed: pattern === "mask-invariant" && !isInputRelevant(i),
      })),
    [inputs, colors, pattern, mask, hoveredNode],
  );

  const outputItems = useMemo(
    () =>
      outputs.map((output, i) => ({
        value: output,
        index: i,
        color: getColorForValue(output, colors),
        masked: pattern === "mask-invariant" && !mask[i],
        dimmed: pattern === "mask-invariant" && !isOutputRelevant(i),
      })),
    [outputs, colors, pattern, mask, hoveredNode],
  );

  const lineHighlights = useMemo(
    () =>
      new Map(
        connections.map((connection) => [
          `${connection.inputValue}-${connection.outputValue}`,
          isLineHighlighted(
            connection.inputIdx,
            connection.outputIdx,
            hoveredNode,
            pattern,
          ),
        ]),
      ),
    [connections, hoveredNode, pattern],
  );

  return (
    <div className="flex-1">
      <div ref={containerRef} className="grid grid-cols-2 gap-4 relative">
        <div className="col-span-1 flex flex-col gap-6 justify-between items-center z-10">
          <div className="text-sm font-sans-tight uppercase">Inputs</div>
          <div className="flex flex-col gap-4">
            {inputItems.map((item) => (
              <Circle
                key={item.value}
                index={item.index}
                value={item.value}
                type="input"
                color={item.color}
                isMasked={item.masked}
                dimmed={item.dimmed}
                onClick={
                  pattern === "mask-invariant"
                    ? () => toggleMask(item.index)
                    : undefined
                }
              >
                {item.value}
              </Circle>
            ))}
          </div>
          {pattern === "mask-invariant" && (
            <div className="text-xs text-muted-foreground uppercase font-light">
              click to toggle
            </div>
          )}
        </div>
        <div className="col-span-1 flex flex-col gap-6 justify-between items-center z-10">
          <div className="text-sm font-sans-tight uppercase">Outputs</div>
          <div className="flex flex-col gap-4">
            {outputItems.map((item) => (
              <Circle
                key={item.value}
                index={item.index}
                value={item.value}
                type="output"
                color={item.color}
                isMasked={item.masked}
                dimmed={item.dimmed}
              >
                {item.value}
              </Circle>
            ))}
          </div>
          {pattern === "mask-invariant" && (
            <div className="text-xs text-muted-foreground font-light uppercase">
              Hover to inspect
            </div>
          )}
        </div>
        <svg
          className="absolute top-0 left-0 w-full h-full pointer-events-none z-9 transition-opacity duration-150"
          style={{ opacity: linesVisible ? 1 : 0 }}
        >
          {connections.map((connection) => {
            const { inputValue, outputValue, inputIdx, inputMasked } =
              connection;
            const inputCenter = centers[inputValue];
            const outputCenter = centers[outputValue];
            if (!inputCenter || !outputCenter) return null;

            const startX = inputCenter.x - containerBounding.left;
            const startY = inputCenter.y - containerBounding.top;
            const endX = outputCenter.x - containerBounding.left;
            const endY = outputCenter.y - containerBounding.top;

            const key = `${inputValue}-${outputValue}`;
            const highlighted = lineHighlights.get(key) ?? false;

            // For mask-invariant, use input color or gray for masked inputs
            let lineColor: string | undefined;
            if (pattern === "mask-invariant") {
              if (highlighted) {
                lineColor = inputMasked
                  ? "#9ca3af"
                  : getColorForValue(inputValue, colors);
              }
            } else {
              lineColor = highlighted
                ? colors[hoveredNode?.index ?? 0] || "#f4d35e"
                : undefined;
            }

            const opacity =
              pattern === "mask-invariant" ? (highlighted ? 0.7 : 0.15) : 1;

            return (
              <line
                key={key}
                x1={startX}
                y1={startY}
                x2={endX}
                y2={endY}
                stroke={lineColor || "currentColor"}
                strokeWidth={highlighted ? 2 : 1}
                strokeDasharray={inputMasked ? "4 4" : "none"}
                opacity={opacity}
                className={highlighted ? "" : "text-gray-300"}
                style={{
                  transition: "stroke 0.2s, stroke-width 0.2s, opacity 0.2s",
                }}
              />
            );
          })}
        </svg>
      </div>
    </div>
  );
};

export default InputOutput;
