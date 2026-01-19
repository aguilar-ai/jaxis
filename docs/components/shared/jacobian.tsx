"use client";

import clsx from "clsx";
import { useMemo } from "react";
import { useAtomValue } from "jotai";
import {
  colorsAtom,
  hoveredNodeAtom,
  inputsAtom,
  maskAtom,
  outputsAtom,
  patternAtom,
  type DependencyPattern,
} from "./atoms";

interface CellProps {
  x: number;
  y: number;
  value: string;
  forceZero?: boolean;
  inputColor?: string;
}

const Cell = ({ x, y, value, forceZero, inputColor }: CellProps) => {
  const hoveredNode = useAtomValue(hoveredNodeAtom);
  const colors = useAtomValue(colorsAtom);
  const pattern = useAtomValue(patternAtom);

  // Determine if this cell should be highlighted based on pattern
  const isHighlighted = (): boolean => {
    if (hoveredNode === null) return false;

    const { index, type } = hoveredNode;

    switch (pattern) {
      case "mask-invariant":
        // For mask-invariant, highlight row when hovering output
        if (type === "output") {
          return y === index;
        }
        return false;
      case "lower-triangular":
        if (type === "output") {
          // Hovering output i: highlight row i, columns 0 to i (inputs this output depends on)
          return y === index && x <= index;
        } else {
          // Hovering input i: highlight column i, rows i to n-1 (outputs that depend on this input)
          return x === index && y >= index;
        }
      case "diagonal":
      default:
        // Hovering input/output i: highlight row i and column i
        return x === index || y === index;
    }
  };

  const hovered = isHighlighted();

  // For mask-invariant, use the input's color for non-zero cells
  let hoverColor: string | null = null;
  if (hovered && hoveredNode !== null) {
    if (pattern === "mask-invariant" && inputColor && !forceZero) {
      hoverColor = inputColor;
    } else if (!forceZero) {
      hoverColor = colors[hoveredNode.index] || colors[0] || "#f4d35e";
    }
  }

  return (
    <div
      style={
        hovered && hoverColor
          ? ({ "--hover-bg": hoverColor } as React.CSSProperties)
          : forceZero
            ? { backgroundColor: "rgb(243 244 246)" }
            : undefined
      }
      className={clsx(
        [
          "col-span-1",
          "row-span-1",
          "flex",
          "items-center",
          "justify-center",
          "border",
          "border-gray-300",
          "rounded-md",
          "transition-all",
          "duration-200",
        ],
        hovered && hoverColor
          ? "bg-(--hover-bg) text-white opacity-100"
          : "opacity-70",
        forceZero && "text-gray-400",
        !forceZero && !hovered && "text-muted-foreground",
      )}
    >
      {value}
    </div>
  );
};

const isNonZero = (
  inputIdx: number,
  outputIdx: number,
  pattern: DependencyPattern,
  mask?: boolean[],
): boolean => {
  switch (pattern) {
    case "mask-invariant":
      // For unmasked outputs, masked inputs must have zero gradient
      if (mask) {
        const outputActive = mask[outputIdx] ?? false;
        const inputActive = mask[inputIdx] ?? false;
        return outputActive ? inputActive : true;
      }
      return true;
    case "lower-triangular":
      // Each output depends on all previous inputs (including its own index)
      return inputIdx <= outputIdx;
    case "diagonal":
    default:
      // Each output depends only on its corresponding input
      return inputIdx === outputIdx;
  }
};

const Jacobian = () => {
  const inputs = useAtomValue(inputsAtom);
  const outputs = useAtomValue(outputsAtom);
  const pattern = useAtomValue(patternAtom);
  const mask = useAtomValue(maskAtom);
  const colors = useAtomValue(colorsAtom);
  const x = inputs.length;
  const y = outputs.length;

  const gridStyle = useMemo(
    () => ({
      gridTemplateColumns: `repeat(${x}, 1fr)`,
      gridTemplateRows: `repeat(${y}, 1fr)`,
    }),
    [x, y],
  );

  const cells = useMemo(() => {
    const nextCells: React.ReactNode[] = [];
    for (let outputIdx = 0; outputIdx < y; outputIdx++) {
      for (let inputIdx = 0; inputIdx < x; inputIdx++) {
        const canBeNonZero = Boolean(
          isNonZero(inputIdx, outputIdx, pattern, mask),
        );
        const value = canBeNonZero ? "∂" : "0";
        nextCells.push(
          <Cell
            key={`${inputIdx}-${outputIdx}`}
            x={inputIdx}
            y={outputIdx}
            value={value}
            forceZero={!canBeNonZero}
            inputColor={
              pattern === "mask-invariant" ? colors[inputIdx] : undefined
            }
          />,
        );
      }
    }
    return nextCells;
  }, [x, y, pattern, mask, colors]);

  const legendItems = useMemo(
    () =>
      inputs.map((_, index) => {
        const active = mask[index] ?? false;
        return {
          key: index,
          color: active ? colors[index] : "#9ca3af",
          symbol: active ? "●" : "○",
        };
      }),
    [inputs, mask, colors],
  );

  return (
    <div className="flex flex-col gap-4 flex-1">
      <div className="uppercase font-sans-tight text-sm">Jacobian ∂y/∂x</div>
      <div className="grid gap-2 flex-1" style={gridStyle}>
        {cells}
      </div>
      {pattern === "mask-invariant" && (
        <div className="flex gap-2 justify-center">
          {legendItems.map((item) => (
            <div
              key={item.key}
              className="w-8 text-center text-xs font-mono"
              style={{ color: item.color }}
            >
              {item.symbol}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Jacobian;
