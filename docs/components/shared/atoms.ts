import { atom } from "jotai";
import chroma from "chroma-js";
import { Poline, positionFunctions } from "poline";
import { DEFAULT_INPUTS, DEFAULT_OUTPUTS } from "@/lib/constants";

// Generate colors using poline with two anchor colors
const generateColors = (count: number): string[] => {
  const anchorColors: [number, number, number][] = [
    chroma.hex("#f4d35e").hsl(),
    chroma.hex("#ff3434").hsl(),
  ];

  const poline = new Poline({
    anchorColors,
    numPoints: count,
    positionFunctionX: positionFunctions.quadraticPosition,
    positionFunctionY: positionFunctions.sinusoidalPosition,
    positionFunctionZ: positionFunctions.exponentialPosition,
    closedLoop: true,
    invertedLightness: true,
  });

  return poline.colorsCSS;
};

// Dependency pattern type
export type DependencyPattern =
  | "diagonal"
  | "lower-triangular"
  | "mask-invariant";

export type TimedAnimationPattern =
  | "permutation-equivariant"
  | "permutation-invariant"
  | null;

// Base atoms
export const inputsAtom = atom<string[]>(DEFAULT_INPUTS);
export const outputsAtom = atom<string[]>(DEFAULT_OUTPUTS);
export const patternAtom = atom<DependencyPattern>("diagonal");
export const timedAnimationPatternAtom = atom<TimedAnimationPattern>(null);
export const hoveredNodeAtom = atom<{
  index: number;
  type: "input" | "output";
} | null>(null);

// Mask atom for mask-invariant pattern (true = active/unmasked, false = masked)
export const maskAtom = atom<boolean[]>([true, true, true, false, false]);

// Derived atom for max length
export const maxLengthAtom = atom((get) => {
  const inputs = get(inputsAtom);
  const outputs = get(outputsAtom);
  return Math.max(inputs.length, outputs.length);
});

// Derived atom for colors array
export const colorsAtom = atom((get) => {
  const maxLength = get(maxLengthAtom);
  return generateColors(maxLength);
});

// Derived atom for total length (inputs + outputs)
export const totalLengthAtom = atom((get) => {
  const inputs = get(inputsAtom);
  const outputs = get(outputsAtom);
  return inputs.length + outputs.length;
});

// Writable atom for circle centers (keyed by element value, e.g., "x₁" or "y₁")
export const centersAtom = atom<Record<string, { x: number; y: number }>>({});
