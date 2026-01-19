"use client";

import { useSetAtom } from "jotai";
import { useEffect } from "react";
import InputOutput from "./shared/input-output";
import {
  inputsAtom,
  outputsAtom,
  timedAnimationPatternAtom,
} from "./shared/atoms";
import { DEFAULT_INPUTS, DEFAULT_OUTPUTS } from "@/lib/constants";

export default function PermutationEquivariant() {
  const setTimedAnimationPattern = useSetAtom(timedAnimationPatternAtom);
  const setInputs = useSetAtom(inputsAtom);
  const setOutputs = useSetAtom(outputsAtom);

  useEffect(() => {
    // Reset to default order when component mounts
    setInputs(DEFAULT_INPUTS);
    setOutputs(DEFAULT_OUTPUTS);
    setTimedAnimationPattern("permutation-equivariant");
    return () => setTimedAnimationPattern(null);
  }, [setTimedAnimationPattern, setInputs, setOutputs]);

  return (
    <div className="flex flex-col gap-2 my-8">
      <InputOutput />
    </div>
  );
}
